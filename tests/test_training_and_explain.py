import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from realm_eeg.adaptation import adapt_to_patient
from realm_eeg.config import ModelConfig
from realm_eeg.data import Episode
from realm_eeg.explain import explain_prediction, fit_centroids
from realm_eeg.model import RealmModel
from realm_eeg.training import (
    load_checkpoint,
    maml_episode_loss,
    save_checkpoint,
    train_stage1_epoch,
    train_stage2_epoch,
)


class TrainingAndExplainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = RealmModel(
            ModelConfig(
                channels=2,
                conv_width=8,
                latent_dim=8,
                transformer_layers=1,
                attention_heads=2,
                feedforward_dim=16,
                dropout=0.0,
                patient_classes=2,
            )
        )
        self.x = torch.randn(8, 2, 64)
        self.y = torch.tensor([0.0, 1.0] * 4)
        self.patient = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])

    def test_stage_training_and_checkpoint(self):
        loader = DataLoader(TensorDataset(self.x, self.y, self.patient), batch_size=4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.assertGreater(train_stage1_epoch(self.model, loader, optimizer)["loss"], 0)
        stage2 = train_stage2_epoch(self.model, loader, optimizer, 0, 1)
        self.assertGreater(stage2["patient_loss"], 0)
        self.assertEqual(stage2["grl_coefficient_start"], 0.0)
        self.assertGreater(stage2["grl_coefficient"], 0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model, "stage2", {"synthetic_only": True})
            loaded, payload = load_checkpoint(path)
            self.assertEqual(payload["stage"], "stage2")
            torch.testing.assert_close(
                loaded.state_dict()["seizure_head.weight"],
                self.model.state_dict()["seizure_head.weight"],
            )
            with self.assertRaises(TypeError):
                save_checkpoint(path, self.model, "stage2", {"unsafe": Path("local")})

    def test_maml_and_adaptation_do_not_mutate_source(self):
        episode = Episode(self.x[:2], self.y[:2], self.x[2:6], self.y[2:6], 0)
        loss = maml_episode_loss(self.model, episode, inner_steps=1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        original = copy.deepcopy(self.model.state_dict())
        adapted = adapt_to_patient(self.model, episode.support_x, episode.support_y, steps=2)
        for name, value in self.model.state_dict().items():
            torch.testing.assert_close(value, original[name])
        self.assertIsNot(adapted, self.model)

    def test_explanation_bundle_shapes(self):
        with torch.no_grad():
            embeddings = self.model(self.x).latent
        centroids = fit_centroids(embeddings, self.y)
        explanation = explain_prediction(self.model, self.x[:2], centroids, integration_steps=4)
        self.assertEqual(explanation.input_attribution.shape, self.x[:2].shape)
        self.assertEqual(tuple(explanation.seizure_probability.shape), (2,))
        self.assertIsNotNone(explanation.distance_to_seizure)
        self.assertTrue(self.model.training)


if __name__ == "__main__":
    unittest.main()
