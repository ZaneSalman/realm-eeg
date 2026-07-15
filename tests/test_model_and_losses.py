import unittest

import torch

from realm_eeg.config import ModelConfig
from realm_eeg.losses import dann_coefficient, stage1_loss, stage2_loss
from realm_eeg.model import RealmModel, gradient_reverse, sinusoidal_position_encoding


class ModelAndLossTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            channels=4,
            conv_width=8,
            latent_dim=12,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.0,
            patient_classes=3,
        )

    def test_forward_shapes_and_attention(self):
        model = RealmModel(self.config)
        output = model(torch.randn(3, 4, 64), grl_coefficient=0.5, return_attention=True)
        self.assertEqual(tuple(output.seizure_logits.shape), (3,))
        self.assertEqual(tuple(output.patient_logits.shape), (3, 3))
        self.assertEqual(tuple(output.latent.shape), (3, 12))
        self.assertEqual(len(output.attention), 1)
        with self.assertRaises(ValueError):
            model(torch.randn(3, 3, 64))

    def test_gradient_reversal_sign_and_scale(self):
        values = torch.tensor([1.0, -2.0], requires_grad=True)
        gradient_reverse(values, 0.4).sum().backward()
        torch.testing.assert_close(values.grad, torch.full_like(values, -0.4))

    def test_patient_head_minimizes_ce_while_backbone_gradient_reverses(self):
        coefficient = 0.35
        reference_latent = torch.randn(5, 4, requires_grad=True)
        reversed_latent = reference_latent.detach().clone().requires_grad_(True)
        labels = torch.tensor([0, 1, 2, 0, 1])
        reference_head = torch.nn.Linear(4, 3)
        reversed_head = torch.nn.Linear(4, 3)
        reversed_head.load_state_dict(reference_head.state_dict())

        reference_loss = torch.nn.functional.cross_entropy(reference_head(reference_latent), labels)
        reversed_loss = torch.nn.functional.cross_entropy(
            reversed_head(gradient_reverse(reversed_latent, coefficient)), labels
        )
        reference_loss.backward()
        reversed_loss.backward()

        torch.testing.assert_close(reversed_head.weight.grad, reference_head.weight.grad)
        torch.testing.assert_close(reversed_head.bias.grad, reference_head.bias.grad)
        torch.testing.assert_close(
            reversed_latent.grad,
            -coefficient * reference_latent.grad,
        )

    def test_temporal_tokens_receive_distinct_positions(self):
        positions = sinusoidal_position_encoding(torch.zeros(1, 4, 8))
        self.assertEqual(tuple(positions.shape), (1, 4, 8))
        self.assertFalse(torch.equal(positions[:, 0], positions[:, 1]))

    def test_loss_paths_are_finite(self):
        model = RealmModel(self.config)
        x = torch.randn(4, 4, 64)
        y = torch.tensor([0.0, 1.0, 0.0, 1.0])
        patient = torch.tensor([0, 1, 2, 0])
        self.assertTrue(torch.isfinite(stage1_loss(model(x), y).total))
        breakdown = stage2_loss(model(x, grl_coefficient=0.5), y, patient)
        self.assertTrue(torch.isfinite(breakdown.total))
        self.assertIsNotNone(breakdown.patient)

    def test_schedule_endpoints(self):
        self.assertEqual(dann_coefficient(0.0), 0.0)
        self.assertGreater(dann_coefficient(1.0), 0.99)

    def test_predict_proba_is_deterministic_and_restores_training_mode(self):
        config = ModelConfig(
            channels=4,
            conv_width=8,
            latent_dim=12,
            transformer_layers=1,
            attention_heads=2,
            feedforward_dim=16,
            dropout=0.5,
            patient_classes=3,
        )
        model = RealmModel(config).train()
        values = torch.randn(3, 4, 64)
        first = model.predict_proba(values)
        second = model.predict_proba(values)
        torch.testing.assert_close(first, second)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
