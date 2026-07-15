import tempfile
import unittest
from pathlib import Path

import math

from realm_eeg.config import DataConfig, ModelConfig, TrainingConfig, load_config, save_config


class ConfigTests(unittest.TestCase):
    def test_json_round_trip(self):
        data = DataConfig(channels=8)
        model = ModelConfig(channels=8, conv_width=16, latent_dim=24, patient_classes=4)
        training = TrainingConfig(stage1_epochs=1, stage2_epochs=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_config(path, data, model, training, extra={"seed": 17})
            self.assertEqual(load_config(path), (data, model, training))

    def test_invalid_attention_width_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelConfig(conv_width=10, attention_heads=4).validate()

    def test_non_finite_and_incorrect_types_are_rejected(self):
        for config in (
            DataConfig(segment_seconds=math.nan),
            DataConfig(sampling_rate_hz=True),
            DataConfig(sampling_rate_hz=0.1, segment_seconds=0.1),
            ModelConfig(dropout=math.inf),
            TrainingConfig(learning_rate=math.nan),
            TrainingConfig(stage1_epochs=0),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_data_and_model_channel_counts_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            with self.assertRaisesRegex(ValueError, "data.channels must equal model.channels"):
                save_config(path, DataConfig(channels=8), ModelConfig(channels=4), TrainingConfig())


if __name__ == "__main__":
    unittest.main()
