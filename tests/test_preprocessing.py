import unittest

import numpy as np

from realm_eeg.preprocessing import (
    canonical_channel_name,
    channelwise_zscore,
    harmonize_channels,
    preprocess_window,
    resample_linear,
    segment_signal,
)


class PreprocessingTests(unittest.TestCase):
    def test_aliases_and_missing_channel_mask(self):
        values = np.vstack([np.arange(8), np.arange(8) + 10]).astype(np.float32)
        harmonized, mask = harmonize_channels(values, ["EEG T7-REF", "Fp1"], ["T3", "Fp1", "O2"])
        self.assertEqual(canonical_channel_name("EEG T7-REF"), "T3")
        self.assertEqual(mask.tolist(), [True, True, False])
        np.testing.assert_array_equal(harmonized[0], values[0])
        np.testing.assert_array_equal(harmonized[2], 0)

    def test_duplicate_aliases_are_rejected(self):
        with self.assertRaises(ValueError):
            harmonize_channels(np.zeros((2, 8)), ["T7", "T3"], ["T3"])

    def test_resample_segment_and_normalize(self):
        values = np.vstack([np.linspace(0, 1, 10), np.ones(10)]).astype(np.float32)
        resampled = resample_linear(values, 10, 20)
        self.assertEqual(resampled.shape, (2, 20))
        normalized = channelwise_zscore(resampled)
        self.assertAlmostEqual(float(normalized[0].mean()), 0.0, places=5)
        self.assertTrue(np.allclose(normalized[1], 0.0))
        self.assertEqual(segment_signal(resampled, 8, 4).shape, (4, 2, 8))

    def test_downsampling_suppresses_out_of_band_tone(self):
        source_rate = 256
        time = np.arange(source_rate) / source_rate
        out_of_band = np.sin(2 * np.pi * 100 * time)[None, :].astype(np.float32)
        downsampled = resample_linear(out_of_band, source_rate, 64)
        self.assertEqual(downsampled.shape, (1, 64))
        self.assertLess(float(np.sqrt(np.mean(downsampled**2))), 0.05)

    def test_preprocess_enforces_minimum_channel_coverage(self):
        values = np.ones((2, 16), dtype=np.float32)
        processed = preprocess_window(
            values,
            ["Fp1", "Fp2"],
            128,
            target_rate_hz=128,
            target_names=["Fp1", "Fp2", "F3"],
            minimum_channel_fraction=0.5,
        )
        self.assertEqual(processed.channel_mask.tolist(), [True, True, False])
        with self.assertRaisesRegex(ValueError, "channel coverage"):
            preprocess_window(
                values[:1],
                ["Fp1"],
                128,
                target_rate_hz=128,
                target_names=["Fp1", "Fp2", "F3"],
            )

    def test_preprocess_rejects_bipolar_names_without_reconstruction(self):
        with self.assertRaisesRegex(ValueError, "bipolar montage reconstruction"):
            preprocess_window(
                np.ones((2, 16), dtype=np.float32),
                ["Fp1-F7", "F7-T3"],
                128,
                target_rate_hz=128,
                target_names=["Fp1", "F7", "T3"],
            )

    def test_preprocess_rejects_invalid_channel_threshold(self):
        for threshold in (0, -0.1, 1.1, np.nan):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                preprocess_window(
                    np.ones((1, 8), dtype=np.float32),
                    ["Fp1"],
                    128,
                    target_rate_hz=128,
                    target_names=["Fp1"],
                    minimum_channel_fraction=threshold,
                )


if __name__ == "__main__":
    unittest.main()
