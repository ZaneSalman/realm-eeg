import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from realm_eeg.data import (
    NPYDirectoryDataset,
    NPZWindowDataset,
    patient_episodes,
    patient_level_split,
    synthetic_eeg_windows,
    validate_npz_arrays,
)


class DataTests(unittest.TestCase):
    def test_patient_split_is_disjoint_and_exhaustive(self):
        patient_ids = np.repeat(np.arange(10), 3)
        split = patient_level_split(patient_ids, seed=17)
        partitions = [
            set(split.train_patients),
            set(split.validation_patients),
            set(split.test_patients),
        ]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertEqual(set.union(*partitions), set(range(10)))
        all_indices = np.concatenate(
            [split.train_indices, split.validation_indices, split.test_indices]
        )
        self.assertEqual(set(all_indices.tolist()), set(range(patient_ids.size)))

    def test_balanced_patient_episodes(self):
        x = torch.randn(16, 2, 32)
        y = torch.tensor([0, 1] * 8)
        patient = torch.tensor([0] * 8 + [1] * 8)
        group = torch.arange(16)
        episodes = list(patient_episodes(x, y, patient, 1, 2, seed=17, group_ids=group))
        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            self.assertEqual(torch.bincount(episode.support_y.long()).tolist(), [1, 1])
            self.assertEqual(torch.bincount(episode.query_y.long()).tolist(), [2, 2])

    def test_episode_groups_are_disjoint(self):
        x = torch.arange(24 * 2 * 32, dtype=torch.float32).reshape(24, 2, 32)
        y = torch.tensor([0, 1, 0, 1] * 6)
        patient = torch.zeros(24, dtype=torch.long)
        group = torch.repeat_interleave(torch.arange(6), 4)
        episode = next(patient_episodes(x, y, patient, 1, 2, seed=17, group_ids=group))
        row_width = x.shape[1] * x.shape[2]
        support_indices = (episode.support_x[:, 0, 0] / row_width).long()
        query_indices = (episode.query_x[:, 0, 0] / row_width).long()
        self.assertFalse(set(group[support_indices].tolist()) & set(group[query_indices].tolist()))

    def test_npz_contract_and_dataset(self):
        x, y, patient = synthetic_eeg_windows(
            patients=3, windows_per_patient=4, channels=2, samples=32, seed=17
        )
        validate_npz_arrays({"x": x, "y": y, "patient_id": patient})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.npz"
            np.savez_compressed(path, x=x, y=y, patient_id=patient)
            dataset = NPZWindowDataset(path)
            self.assertEqual(len(dataset), 12)
            self.assertEqual(tuple(dataset[0]["x"].shape), (2, 32))

            array_directory = Path(directory) / "arrays"
            array_directory.mkdir()
            np.save(array_directory / "x.npy", x)
            np.save(array_directory / "y.npy", y)
            np.save(array_directory / "patient_id.npy", patient)
            mapped = NPYDirectoryDataset(array_directory)
            self.assertIsInstance(mapped.x, np.memmap)
            self.assertEqual(len(mapped), 12)

    def test_optional_aligned_arrays_are_exposed_and_memory_mapped(self):
        x, y, patient = synthetic_eeg_windows(
            patients=3, windows_per_patient=4, channels=2, samples=32, seed=17
        )
        recording = patient.copy()
        group = np.arange(x.shape[0], dtype=np.int64)
        starts = np.tile(np.arange(4, dtype=np.float64) * 2.0, 3)
        arrays = {
            "x": x,
            "y": y,
            "patient_id": patient,
            "recording_id": recording,
            "group_id": group,
            "window_start_seconds": starts,
        }
        validate_npz_arrays(arrays)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "windows.npz"
            np.savez_compressed(path, **arrays)
            dataset = NPZWindowDataset(path)
            self.assertEqual(
                set(dataset[0]),
                {
                    "x",
                    "y",
                    "patient_id",
                    "recording_id",
                    "group_id",
                    "window_start_seconds",
                },
            )
            self.assertEqual(dataset[0]["recording_id"].item(), 0)
            self.assertEqual(dataset[1]["window_start_seconds"].item(), 2.0)

            array_directory = Path(directory) / "arrays"
            array_directory.mkdir()
            for name, array in arrays.items():
                np.save(array_directory / f"{name}.npy", array)
            mapped = NPYDirectoryDataset(array_directory)
            self.assertIsInstance(mapped.recording_id, np.memmap)
            self.assertIsInstance(mapped.window_start_seconds, np.memmap)
            self.assertEqual(mapped[1]["group_id"].item(), 1)

    def test_optional_aligned_arrays_are_all_or_none(self):
        base = {
            "x": np.zeros((2, 1, 8), dtype=np.float32),
            "y": np.array([0, 1]),
            "patient_id": np.array([0, 1]),
        }
        with self.assertRaisesRegex(ValueError, "provided together"):
            validate_npz_arrays({**base, "recording_id": np.array([0, 1])})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name, array in base.items():
                np.save(path / f"{name}.npy", array)
            np.save(path / "group_id.npy", np.array([0, 1]))
            with self.assertRaisesRegex(ValueError, "provided together"):
                NPYDirectoryDataset(path)

    def test_invalid_optional_aligned_arrays_are_rejected(self):
        base = {
            "x": np.zeros((2, 1, 8), dtype=np.float32),
            "y": np.array([0, 1]),
            "patient_id": np.array([0, 1]),
            "recording_id": np.array([0, 1]),
            "group_id": np.array([0, 1]),
            "window_start_seconds": np.array([0.0, 1.0]),
        }
        invalid = {
            "recording shape": {"recording_id": np.array([[0], [1]])},
            "group type": {"group_id": np.array([0.0, 1.0])},
            "negative group": {"group_id": np.array([0, -1])},
            "start type": {"window_start_seconds": np.array([0, 1])},
            "non-finite start": {"window_start_seconds": np.array([0.0, np.nan])},
            "negative start": {"window_start_seconds": np.array([0.0, -1.0])},
        }
        for case, replacement in invalid.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_npz_arrays({**base, **replacement})

    def test_patient_split_allows_empty_optional_partitions(self):
        ids = np.repeat(np.arange(3), 2)
        train_only = patient_level_split(ids, validation_fraction=0, test_fraction=0)
        self.assertEqual(set(train_only.train_patients), {0, 1, 2})
        self.assertEqual(train_only.validation_patients, ())
        self.assertEqual(train_only.test_patients, ())
        self.assertEqual(train_only.validation_indices.size, 0)
        self.assertEqual(train_only.test_indices.size, 0)

        train_validation = patient_level_split(ids[:4], validation_fraction=0.5, test_fraction=0)
        self.assertTrue(train_validation.train_patients)
        self.assertTrue(train_validation.validation_patients)
        self.assertEqual(train_validation.test_patients, ())

    def test_patient_split_rejects_invalid_ids_before_casting(self):
        for ids in (
            np.array([0.0, 1.0, 2.0]),
            np.array([[0, 1], [2, 3]]),
            np.array([0, -1, 2]),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                patient_level_split(ids, validation_fraction=0, test_fraction=0)

    def test_patient_split_requires_patients_only_for_requested_partitions(self):
        patient_level_split(np.array([7]), validation_fraction=0, test_fraction=0)
        with self.assertRaisesRegex(ValueError, "at least 2 patients"):
            patient_level_split(np.array([7]), validation_fraction=0.2, test_fraction=0)
        with self.assertRaisesRegex(ValueError, "at least 3 patients"):
            patient_level_split(np.array([7, 8]), validation_fraction=0.2, test_fraction=0.2)

    def test_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_npz_arrays(
                {
                    "x": np.zeros((2, 1, 4), dtype=np.float32),
                    "y": np.array([0, 2]),
                    "patient_id": np.array([0, 1]),
                }
            )

    def test_negative_patient_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_npz_arrays(
                {
                    "x": np.zeros((2, 1, 8), dtype=np.float32),
                    "y": np.array([0, 1]),
                    "patient_id": np.array([0, -1]),
                }
            )

    def test_non_floating_eeg_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_npz_arrays(
                {
                    "x": np.zeros((2, 1, 8), dtype=np.int16),
                    "y": np.array([0, 1]),
                    "patient_id": np.array([0, 1]),
                }
            )


if __name__ == "__main__":
    unittest.main()
