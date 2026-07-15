import unittest

import numpy as np
import torch

from realm_eeg.diagnostics import patient_identity_linear_probe


class PatientIdentityProbeTests(unittest.TestCase):
    def setUp(self):
        centers = torch.eye(3, dtype=torch.float64)
        self.embeddings = torch.cat(
            [centers[index].repeat(6, 1) for index in range(3)]
        ).requires_grad_()
        self.patient_ids = np.repeat(np.array([10, 20, 30], dtype=np.int64), 6)
        self.train_indices = np.concatenate([np.arange(start, start + 4) for start in (0, 6, 12)])
        self.test_indices = np.concatenate(
            [np.arange(start + 4, start + 6) for start in (0, 6, 12)]
        )

    def test_probe_is_accurate_deterministic_and_does_not_backpropagate(self):
        torch.manual_seed(1234)
        rng_state = torch.random.get_rng_state()
        first = patient_identity_linear_probe(
            self.embeddings,
            self.patient_ids,
            train_indices=self.train_indices,
            test_indices=self.test_indices,
            steps=100,
        )
        second = patient_identity_linear_probe(
            self.embeddings,
            self.patient_ids,
            train_indices=self.train_indices,
            test_indices=self.test_indices,
            steps=100,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.accuracy, 1.0)
        self.assertAlmostEqual(first.majority_baseline_accuracy, 1 / 3)
        self.assertAlmostEqual(first.uniform_random_baseline_accuracy, 1 / 3)
        self.assertEqual(first.majority_patient_id, 10)
        self.assertEqual(first.patient_classes, (10, 20, 30))
        self.assertIsNone(self.embeddings.grad)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))

    def test_rejects_invalid_embeddings_and_patient_ids(self):
        cases = (
            (torch.ones(18), self.patient_ids, "2-D"),
            (torch.full((18, 3), float("nan")), self.patient_ids, "finite"),
            (torch.ones(18, 3), self.patient_ids.astype(float), "integer"),
            (torch.ones(18, 3), -np.ones(18, dtype=np.int64), "negative"),
        )
        for embeddings, patient_ids, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                patient_identity_linear_probe(
                    embeddings,
                    patient_ids,
                    train_indices=self.train_indices,
                    test_indices=self.test_indices,
                    steps=1,
                )

    def test_rejects_invalid_explicit_splits(self):
        cases = (
            (np.array([], dtype=np.int64), self.test_indices, "empty"),
            (np.array([0, 0, 6], dtype=np.int64), self.test_indices, "duplicate"),
            (np.array([0, 6, 12], dtype=np.int64), np.array([0, 4]), "disjoint"),
            (np.array([0, 6, 12], dtype=np.int64), np.array([18]), "out-of-range"),
            (np.array([0.0, 6.0]), self.test_indices, "integer"),
        )
        for train, test, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                patient_identity_linear_probe(
                    self.embeddings,
                    self.patient_ids,
                    train_indices=train,
                    test_indices=test,
                    steps=1,
                )

    def test_rejects_test_patient_class_absent_from_training(self):
        with self.assertRaisesRegex(ValueError, "absent from training"):
            patient_identity_linear_probe(
                self.embeddings,
                self.patient_ids,
                train_indices=np.arange(12),
                test_indices=np.arange(12, 18),
                steps=1,
            )


if __name__ == "__main__":
    unittest.main()
