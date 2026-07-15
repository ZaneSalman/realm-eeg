import unittest

import numpy as np

from realm_eeg.metrics import binary_metrics, bootstrap_interval, false_alarm_events_per_hour


class MetricsTests(unittest.TestCase):
    def test_known_binary_metrics(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.8, 0.7, 0.9])
        metrics = binary_metrics(labels, probabilities)
        self.assertEqual((metrics.true_positives, metrics.false_positives), (2, 1))
        self.assertAlmostEqual(metrics.sensitivity, 1.0)
        self.assertAlmostEqual(metrics.specificity, 0.5)
        self.assertAlmostEqual(metrics.roc_auc, 0.75)

    def test_false_alarm_runs_reset_by_patient(self):
        labels = np.zeros(6, dtype=int)
        probabilities = np.array([0.9, 0.8, 0.1, 0.9, 0.9, 0.2])
        recording = np.array([0, 0, 0, 1, 1, 1])
        starts = np.array([0, 600, 1200, 0, 600, 1200])
        rate = false_alarm_events_per_hour(
            labels,
            probabilities,
            recording_ids=recording,
            window_start_seconds=starts,
            total_recording_hours=1.0,
            merge_gap_seconds=600,
        )
        self.assertAlmostEqual(rate, 2.0)

    def test_bootstrap_interval_is_ordered(self):
        low, high = bootstrap_interval(np.arange(10.0), resamples=100, seed=17)
        self.assertLess(low, high)

    def test_multidimensional_metric_inputs_are_rejected(self):
        labels = np.array([[0, 1]])
        probabilities = np.array([[0.1, 0.9]])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            binary_metrics(labels, probabilities)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            bootstrap_interval(np.ones((2, 2)))

    def test_false_alarm_timing_and_recording_ids_are_validated(self):
        labels = np.array([0, 0])
        probabilities = np.array([0.9, 0.1])
        starts = np.array([0.0, 10.0])
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            false_alarm_events_per_hour(
                labels,
                probabilities,
                recording_ids=np.array([0, 0]),
                window_start_seconds=starts,
                total_recording_hours=np.nan,
                merge_gap_seconds=10,
            )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            false_alarm_events_per_hour(
                labels,
                probabilities,
                recording_ids=np.array(["a", "a"]),
                window_start_seconds=starts,
                total_recording_hours=1,
                merge_gap_seconds=10,
            )


if __name__ == "__main__":
    unittest.main()
