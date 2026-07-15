# Examples

Run the complete synthetic path from the repository root:

```bash
realm-eeg synthetic-demo --output realm-demo
realm-eeg validate-npz realm-demo/synthetic-windows.npz
```

For real data, write a local adapter that parses the source dataset under its own access terms,
applies a declared preprocessing policy, and emits numeric `x`, `y`, and `patient_id` arrays.
For grouped support/query sampling and continuous-timeline alert metrics, also emit the
all-or-none `recording_id`, `group_id`, and `window_start_seconds` arrays documented in
[`docs/DATASETS.md`](../docs/DATASETS.md). Do not commit raw EEG, private manifests, or
generated checkpoints.
