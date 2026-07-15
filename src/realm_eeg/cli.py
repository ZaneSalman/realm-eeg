"""Command-line entry points for validation and a synthetic end-to-end demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import __version__
from .adaptation import adapt_to_patient
from .config import DataConfig, ModelConfig, TrainingConfig, save_config
from .data import (
    patient_episodes,
    synthetic_eeg_windows,
    validate_npz_arrays,
)
from .explain import explain_prediction, fit_centroids
from .metrics import binary_metrics
from .model import RealmModel
from .training import save_checkpoint, train_meta_epoch, train_stage1_epoch, train_stage2_epoch


def _probabilities(model: RealmModel, values: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.predict_proba(values).cpu().numpy()


def _json_ready(value):
    """Replace non-finite floats with JSON null for degenerate smoke-test metrics."""

    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _demo(args: argparse.Namespace) -> int:
    if args.patients < 4:
        raise ValueError("the demo needs at least four patients")
    if args.windows_per_patient < 8 or args.windows_per_patient % 2:
        raise ValueError("windows-per-patient must be an even integer of at least 8")
    if args.support_per_class < 1:
        raise ValueError("support-per-class must be positive")
    if args.support_per_class >= args.windows_per_patient // 2:
        raise ValueError("support-per-class must leave at least one query example per class")
    for name in (
        "stage1_epochs",
        "stage2_epochs",
        "meta_epochs",
        "inner_steps",
        "adaptation_steps",
        "integration_steps",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for name in (
        "sampling_rate_hz",
        "learning_rate",
        "inner_learning_rate",
        "adaptation_learning_rate",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    x_array, y_array, patient_array = synthetic_eeg_windows(
        patients=args.patients,
        windows_per_patient=args.windows_per_patient,
        channels=args.channels,
        samples=args.samples,
        sampling_rate_hz=args.sampling_rate_hz,
        seed=args.seed,
    )
    group_array = np.arange(x_array.shape[0], dtype=np.int64)
    recording_array = patient_array.copy()
    window_start_array = np.tile(
        np.arange(args.windows_per_patient, dtype=np.float64)
        * (args.samples / args.sampling_rate_hz),
        args.patients,
    )
    np.savez_compressed(
        output / "synthetic-windows.npz",
        x=x_array,
        y=y_array,
        patient_id=patient_array,
        recording_id=recording_array,
        group_id=group_array,
        window_start_seconds=window_start_array,
    )

    held_out_patient = args.patients - 1
    train_mask = patient_array != held_out_patient
    held_out_mask = ~train_mask
    train_x = torch.from_numpy(x_array[train_mask])
    train_y = torch.from_numpy(y_array[train_mask]).float()
    train_patient = torch.from_numpy(patient_array[train_mask]).long()
    train_groups = torch.from_numpy(group_array[train_mask]).long()
    held_x = torch.from_numpy(x_array[held_out_mask])
    held_y = torch.from_numpy(y_array[held_out_mask]).float()
    held_patient = torch.from_numpy(patient_array[held_out_mask]).long()
    held_groups = torch.from_numpy(group_array[held_out_mask]).long()

    data_config = DataConfig(
        sampling_rate_hz=args.sampling_rate_hz,
        segment_seconds=args.samples / args.sampling_rate_hz,
        channels=args.channels,
    )
    model_config = ModelConfig(
        channels=args.channels,
        conv_width=args.conv_width,
        latent_dim=args.latent_dim,
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=max(32, args.conv_width * 2),
        dropout=0.0,
        patient_classes=args.patients - 1,
    )
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        inner_learning_rate=args.inner_learning_rate,
        inner_steps=args.inner_steps,
    )
    save_config(
        output / "demo-config.json",
        data_config,
        model_config,
        training_config,
        extra={
            "seed": args.seed,
            "synthetic_only": True,
            "held_out_patient": held_out_patient,
        },
    )

    model = RealmModel(model_config)
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_patient),
        batch_size=min(args.batch_size, train_x.shape[0]),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    history: dict[str, list[dict[str, float]]] = {"stage1": [], "stage2": [], "stage3": []}
    stage1_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=training_config.weight_decay
    )
    for _ in range(args.stage1_epochs):
        history["stage1"].append(train_stage1_epoch(model, loader, stage1_optimizer))
    save_checkpoint(
        output / "realm-stage1-synthetic.pt",
        model,
        stage="stage1-population",
        extra={"synthetic_only": True, "seed": args.seed, "optimizer_state_included": False},
    )

    stage2_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=training_config.weight_decay
    )
    for epoch in range(args.stage2_epochs):
        history["stage2"].append(
            train_stage2_epoch(model, loader, stage2_optimizer, epoch, args.stage2_epochs)
        )
    save_checkpoint(
        output / "realm-stage2-synthetic.pt",
        model,
        stage="stage2-disentanglement",
        extra={"synthetic_only": True, "seed": args.seed, "optimizer_state_included": False},
    )

    stage3_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=training_config.weight_decay
    )
    for meta_epoch in range(args.meta_epochs):
        episodes = list(
            patient_episodes(
                train_x,
                train_y,
                train_patient,
                support_per_class=args.support_per_class,
                query_per_class=1,
                seed=args.seed + meta_epoch,
                group_ids=train_groups,
            )
        )
        history["stage3"].append(
            train_meta_epoch(
                model,
                episodes,
                stage3_optimizer,
                inner_learning_rate=args.inner_learning_rate,
                inner_steps=args.inner_steps,
                first_order=True,
            )
        )

    held_episode = next(
        patient_episodes(
            held_x,
            held_y,
            held_patient,
            support_per_class=args.support_per_class,
            query_per_class=(args.windows_per_patient // 2) - args.support_per_class,
            seed=args.seed,
            group_ids=held_groups,
        )
    )
    before = _probabilities(model, held_episode.query_x)
    adapted = adapt_to_patient(
        model,
        held_episode.support_x,
        held_episode.support_y,
        steps=args.adaptation_steps,
        learning_rate=args.adaptation_learning_rate,
    )
    after = _probabilities(adapted, held_episode.query_x)
    labels = held_episode.query_y.numpy().astype(np.int64)
    with torch.no_grad():
        reference_embeddings = adapted(train_x).latent
    centroids = fit_centroids(reference_embeddings, train_y)
    explanation = explain_prediction(
        adapted,
        held_episode.query_x[:2],
        centroids=centroids,
        integration_steps=args.integration_steps,
    )
    np.savez_compressed(
        output / "demo-explanation.npz",
        seizure_probability=explanation.seizure_probability.detach().cpu().numpy(),
        input_attribution=explanation.input_attribution.detach().cpu().numpy(),
        temporal_attention=explanation.temporal_attention.detach().cpu().numpy(),
        distance_to_non_seizure=explanation.distance_to_non_seizure.detach().cpu().numpy(),
        distance_to_seizure=explanation.distance_to_seizure.detach().cpu().numpy(),
    )
    metrics = {
        "warning": "Synthetic smoke-test metrics are not scientific or clinical evidence.",
        "optimizer_policy": (
            "Fresh AdamW optimizer state at each stage boundary; checkpoints contain model "
            "weights and safe metadata, not optimizer state."
        ),
        "held_out_patient": held_out_patient,
        "support_examples": int(held_episode.support_y.numel()),
        "query_examples": int(held_episode.query_y.numel()),
        "before_adaptation": binary_metrics(labels, before).to_dict(),
        "after_adaptation": binary_metrics(labels, after).to_dict(),
        "false_alarm_events_per_hour_after": None,
        "false_alarm_note": (
            "Not computed: the balanced synthetic query set is not a continuous recording "
            "timeline. Use explicit recording IDs, window starts, and recording hours."
        ),
        "stage4_explanations": {
            "examples": 2,
            "file": "demo-explanation.npz",
            "method": "integrated gradients, attention rollout, and latent centroid distance",
        },
        "history": history,
        "model_config": asdict(model_config),
    }
    serializable_metrics = _json_ready(metrics)
    (output / "demo-metrics.json").write_text(
        json.dumps(serializable_metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_checkpoint(
        output / "realm-synthetic-demo.pt",
        model,
        stage="stage3-meta",
        extra={"synthetic_only": True, "seed": args.seed},
    )
    print(json.dumps({"output": str(output), "metrics": serializable_metrics}, indent=2))
    return 0


def _validate_npz(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    with np.load(path, mmap_mode="r", allow_pickle=False) as payload:
        validate_npz_arrays(payload)
        summary = {
            "path": str(path),
            "windows": int(payload["x"].shape[0]),
            "channels": int(payload["x"].shape[1]),
            "samples": int(payload["x"].shape[2]),
            "patients": int(np.unique(payload["patient_id"]).size),
            "seizure_windows": int(np.asarray(payload["y"]).sum()),
            "temporal_metadata": all(
                key in payload for key in ("recording_id", "group_id", "window_start_seconds")
            ),
        }
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="realm-eeg",
        description="Reference implementation of the proposed REALM EEG framework.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "synthetic-demo",
        help="run all train/adapt stages on non-clinical generated data",
    )
    demo.add_argument("--output", default="realm-demo")
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--patients", type=int, default=5)
    demo.add_argument("--windows-per-patient", type=int, default=12)
    demo.add_argument("--channels", type=int, default=8)
    demo.add_argument("--samples", type=int, default=128)
    demo.add_argument("--sampling-rate-hz", type=float, default=128.0)
    demo.add_argument("--conv-width", type=int, default=16)
    demo.add_argument("--latent-dim", type=int, default=24)
    demo.add_argument("--batch-size", type=int, default=16)
    demo.add_argument("--learning-rate", type=float, default=1e-3)
    demo.add_argument("--stage1-epochs", type=int, default=1)
    demo.add_argument("--stage2-epochs", type=int, default=1)
    demo.add_argument("--meta-epochs", type=int, default=1)
    demo.add_argument("--support-per-class", type=int, default=1)
    demo.add_argument("--inner-learning-rate", type=float, default=1e-2)
    demo.add_argument("--inner-steps", type=int, default=1)
    demo.add_argument("--adaptation-steps", type=int, default=5)
    demo.add_argument("--adaptation-learning-rate", type=float, default=1e-2)
    demo.add_argument("--integration-steps", type=int, default=8)
    demo.set_defaults(handler=_demo)

    validate = commands.add_parser("validate-npz", help="validate the documented NPZ format")
    validate.add_argument("path")
    validate.set_defaults(handler=_validate_npz)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ImportError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
