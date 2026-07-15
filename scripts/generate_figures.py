#!/usr/bin/env python3
"""Generate the five publication figures used by the REALM documentation.

The conceptual figures are drawn directly. Simulated and illustrative figures
are rendered from committed CSV files in ``docs/assets/figure-data``. Use
``--refresh-data`` to deterministically regenerate those CSV files.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Figure generation requires numpy and matplotlib. "
        "Install the project figure dependencies with: pip install '.[figures]'"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "docs" / "assets" / "figures"
DATA_DIR = REPO_ROOT / "docs" / "assets" / "figure-data"

INK = "#17324D"
INK_2 = "#284862"
BLUE = "#2F6B9A"
BLUE_LIGHT = "#DCEAF5"
ORANGE = "#D97706"
ORANGE_LIGHT = "#FBE7CC"
GOLD = "#B6860B"
GOLD_LIGHT = "#FAF0C7"
OLIVE = "#667A35"
OLIVE_LIGHT = "#E7EED8"
PINK = "#A34F73"
PINK_LIGHT = "#F3DFE8"
NEUTRAL = "#687381"
NEUTRAL_LIGHT = "#EEF1F5"
GRID = "#D9E0E8"
PAPER = "#FBFCFE"
WHITE = "#FFFFFF"

PATIENT_COLORS = {
    "P1": BLUE,
    "P2": ORANGE,
    "P3": OLIVE,
    "P4": PINK,
    "P5": GOLD,
}
PATIENT_MARKERS = {"P1": "o", "P2": "s", "P3": "^", "P4": "D", "P5": "P"}


def configure_style() -> None:
    """Set deterministic, publication-oriented Matplotlib defaults."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": 300,
            "svg.fonttype": "none",
            "svg.hashsalt": "realm-eeg-figures-v1",
            "lines.linewidth": 2.0,
            "patch.linewidth": 1.25,
        }
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accessible_svg(path: Path, title: str, description: str) -> None:
    """Add explicit SVG title/description and an ARIA label relationship."""

    svg = path.read_text(encoding="utf-8")
    title_id = f"{path.stem}-title"
    desc_id = f"{path.stem}-desc"
    svg = svg.replace(
        "<svg ",
        f'<svg role="img" aria-labelledby="{title_id} {desc_id}" ',
        1,
    )
    end = svg.find(">", svg.find("<svg "))
    accessibility = (
        f'\n <title id="{title_id}">{html.escape(title)}</title>'
        f'\n <desc id="{desc_id}">{html.escape(description)}</desc>'
    )
    svg = svg[: end + 1] + accessibility + svg[end + 1 :]
    path.write_text(svg, encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str, title: str, description: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / f"{stem}.png"
    svg = FIGURE_DIR / f"{stem}.svg"
    png_metadata = {
        "Title": title,
        "Description": description,
        "Creator": "REALM reproducible figure generator",
    }
    svg_metadata = {
        "Description": description,
        "Creator": "REALM reproducible figure generator",
        "Date": None,
    }
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.15, metadata=png_metadata)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.15, metadata=svg_metadata)
    plt.close(fig)
    accessible_svg(svg, title, description)
    print(f"wrote {png.relative_to(REPO_ROOT)}")
    print(f"wrote {svg.relative_to(REPO_ROOT)}")


def add_badge(fig: plt.Figure, text: str, color: str = INK) -> None:
    fig.text(
        0.985,
        0.965,
        text,
        ha="right",
        va="top",
        fontsize=8.5,
        weight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.38",
            "facecolor": WHITE,
            "edgecolor": color,
            "linewidth": 1.1,
        },
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str = INK,
    radius: float = 0.02,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.35,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = INK,
    linestyle: str = "-",
    connectionstyle: str = "arc3",
    mutation_scale: float = 15,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=1.8,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            shrinkA=1,
            shrinkB=1,
        )
    )


def figure_01_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(
        0.03,
        0.95,
        "REALM four-stage cross-patient seizure-detection pipeline",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.03,
        0.905,
        "A shared CNN–Transformer backbone is progressively trained, disentangled, adapted, and explained.",
        fontsize=10.5,
        color=NEUTRAL,
    )
    add_badge(fig, "CONCEPTUAL ARCHITECTURE", BLUE)

    y = 0.40
    h = 0.28
    node_specs = [
        (0.018, 0.105, "INPUT", "Raw EEG", "Multi-patient\nsignals", NEUTRAL_LIGHT, NEUTRAL),
        (
            0.165,
            0.14,
            "STAGE 1",
            "Population\npretraining",
            "CNN–Transformer\nseizure literacy",
            BLUE_LIGHT,
            BLUE,
        ),
        (
            0.345,
            0.14,
            "STAGE 2",
            "Adversarial\ndisentanglement",
            "Gradient reversal\nremoves patient cues",
            ORANGE_LIGHT,
            ORANGE,
        ),
        (
            0.525,
            0.14,
            "STAGE 3",
            "Few-shot\nadaptation",
            "MAML personalization\nwith K labels",
            OLIVE_LIGHT,
            OLIVE,
        ),
        (
            0.705,
            0.14,
            "STAGE 4",
            "Interpretability\nlayer",
            "SHAP • attention •\nfeature position",
            PINK_LIGHT,
            PINK,
        ),
        (
            0.895,
            0.087,
            "OUTPUT",
            "Clinical\nsupport",
            "Prediction +\nexplanation",
            GOLD_LIGHT,
            GOLD,
        ),
    ]

    for x, width, tag, heading, detail, fill, edge in node_specs:
        rounded_box(ax, (x, y), width, h, fill, edge)
        ax.text(
            x + width / 2,
            y + h - 0.055,
            tag,
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color=edge,
        )
        ax.text(
            x + width / 2,
            y + 0.15,
            heading,
            ha="center",
            va="center",
            fontsize=11.5,
            weight="bold",
            color=INK,
            linespacing=1.05,
        )
        ax.text(
            x + width / 2,
            y + 0.055,
            detail,
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK_2,
            linespacing=1.15,
        )

    centers = [(x, width) for x, width, *_ in node_specs]
    for (x1, w1), (x2, _w2) in zip(centers[:-1], centers[1:]):
        arrow(ax, (x1 + w1 + 0.005, y + h / 2), (x2 - 0.006, y + h / 2), INK)

    stage_centers = [x + width / 2 for x, width, *_ in node_specs[1:5]]
    ax.plot(
        [stage_centers[0] - 0.055, stage_centers[-1] + 0.055],
        [0.255, 0.255],
        color=BLUE,
        linestyle=(0, (5, 3)),
        linewidth=2.0,
    )
    for center in stage_centers:
        ax.plot(center, 0.255, marker="o", markersize=5.5, color=BLUE, markeredgecolor=INK)
    ax.text(
        sum(stage_centers) / len(stage_centers),
        0.205,
        "Shared backbone refined in Stages 1–3 and queried for explanations in Stage 4",
        ha="center",
        va="center",
        fontsize=9.5,
        color=BLUE,
        weight="bold",
    )
    ax.text(
        0.5,
        0.09,
        "Population knowledge  →  patient-invariant geometry  →  rapid personalization  →  auditable prediction",
        ha="center",
        fontsize=9.5,
        color=INK_2,
    )

    save_figure(
        fig,
        "figure-01-realm-pipeline",
        "REALM four-stage cross-patient seizure-detection pipeline",
        "Raw multi-patient EEG passes through population pretraining, adversarial patient disentanglement, "
        "few-shot MAML adaptation, and SHAP, attention, and feature-space interpretation before a clinical "
        "prediction and explanation are produced. The backbone is refined during Stages 1–3 and queried, "
        "rather than updated, in Stage 4.",
    )


def figure_02_gradient_reversal() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.text(
        0.035,
        0.955,
        "REALM Stage 2: adversarial patient disentanglement",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.035,
        0.912,
        "The seizure path preserves pathology; the gradient-reversal path suppresses patient-identifying information.",
        fontsize=10.5,
        color=NEUTRAL,
    )
    add_badge(fig, "CONCEPTUAL ARCHITECTURE", ORANGE)

    rounded_box(ax, (0.035, 0.43), 0.12, 0.18, NEUTRAL_LIGHT, NEUTRAL)
    ax.text(0.095, 0.54, "EEG input", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(
        0.095,
        0.48,
        r"$x \in \mathbb{R}^{C\times T}$",
        ha="center",
        va="center",
        fontsize=11,
        color=INK_2,
    )

    rounded_box(ax, (0.225, 0.38), 0.20, 0.28, BLUE_LIGHT, BLUE)
    ax.text(0.325, 0.56, r"Backbone $B_\theta$", ha="center", fontsize=13, weight="bold", color=INK)
    ax.text(0.325, 0.49, "CNN–Transformer", ha="center", fontsize=10.5, color=BLUE, weight="bold")
    ax.text(0.325, 0.44, "shared feature extractor", ha="center", fontsize=9.5, color=INK_2)
    arrow(ax, (0.155, 0.52), (0.22, 0.52), NEUTRAL)

    ax.add_patch(Circle((0.50, 0.52), 0.032, facecolor=WHITE, edgecolor=INK, linewidth=1.5))
    ax.text(0.50, 0.52, r"$z$", ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.50, 0.465, r"$z=B_\theta(x)$", ha="center", va="top", fontsize=9, color=NEUTRAL)
    arrow(ax, (0.425, 0.52), (0.467, 0.52), NEUTRAL)

    rounded_box(ax, (0.61, 0.63), 0.19, 0.17, BLUE_LIGHT, BLUE)
    ax.text(0.705, 0.74, "Seizure head", ha="center", va="center", fontsize=12.5, weight="bold")
    ax.text(
        0.705, 0.68, r"$f_{\mathrm{seiz}}(z)$", ha="center", va="center", fontsize=11, color=BLUE
    )
    rounded_box(ax, (0.855, 0.63), 0.11, 0.17, WHITE, BLUE)
    ax.text(0.91, 0.735, "MINIMIZE", ha="center", fontsize=8, weight="bold", color=BLUE)
    ax.text(0.91, 0.68, r"$\mathcal{L}_{\mathrm{seiz}}$", ha="center", fontsize=12, color=INK)
    arrow(ax, (0.525, 0.54), (0.605, 0.70), BLUE, "-")
    arrow(ax, (0.80, 0.715), (0.85, 0.715), BLUE, "-")
    ax.plot(0.565, 0.615, marker="o", markersize=7, markerfacecolor=BLUE, markeredgecolor=INK)
    ax.text(0.59, 0.60, "solid path", fontsize=8.5, color=BLUE, weight="bold")

    rounded_box(ax, (0.59, 0.25), 0.15, 0.17, ORANGE_LIGHT, ORANGE)
    ax.text(0.665, 0.36, "Gradient reversal", ha="center", fontsize=11.5, weight="bold")
    ax.text(0.665, 0.30, r"$\nabla \mapsto -\lambda\nabla$", ha="center", fontsize=11, color=ORANGE)
    rounded_box(ax, (0.77, 0.25), 0.13, 0.17, ORANGE_LIGHT, ORANGE)
    ax.text(0.835, 0.36, "Patient head", ha="center", fontsize=11.5, weight="bold")
    ax.text(0.835, 0.30, r"$f_{\mathrm{pat}}(z)$", ha="center", fontsize=11, color=ORANGE)
    rounded_box(ax, (0.925, 0.25), 0.055, 0.17, WHITE, ORANGE)
    ax.text(
        0.9525,
        0.365,
        "HEAD\nMIN",
        ha="center",
        va="center",
        fontsize=6.8,
        weight="bold",
        color=ORANGE,
        linespacing=0.9,
    )
    ax.text(0.9525, 0.305, r"$\mathcal{L}_{\mathrm{pat}}$", ha="center", fontsize=10, color=INK)
    arrow(ax, (0.525, 0.50), (0.585, 0.34), ORANGE, (0, (5, 3)))
    arrow(ax, (0.74, 0.335), (0.765, 0.335), ORANGE, (0, (5, 3)))
    arrow(ax, (0.90, 0.335), (0.92, 0.335), ORANGE, (0, (5, 3)))
    ax.plot(
        0.553,
        0.425,
        marker="D",
        markersize=6.5,
        markerfacecolor=WHITE,
        markeredgecolor=ORANGE,
        markeredgewidth=1.5,
    )
    ax.text(0.575, 0.415, "dashed path", fontsize=8.5, color=ORANGE, weight="bold")

    rounded_box(ax, (0.055, 0.12), 0.27, 0.16, WHITE, ORANGE)
    ax.text(0.075, 0.235, "GRL behavior", fontsize=9, weight="bold", color=ORANGE)
    ax.text(0.075, 0.195, r"Forward:  $\mathrm{GRL}(z)=z$", fontsize=9.3)
    ax.text(0.075, 0.158, r"Backward:  $\nabla \rightarrow -\lambda\nabla$", fontsize=9.3)
    ax.text(0.075, 0.125, "Trainable parameters: none", fontsize=9.3)

    rounded_box(ax, (0.41, 0.075), 0.46, 0.12, NEUTRAL_LIGHT, INK)
    ax.text(0.64, 0.155, "Backbone objective", ha="center", fontsize=9, weight="bold", color=INK_2)
    ax.text(
        0.64,
        0.105,
        r"$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{seiz}}-\lambda\mathcal{L}_{\mathrm{pat}}$",
        ha="center",
        fontsize=14,
        weight="bold",
    )
    ax.plot(
        [0.91, 0.91, 0.83], [0.63, 0.21, 0.195], color=NEUTRAL, linestyle=(0, (2, 3)), linewidth=1.2
    )
    ax.plot(
        [0.9525, 0.9525, 0.87],
        [0.25, 0.21, 0.195],
        color=NEUTRAL,
        linestyle=(0, (2, 3)),
        linewidth=1.2,
    )
    ax.text(
        0.64,
        0.025,
        "For the backbone, the patient loss is adversarially maximized while seizure loss is minimized.",
        ha="center",
        fontsize=9,
        color=NEUTRAL,
    )

    save_figure(
        fig,
        "figure-02-gradient-reversal-architecture",
        "REALM Stage 2 gradient-reversal architecture",
        "EEG input enters a CNN-Transformer backbone and produces latent vector z. A solid blue branch sends z "
        "to a seizure head that minimizes seizure loss. A dashed orange branch sends z through a gradient "
        "reversal layer and patient head. The patient head minimizes positive cross-entropy with respect to its "
        "own parameters, while the reversed gradient causes only the backbone to maximize patient loss. The "
        "total backbone objective is seizure loss minus lambda times patient loss.",
    )


def generate_embedding_data(path: Path) -> None:
    seed = 31703
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    point_id = 0
    centers = {
        "P1": (-1.65, 1.05),
        "P2": (-1.15, -1.10),
        "P3": (0.10, 1.55),
        "P4": (1.35, 0.65),
        "P5": (0.85, -1.15),
    }

    def append(panel: str, x: float, y: float, patient: str, klass: str, is_new: bool) -> None:
        nonlocal point_id
        point_id += 1
        rows.append(
            {
                "panel": panel,
                "point_id": f"pt-{point_id:04d}",
                "x": f"{x:.6f}",
                "y": f"{y:.6f}",
                "patient_id": patient,
                "class": klass,
                "is_new_patient": str(is_new).lower(),
                "status": "SIMULATED",
                "random_seed": seed,
            }
        )

    for patient, center in centers.items():
        for panel, contraction, spread in [
            ("stage1_patient_clusters", 1.0, 0.24),
            ("confusion_valley", 0.42, 0.42),
        ]:
            for _ in range(42):
                klass = "seizure" if rng.random() < 0.30 else "non-seizure"
                class_nudge = (
                    np.array([0.09, 0.04]) if klass == "seizure" else np.array([-0.04, -0.02])
                )
                xy = contraction * np.asarray(center) + class_nudge + rng.normal(0, spread, 2)
                append(panel, float(xy[0]), float(xy[1]), patient, klass, False)

    for klass, n, center, scale in [
        ("non-seizure", 150, (-0.95, -0.12), (0.42, 0.48)),
        ("seizure", 90, (1.15, 0.18), (0.37, 0.43)),
    ]:
        for _ in range(n):
            patient = str(rng.choice(list(centers)))
            xy = rng.normal(center, scale)
            append("stage2_pathology_geometry", float(xy[0]), float(xy[1]), patient, klass, False)

    for klass, n, center, scale in [
        ("non-seizure", 105, (-0.95, -0.12), (0.44, 0.50)),
        ("seizure", 65, (1.15, 0.18), (0.40, 0.46)),
    ]:
        for _ in range(n):
            patient = str(rng.choice(list(centers)))
            xy = rng.normal(center, scale)
            append("new_patient_arrival", float(xy[0]), float(xy[1]), patient, klass, False)

    for klass, n, center in [
        ("non-seizure", 26, (-0.82, 0.02)),
        ("seizure", 16, (1.02, 0.32)),
    ]:
        for _ in range(n):
            xy = rng.normal(center, (0.36, 0.40))
            append("new_patient_arrival", float(xy[0]), float(xy[1]), "NEW", klass, True)

    write_csv(
        path,
        [
            "panel",
            "point_id",
            "x",
            "y",
            "patient_id",
            "class",
            "is_new_patient",
            "status",
            "random_seed",
        ],
        rows,
    )


def figure_03_feature_space() -> None:
    path = DATA_DIR / "figure-03-simulated-embedding.csv"
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 4, figsize=(14.3, 5.7))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.72, bottom=0.14, wspace=0.11)
    fig.text(
        0.035,
        0.955,
        "REALM feature-space geometry across training stages",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.035,
        0.912,
        "Conceptual embedding coordinates show the intended transition from patient identity to seizure state.",
        fontsize=10.5,
        color=NEUTRAL,
    )
    add_badge(fig, "SIMULATED • NOT MEASURED EMBEDDINGS", ORANGE)

    patient_handles = [
        Line2D(
            [0],
            [0],
            marker=PATIENT_MARKERS[p],
            linestyle="none",
            markerfacecolor=PATIENT_COLORS[p],
            markeredgecolor=INK,
            markersize=7,
            label=f"Patient {p[1:]}",
        )
        for p in PATIENT_COLORS
    ]
    fig.legend(
        handles=patient_handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.855),
        ncol=5,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
        title="Panels 1–2: patient marker/color   •   open = non-seizure   •   filled = seizure",
        title_fontsize=9,
        fontsize=8.5,
    )

    panels = [
        ("stage1_patient_clusters", "After Stage 1", "Patient clusters dominate"),
        ("confusion_valley", "Mid-training", "Patient boundaries blur"),
        ("stage2_pathology_geometry", "After Stage 2", "Seizure state dominates"),
        ("new_patient_arrival", "New patient", "Maps into shared geometry"),
    ]
    for ax, (panel, title, takeaway) in zip(axes, panels):
        subset = [row for row in rows if row["panel"] == panel]
        ax.set_title(title, fontsize=11.5, weight="bold", pad=9)
        ax.text(
            0.5,
            -0.09,
            takeaway,
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color=NEUTRAL,
            style="italic",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-2.55, 2.45)
        ax.set_ylim(-2.15, 2.25)
        for spine in ax.spines.values():
            spine.set_color(GRID)
            spine.set_linewidth(1.0)

        if panel in {"stage1_patient_clusters", "confusion_valley"}:
            for patient in PATIENT_COLORS:
                for klass in ["non-seizure", "seizure"]:
                    group = [
                        r for r in subset if r["patient_id"] == patient and r["class"] == klass
                    ]
                    ax.scatter(
                        [float(r["x"]) for r in group],
                        [float(r["y"]) for r in group],
                        s=27 if klass == "non-seizure" else 32,
                        marker=PATIENT_MARKERS[patient],
                        facecolors=WHITE if klass == "non-seizure" else PATIENT_COLORS[patient],
                        edgecolors=PATIENT_COLORS[patient] if klass == "non-seizure" else INK,
                        linewidths=1.0 if klass == "non-seizure" else 0.45,
                        alpha=0.82,
                    )
        else:
            existing = [r for r in subset if r["is_new_patient"] == "false"]
            for klass, color, marker in [
                ("non-seizure", BLUE, "o"),
                ("seizure", ORANGE, "*"),
            ]:
                group = [r for r in existing if r["class"] == klass]
                ax.scatter(
                    [float(r["x"]) for r in group],
                    [float(r["y"]) for r in group],
                    s=28 if klass == "non-seizure" else 42,
                    marker=marker,
                    facecolors=BLUE_LIGHT if klass == "non-seizure" else ORANGE,
                    edgecolors=BLUE if klass == "non-seizure" else INK,
                    linewidths=0.75,
                    alpha=0.75,
                    label="Non-seizure" if klass == "non-seizure" else "Seizure",
                )
            new = [r for r in subset if r["is_new_patient"] == "true"]
            if new:
                for klass, color in [("non-seizure", BLUE), ("seizure", ORANGE)]:
                    group = [r for r in new if r["class"] == klass]
                    ax.scatter(
                        [float(r["x"]) for r in group],
                        [float(r["y"]) for r in group],
                        s=64,
                        marker="^",
                        facecolor=WHITE if klass == "non-seizure" else color,
                        edgecolor=INK,
                        linewidth=1.2,
                        alpha=0.95,
                        label=f"New patient: {klass}",
                    )
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(
                unique.values(),
                unique.keys(),
                loc="lower right",
                frameon=True,
                facecolor=WHITE,
                edgecolor=GRID,
                fontsize=7.4,
                handletextpad=0.35,
            )

    fig.text(
        0.5,
        0.025,
        "SIMULATED source rows are committed with a fixed random seed; coordinates are conceptual and must not be read as model results.",
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )
    save_figure(
        fig,
        "figure-03-feature-space-evolution-simulated",
        "Simulated REALM feature-space evolution",
        "Four simulated embedding panels show patient-specific clusters after Stage 1, blurred patient boundaries "
        "during adversarial training, separate non-seizure and seizure clusters after Stage 2, and samples from "
        "a new patient mapping into the two shared seizure-state clusters. These are conceptual coordinates, not "
        "measured model embeddings.",
    )


def generate_eeg_shap_data(path: Path) -> None:
    seed = 40404
    rng = np.random.default_rng(seed)
    sampling_rate = 200
    time = np.linspace(0.0, 2.0, 2 * sampling_rate + 1)
    onset = 0.68
    offset = 1.36
    rise = np.clip((time - onset) / 0.14, 0.0, 1.0)
    fall = np.clip((offset - time) / 0.14, 0.0, 1.0)
    window = np.minimum(rise, fall)
    channels = ["Fp1", "F3", "C3", "P3", "Fp2", "F4", "C4", "P4"]
    weights = [0.52, 0.72, 1.00, 0.92, 0.78, 0.88, 0.96, 0.70]
    rows: list[dict[str, object]] = []

    for index, (channel, weight) in enumerate(zip(channels, weights)):
        noise = rng.normal(0, 8.0, time.size)
        noise = np.convolve(noise, np.ones(3) / 3.0, mode="same")
        background = 5.0 * np.sin(2 * np.pi * (5.0 + 0.35 * index) * time + 0.5 * index)
        rhythmic = (
            weight
            * window
            * (
                40.0 * np.sin(2 * np.pi * (10.0 + 0.45 * index) * time + 0.22 * index)
                + 9.0 * np.sin(2 * np.pi * 5.2 * time)
            )
        )
        signal = background + noise + rhythmic
        attribution = np.clip(
            window * weight * (0.72 + 0.22 * np.sin(2 * np.pi * 2.2 * time + index) ** 2)
            + rng.normal(0, 0.015, time.size),
            0.0,
            1.0,
        )
        attribution = np.convolve(attribution, np.ones(5) / 5.0, mode="same")
        for t, value, attr in zip(time, signal, attribution):
            rows.append(
                {
                    "time_s": f"{t:.5f}",
                    "channel": channel,
                    "signal_uv": f"{value:.5f}",
                    "positive_attribution": f"{attr:.6f}",
                    "seizure_state": "ictal" if onset <= t <= offset else "non-ictal",
                    "sampling_rate_hz": sampling_rate,
                    "onset_s": onset,
                    "offset_s": offset,
                    "status": "SIMULATED",
                    "random_seed": seed,
                }
            )

    write_csv(
        path,
        [
            "time_s",
            "channel",
            "signal_uv",
            "positive_attribution",
            "seizure_state",
            "sampling_rate_hz",
            "onset_s",
            "offset_s",
            "status",
            "random_seed",
        ],
        rows,
    )


def figure_04_shap_eeg() -> None:
    path = DATA_DIR / "figure-04-simulated-eeg-shap.csv"
    rows = read_csv(path)
    channels = ["Fp1", "F3", "C3", "P3", "Fp2", "F4", "C4", "P4"]
    onset = float(rows[0]["onset_s"])
    offset = float(rows[0]["offset_s"])
    cmap = LinearSegmentedColormap.from_list(
        "realm_attribution", [WHITE, GOLD_LIGHT, ORANGE_LIGHT, ORANGE]
    )

    fig, axes = plt.subplots(8, 1, sharex=True, figsize=(13.4, 8.7), gridspec_kw={"hspace": 0.08})
    fig.subplots_adjust(left=0.10, right=0.89, top=0.80, bottom=0.10)
    fig.text(
        0.035,
        0.958,
        "REALM Stage 4: positive attribution on an eight-channel EEG trace",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.035,
        0.915,
        "Warm shading encodes simulated seizure-supporting attribution; traces retain amplitude in microvolts.",
        fontsize=10.5,
        color=NEUTRAL,
    )
    add_badge(fig, "SIMULATED • SYNTHETIC EEG + ATTRIBUTION", ORANGE)

    image_for_colorbar = None
    for index, (ax, channel) in enumerate(zip(axes, channels)):
        group = [row for row in rows if row["channel"] == channel]
        time = np.asarray([float(row["time_s"]) for row in group])
        signal = np.asarray([float(row["signal_uv"]) for row in group])
        attribution = np.asarray([float(row["positive_attribution"]) for row in group])
        y_limit = 75.0
        image_for_colorbar = ax.imshow(
            attribution[np.newaxis, :],
            extent=[time.min(), time.max(), -y_limit, y_limit],
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            alpha=0.78,
            zorder=0,
        )
        ax.plot(time, signal, color=INK, linewidth=1.05, zorder=2)
        ax.axvline(onset, color=INK, linestyle=(0, (5, 3)), linewidth=1.15, zorder=3)
        ax.axvline(offset, color=ORANGE, linestyle=(0, (1.5, 2.2)), linewidth=1.4, zorder=3)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_ylabel(
            channel, rotation=0, ha="right", va="center", labelpad=15, fontsize=10, weight="bold"
        )
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_visible(index == len(channels) - 1)
        if index == 0:
            ax.text(
                onset + 0.015, 69, "Seizure onset", fontsize=8.8, weight="bold", color=INK, va="top"
            )
            ax.text(
                offset + 0.015,
                69,
                "Seizure offset",
                fontsize=8.8,
                weight="bold",
                color=ORANGE,
                va="top",
            )

    axes[-1].set_xlabel("Time (seconds)", fontsize=10.5, weight="bold")
    axes[-1].set_xticks(np.arange(0.0, 2.01, 0.2))
    axes[-1].set_xlim(0.0, 2.0)
    if image_for_colorbar is not None:
        colorbar_ax = fig.add_axes([0.915, 0.22, 0.018, 0.45])
        colorbar = fig.colorbar(image_for_colorbar, cax=colorbar_ax)
        colorbar.set_label("Positive attribution\n(simulated, 0–1)", fontsize=9, color=INK)
        colorbar.set_ticks([0.0, 0.5, 1.0])
        colorbar.set_ticklabels(["Low", "Medium", "High"])

    axes[-1].plot([0.04, 0.04], [-58, -8], color=INK, linewidth=2.2, clip_on=False)
    axes[-1].text(0.052, -33, "50 µV", va="center", fontsize=8.5, color=INK)
    fig.text(
        0.5,
        0.035,
        "SIMULATED source rows include time, channel, signal amplitude, attribution, seizure state, sampling rate, and fixed random seed.",
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )
    save_figure(
        fig,
        "figure-04-shap-eeg-simulated",
        "Simulated REALM SHAP-style attribution on EEG",
        "Eight synthetic EEG channels are plotted from zero to two seconds. A seizure begins at 0.68 seconds "
        "and ends at 1.36 seconds. Warm orange positive attribution is concentrated over the rhythmic ictal "
        "activity and is strongest in central and parietal channels. These are simulated EEG and attribution "
        "values, not patient data or measured model output.",
    )


def generate_performance_data(path: Path) -> None:
    rows: list[dict[str, object]] = []
    observations = {
        "AUC": {
            "REALM": [(1, 0.812, 0.042), (5, 0.881, 0.031), (10, 0.923, 0.024), (20, 0.941, 0.019)],
            "Transfer learning": [
                (1, 0.702, 0.058),
                (5, 0.774, 0.044),
                (10, 0.821, 0.038),
                (20, 0.856, 0.029),
            ],
        },
        "Sensitivity": {
            "REALM": [(1, 0.793, 0.042), (5, 0.858, 0.031), (10, 0.901, 0.023), (20, 0.919, 0.019)],
            "Transfer learning": [
                (1, 0.671, 0.058),
                (5, 0.741, 0.044),
                (10, 0.789, 0.037),
                (20, 0.832, 0.029),
            ],
        },
    }
    for metric, methods in observations.items():
        for method, values in methods.items():
            for k, mean, sd in values:
                rows.append(
                    {
                        "record_type": "estimate",
                        "metric": metric,
                        "k": k,
                        "method": method,
                        "mean": f"{mean:.3f}",
                        "sd": f"{sd:.3f}",
                        "reference": "",
                        "value": "",
                        "unit": "proportion",
                        "status": "ILLUSTRATIVE",
                        "notes": "Hypothetical values for framework exposition; not measured performance",
                    }
                )
    references = {
        "AUC": {"Backbone-only floor": 0.662, "Patient-specific ceiling": 0.958},
        "Sensitivity": {"Backbone-only floor": 0.638, "Patient-specific ceiling": 0.933},
    }
    for metric, values in references.items():
        for reference, value in values.items():
            rows.append(
                {
                    "record_type": "reference",
                    "metric": metric,
                    "k": "",
                    "method": "",
                    "mean": "",
                    "sd": "",
                    "reference": reference,
                    "value": f"{value:.3f}",
                    "unit": "proportion",
                    "status": "ILLUSTRATIVE",
                    "notes": "Hypothetical reference value; not measured performance",
                }
            )
    write_csv(
        path,
        [
            "record_type",
            "metric",
            "k",
            "method",
            "mean",
            "sd",
            "reference",
            "value",
            "unit",
            "status",
            "notes",
        ],
        rows,
    )


def figure_05_performance() -> None:
    path = DATA_DIR / "figure-05-illustrative-performance.csv"
    rows = read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.73, bottom=0.17, wspace=0.13)
    fig.text(
        0.035,
        0.955,
        "REALM Stage 3: few-shot adaptation versus labelled examples",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.035,
        0.912,
        "Hypothetical cross-patient performance with one-standard-deviation error bars; focused scale 0.50–1.00.",
        fontsize=10.5,
        color=NEUTRAL,
    )
    add_badge(fig, "ILLUSTRATIVE • NOT EMPIRICAL PERFORMANCE", ORANGE)

    method_styles = {
        "REALM": {"color": BLUE, "linestyle": "-", "marker": "o", "mfc": BLUE},
        "Transfer learning": {
            "color": ORANGE,
            "linestyle": (0, (5, 3)),
            "marker": "s",
            "mfc": WHITE,
        },
    }
    reference_styles = {
        "Backbone-only floor": (0, (5, 3)),
        "Patient-specific ceiling": (0, (1.5, 2.2)),
    }
    for ax, metric in zip(axes, ["AUC", "Sensitivity"]):
        estimates = [r for r in rows if r["record_type"] == "estimate" and r["metric"] == metric]
        for method, style in method_styles.items():
            group = sorted(
                (r for r in estimates if r["method"] == method), key=lambda r: int(r["k"])
            )
            x = np.asarray([int(r["k"]) for r in group])
            y = np.asarray([float(r["mean"]) for r in group])
            sd = np.asarray([float(r["sd"]) for r in group])
            ax.errorbar(
                x,
                y,
                yerr=sd,
                label=method,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markerfacecolor=style["mfc"],
                markeredgecolor=INK,
                markeredgewidth=1.0,
                markersize=7.5,
                linewidth=2.4,
                capsize=3.5,
                capthick=1.5,
                zorder=3,
            )
        references = [r for r in rows if r["record_type"] == "reference" and r["metric"] == metric]
        for reference in references:
            value = float(reference["value"])
            ax.axhline(
                value,
                color=NEUTRAL,
                linestyle=reference_styles[reference["reference"]],
                linewidth=1.35,
                zorder=1,
            )
            ax.text(
                20.45,
                value + (0.008 if reference["reference"].startswith("Patient") else 0.012),
                reference["reference"].replace(" ", "\n", 1),
                ha="right",
                va="bottom",
                fontsize=8,
                color=NEUTRAL,
                style="italic",
                linespacing=0.92,
            )
        ax.set_title(f"Cross-patient {metric}", fontsize=12.5, weight="bold", pad=10)
        ax.set_xlabel("Labelled seizure examples (K)", fontsize=10, weight="bold")
        ax.set_xticks([1, 5, 10, 20])
        ax.set_xticklabels(["K=1", "K=5", "K=10", "K=20"])
        ax.set_xlim(0, 21.5)
        ax.set_ylim(0.50, 1.00)
        ax.set_yticks(np.arange(0.50, 1.01, 0.05))
        ax.grid(axis="y", color=GRID, linestyle=(0, (3, 3)), linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Metric value (proportion)", fontsize=10, weight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.835),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.6,
    )
    fig.text(
        0.5,
        0.045,
        "ILLUSTRATIVE values only. Replace the CSV with measured experiment outputs before making performance claims.",
        ha="center",
        fontsize=8.8,
        color=NEUTRAL,
        weight="bold",
    )
    save_figure(
        fig,
        "figure-05-few-shot-performance-illustrative",
        "Illustrative REALM few-shot adaptation performance",
        "Two illustrative line charts compare REALM with transfer learning at K values of 1, 5, 10, and 20 "
        "labelled seizure examples. Hypothetical REALM AUC and sensitivity remain above transfer learning and "
        "approach hypothetical patient-specific references as K increases. Error bars are hypothetical one "
        "standard deviation values; no empirical performance is shown.",
    )


def ensure_source_data(refresh: bool) -> None:
    generators = {
        DATA_DIR / "figure-03-simulated-embedding.csv": generate_embedding_data,
        DATA_DIR / "figure-04-simulated-eeg-shap.csv": generate_eeg_shap_data,
        DATA_DIR / "figure-05-illustrative-performance.csv": generate_performance_data,
    }
    for path, generator in generators.items():
        if refresh or not path.exists():
            generator(path)
            print(f"wrote {path.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Deterministically regenerate simulated/illustrative CSV files before plotting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    ensure_source_data(args.refresh_data)
    figure_01_pipeline()
    figure_02_gradient_reversal()
    figure_03_feature_space()
    figure_04_shap_eeg()
    figure_05_performance()


if __name__ == "__main__":
    main()
