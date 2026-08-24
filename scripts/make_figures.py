"""Draw the README figures from reports/*.csv.

Reads the saved sweeps only -- no MVTec download, no training.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

DETECTORS = {"padim": "#2166ac", "patchcore": "#b2182b"}


def sweep() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "real_sweep_summary.csv")


def dissociation(out: Path) -> Path:
    """The result the benchmark exists to produce.

    As the planted confound strengthens, image-level AUROC goes *up* and the
    fraction of heatmap peaks landing on the actual defect goes *down*. The effect
    is real and reproducible.

    It is not, however, a label shortcut -- see :func:`mechanism`, which pins the
    training confound rate and makes the whole collapse disappear. This figure
    shows what the naive construction produces; the next one shows why that is the
    wrong thing to measure.
    """
    table = sweep()

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for detector, colour in DETECTORS.items():
        rows = table[table.detector == detector].sort_values("rho")
        left.plot(rows.rho, rows.auroc, "o-", color=colour, lw=2, label=detector)
        right.plot(rows.rho, rows.peak_on_defect, "o-", color=colour, lw=2,
                   label=detector)
    left.set_ylabel("image AUROC")
    left.set_title("the score everyone reports goes up", fontsize=10)
    right.set_ylabel("fraction of peaks on the real defect")
    right.set_title("while the model stops looking at the defect", fontsize=10)
    for ax in (left, right):
        ax.set_xlabel("confound strength $\\rho$")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Same detectors, same defects, only the confound strength changes. "
        "Real effect --\nbut the mechanism figure shows it is not the one it "
        "looks like.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def confound_alone(out: Path) -> Path:
    """What the confound predicts on its own, against what the detector scores.

    ``confound_auroc`` is the AUROC obtainable from the artefact alone, ignoring
    the image. At rho=1.0 it reaches 1.000, so a perfect score there is available
    without doing any anomaly detection at all.
    """
    table = sweep()
    rhos = sorted(table.rho.unique())
    base = np.arange(len(rhos))

    figure, ax = plt.subplots(figsize=(9.5, 4.6))
    confound = table.drop_duplicates("rho").sort_values("rho")
    ax.bar(base, confound.confound_auroc, 0.55, color="#bdbdbd",
           edgecolor="0.3", lw=0.5, label="confound alone (no image)")
    for detector, colour in DETECTORS.items():
        rows = table[table.detector == detector].sort_values("rho")
        ax.plot(base, rows.auroc, "o-", color=colour, lw=2, label=detector)
    ax.set_xticks(base)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlabel("confound strength $\\rho$")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.set_title(
        "At $\\rho=1$ the artefact alone scores 1.000, so the detector's near-"
        "perfect AUROC\nis available without looking at the image.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def attribution(out: Path) -> Path:
    """Confound attribution against the random-attribution baseline.

    ``car`` is the share of the anomaly score attributable to the confound region;
    ``car_random`` is what an uninformative attribution gives on the same regions.
    The gap is what matters, and it only opens up at the strongest confound.
    """
    table = sweep()

    figure, ax = plt.subplots(figsize=(9.5, 4.6))
    for detector, colour in DETECTORS.items():
        rows = table[table.detector == detector].sort_values("rho")
        ax.errorbar(rows.rho, rows.car, yerr=rows.car_sd, fmt="o-", color=colour,
                    lw=2, capsize=3, label=f"{detector} (measured)")
    baseline = table.drop_duplicates("rho").sort_values("rho")
    ax.plot(baseline.rho, baseline.car_random, "s--", color="0.45", lw=1.6,
            label="random attribution")
    ax.set_xlabel("confound strength $\\rho$")
    ax.set_ylabel("confound attribution ratio")
    ax.set_title(
        "Attribution to the confound stays below the random baseline until "
        "$\\rho=1$,\nwhich is why attribution alone is a weak detector of this "
        "failure.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def by_category(out: Path) -> Path:
    """Per-category localisation collapse between the weakest and strongest confound."""
    table = pd.read_csv(REPORTS / "real_sweep_by_category.csv")
    table = table[table.detector == "padim"]
    low = table[table.rho == table.rho.min()].set_index("category")
    high = table[table.rho == table.rho.max()].set_index("category")
    categories = sorted(set(low.index) & set(high.index))
    positions = np.arange(len(categories))

    figure, ax = plt.subplots(figsize=(11, 5.4))
    ax.barh(positions - 0.2, [low.loc[c, "peak_on_defect"] for c in categories], 0.4,
            label=f"$\\rho$={table.rho.min():g}", color="#2166ac",
            edgecolor="0.3", lw=0.4)
    ax.barh(positions + 0.2, [high.loc[c, "peak_on_defect"] for c in categories], 0.4,
            label=f"$\\rho$={table.rho.max():g}", color="#b2182b",
            edgecolor="0.3", lw=0.4)
    ax.set_yticks(positions)
    ax.set_yticklabels(categories, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("fraction of peaks on the real defect")
    ax.set_title(
        "PaDiM, per MVTec category. The collapse is not driven by one category.",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def backbone(out: Path) -> Path:
    """Does a different backbone change the conclusion?

    If the dissociation were an artefact of one feature extractor, swapping it
    would remove it. It does not.
    """
    default = sweep()
    other = pd.read_csv(REPORTS / "real_backbone_summary.csv")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for label, table, style in (("default backbone", default, "o-"),
                                ("alternate backbone", other, "s--")):
        for detector, colour in DETECTORS.items():
            rows = table[table.detector == detector].sort_values("rho")
            left.plot(rows.rho, rows.auroc, style, color=colour, lw=1.8,
                      label=f"{detector}, {label}")
            right.plot(rows.rho, rows.peak_on_defect, style, color=colour, lw=1.8,
                       label=f"{detector}, {label}")
    left.set_ylabel("image AUROC")
    left.set_title("AUROC", fontsize=10)
    right.set_ylabel("fraction of peaks on the real defect")
    right.set_title("localisation", fontsize=10)
    for ax in (left, right):
        ax.set_xlabel("confound strength $\\rho$")
        ax.spines[["top", "right"]].set_visible(False)
    right.legend(frameon=False, fontsize=7)
    figure.suptitle(
        "The dissociation survives a backbone swap, so it is not a property of "
        "one feature extractor.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def mechanism(out: Path) -> Path:
    """The ablation that refutes the label-shortcut reading.

    Raising rho does two things at once: it makes the mark predict the label, and
    it drives the mark out of the normal-only training set. Pinning the training
    confound rate at 0.465 separates them. With it pinned, the localisation
    collapse disappears entirely even at rho=1.0, where the mark predicts the
    label perfectly.

    An unsupervised detector never sees a label, so a label shortcut is not
    mechanically available to it. What it is reacting to is a distributional
    departure, and flagging that is arguably correct behaviour.
    """
    table = pd.read_csv(REPORTS / "real_mechanism_summary.csv")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for detector, colour in DETECTORS.items():
        for pinned, style, suffix in ((False, "o-", "free"), (True, "s--", "pinned")):
            rows = table[(table.detector == detector)
                         & (table.pinned_train_rate == pinned)].sort_values("rho")
            left.plot(rows.rho, rows.peak_on_defect, style, color=colour, lw=1.9,
                      alpha=1.0 if not pinned else 0.6,
                      label=f"{detector}, training rate {suffix}")
            right.plot(rows.rho, rows.car, style, color=colour, lw=1.9,
                       alpha=1.0 if not pinned else 0.6,
                       label=f"{detector}, training rate {suffix}")
    left.set_ylabel("fraction of peaks on the real defect")
    left.set_title("localisation", fontsize=10)
    right.set_ylabel("confound attribution ratio")
    right.set_title("attribution to the confound", fontsize=10)
    for ax in (left, right):
        ax.set_xlabel("confound strength $\\rho$")
        ax.spines[["top", "right"]].set_visible(False)
    left.legend(frameon=False, fontsize=7.5)
    figure.suptitle(
        "Dashed: training confound rate pinned at 0.465. The collapse disappears, "
        "so it was\nthe mark leaving the training set -- not the mark predicting "
        "the label.",
        fontsize=10, y=0.02, color="0.35",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        dissociation(FIGURES / "dissociation.png"),
        mechanism(FIGURES / "mechanism.png"),
        confound_alone(FIGURES / "confound-alone.png"),
        attribution(FIGURES / "attribution.png"),
        by_category(FIGURES / "by-category.png"),
        backbone(FIGURES / "backbone.png"),
    ):
        print(f"-> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
