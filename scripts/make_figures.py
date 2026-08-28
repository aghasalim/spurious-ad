"""Draw the README figures from reports/*.csv.

Reads the saved sweeps only, no MVTec download, no training. A figure here can
never disagree with the numbers quoted in the README, because it has no way to
produce a number of its own.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from PIL import Image

from style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# One colour per detector family, the same in every figure, so a reader who
# learns them once does not have to check the legend again. Grey is always a
# control: the random-heatmap null, or the mark scored on its own.
DETECTORS = {"padim": PALETTE[0], "patchcore": PALETTE[1]}
CONTROL = PALETTE[5]

RHO = "confound-label correlation rho (0 = independent, 1 = perfect)"
AUROC = "image AUROC (0 to 1)"
PEAK = "peaks on the real defect\n(fraction of anomalous images)"
CAR = "confound attribution ratio (0 to 1)"


def sweep() -> pd.DataFrame:
    return pd.read_csv(REPORTS / "real_sweep_summary.csv")


def _bottom_legend(figure, ax, ncol: int) -> None:
    """One legend under the whole figure, anchored by its top edge.

    Inside the axes it would sit on a line in at least one of these figures, and
    anchored by its bottom edge it climbs back into the x labels.
    """
    figure.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=ncol,
                  bbox_to_anchor=(0.5, -0.005))


def dissociation(out: Path) -> Path:
    """The result the benchmark exists to produce.

    As the planted confound strengthens, image AUROC goes up and the fraction of
    heatmap peaks landing on the actual defect goes down. The effect is real and
    reproducible.

    It is not, however, a label shortcut. :func:`mechanism` pins the training
    confound rate and the whole collapse disappears. This figure shows what the
    naive construction produces; that one shows why it is the wrong thing to
    measure.
    """
    table = sweep()

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True)
    for detector, colour in DETECTORS.items():
        rows = table[table.detector == detector].sort_values("rho")
        left.plot(rows.rho, rows.auroc, "o-", color=colour, label=detector)
        right.plot(rows.rho, rows.peak_on_defect, "o-", color=colour, label=detector)

    left.set_ylabel(AUROC)
    left.set_ylim(0.86, 1.02)
    titled(left, "The score everyone reports goes up",
           "5 MVTec categories, 3 seeds, wide_resnet50_2")
    left.legend(loc="lower right")

    right.set_ylabel(PEAK)
    right.set_ylim(0.28, 0.86)
    titled(right, "The hottest pixel walks off the defect",
           "the same runs, on the same images")

    for ax in (left, right):
        ax.set_xlabel(RHO)
        ax.set_xticks(sorted(table.rho.unique()))
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def confound_alone(out: Path) -> Path:
    """What the mark predicts on its own, against what the detectors score.

    ``confound_auroc`` is the AUROC obtainable from the artefact alone, ignoring
    the image. At rho=1 it is 1.000, so a perfect score is available there
    without doing any anomaly detection at all.
    """
    table = sweep()
    alone = table.drop_duplicates("rho").sort_values("rho")

    figure, ax = plt.subplots(figsize=(9.5, 4.9))
    ax.plot(alone.rho, alone.confound_auroc, "s--", color=CONTROL,
            label="the mark alone, image never opened")
    for detector, colour in DETECTORS.items():
        rows = table[table.detector == detector].sort_values("rho")
        ax.plot(rows.rho, rows.auroc, "o-", color=colour, label=detector)
    ax.axhline(0.5, color="#999999", lw=1.0, ls=":")
    ax.text(0.02, 0.505, "coin flip", fontsize=9, color="#777777", va="bottom")

    ax.set_xlabel(RHO)
    ax.set_ylabel(AUROC)
    ax.set_xticks(sorted(table.rho.unique()))
    ax.set_ylim(0.45, 1.05)
    titled(ax, "At full correlation the mark alone scores a perfect AUROC",
           "so the detectors' 0.993 and 0.998 there are available without "
           "looking at the image")
    ax.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def attribution(out: Path) -> Path:
    """Confound attribution against the random-attribution baseline.

    ``car`` is the share of the anomaly score that lands on the confound region;
    ``car_random`` is what an uninformative heatmap gives on the same two
    regions. Only the gap means anything, and it never opens.
    """
    table = sweep()

    figure, ax = plt.subplots(figsize=(9.7, 5.0))
    # The two detectors sit within 0.02 of each other at four of the five stops,
    # so the spread bars are nudged apart. Only the bars move, not the points.
    for dodge, (detector, colour) in zip((-0.012, 0.012), DETECTORS.items()):
        rows = table[table.detector == detector].sort_values("rho")
        ax.plot(rows.rho, rows.car, "o-", color=colour, label=detector)
        ax.errorbar(rows.rho + dodge, rows.car, yerr=rows.car_sd, fmt="none",
                    ecolor=colour, capsize=3, elinewidth=1.0, alpha=0.7)
    null = table.drop_duplicates("rho").sort_values("rho")
    ax.plot(null.rho, null.car_random, "s--", color=CONTROL,
            label="random heatmap on the same two regions")

    ax.set_xlabel(RHO)
    ax.set_ylabel(CAR)
    ax.set_xticks(sorted(table.rho.unique()))
    ax.set_ylim(0.0, 0.75)
    titled(ax, "Attribution to the mark never passes what a random heatmap gives",
           "5 MVTec categories. point is the mean over anomalous images, bar is "
           "one across-image sd, nudged apart, not an error on the mean")
    _bottom_legend(figure, ax, ncol=3)
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def by_category(out: Path) -> Path:
    """Per-category localisation, weakest confound against strongest.

    Open marker is rho=0, filled marker is rho=1, one row per detector and
    category. Bars would have worked here too, but the thing worth reading is
    the length of the move, and a dumbbell puts that on one line.
    """
    table = pd.read_csv(REPORTS / "real_sweep_by_category.csv")
    low, high = table.rho.min(), table.rho.max()
    categories = sorted(table.category.unique())
    detectors = list(DETECTORS)

    figure, ax = plt.subplots(figsize=(10, 5.6))
    for row, category in enumerate(categories):
        for offset, detector in zip((-0.19, 0.19), detectors):
            cell = table[(table.category == category) & (table.detector == detector)]
            a = cell[cell.rho == low].peak_on_defect.iloc[0]
            b = cell[cell.rho == high].peak_on_defect.iloc[0]
            y = row + offset
            colour = DETECTORS[detector]
            ax.plot([a, b], [y, y], color=colour, lw=2.2, alpha=0.45,
                    solid_capstyle="round", zorder=1)
            ax.plot([a], [y], "o", color=colour, markerfacecolor="white",
                    markersize=7, zorder=2)
            ax.plot([b], [y], "o", color=colour, markersize=7, zorder=2)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_ylim(len(categories) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("peaks on the real defect (fraction of anomalous images)")
    titled(ax, "Four of the five categories lose the defect, so no one category "
               "carries the effect",
           f"open marker rho={low:g}, filled marker rho={high:g}. bottle is the "
           "exception: its heat never moves to the mark")
    marks = [(Line2D([], [], color=DETECTORS[d], marker="o", lw=2.2, alpha=0.7), d)
             for d in detectors]
    marks += [
        (Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
                lw=0), f"rho={low:g}"),
        (Line2D([], [], color="#555555", marker="o", lw=0), f"rho={high:g}"),
    ]
    figure.legend([h for h, _ in marks], [t for _, t in marks],
                  loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def backbone(out: Path) -> Path:
    """Does a different feature extractor change the conclusion? It does not."""
    default = sweep()
    other = pd.read_csv(REPORTS / "real_backbone_summary.csv")

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True)
    for label, table, style in (("wide_resnet50_2", default, "o-"),
                                ("resnet18", other, "s--")):
        for detector, colour in DETECTORS.items():
            rows = table[table.detector == detector].sort_values("rho")
            left.plot(rows.rho, rows.auroc, style, color=colour,
                      label=f"{detector}, {label}")
            right.plot(rows.rho, rows.peak_on_defect, style, color=colour)

    left.set_ylabel(AUROC)
    left.set_ylim(0.86, 1.02)
    titled(left, "The score rises on either backbone",
           "solid wide_resnet50_2, dashed resnet18, same images and seeds")

    right.set_ylabel(PEAK)
    right.set_ylim(0.28, 0.90)
    titled(right, "And the peak leaves the defect on either backbone",
           "so this is not a property of one feature extractor")

    for ax in (left, right):
        ax.set_xlabel(RHO)
        ax.set_xticks(sorted(default.rho.unique()))
    _bottom_legend(figure, left, ncol=4)
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def mechanism(out: Path) -> Path:
    """The ablation that refutes the label-shortcut reading.

    Raising rho does two things at once: it makes the mark predict the label, and
    it drives the mark out of the normal-only training set. Pinning the training
    confound rate at 0.51 separates them, and with it pinned the collapse
    disappears even at rho=1, where the mark predicts the label perfectly.

    An unsupervised detector never sees a label, so a label shortcut is not
    mechanically available to it. What it reacts to is a distributional
    departure, and flagging that is arguably correct behaviour.
    """
    table = pd.read_csv(REPORTS / "real_mechanism_summary.csv")
    pinned_rate = table[table.pinned_train_rate].train_conf_rate.max()

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.9), sharex=True)
    for detector, colour in DETECTORS.items():
        for pinned, style, suffix in ((False, "o-", "training rate free"),
                                      (True, "s--", "training rate pinned")):
            rows = table[(table.detector == detector)
                         & (table.pinned_train_rate == pinned)].sort_values("rho")
            left.plot(rows.rho, rows.peak_on_defect, style, color=colour,
                      alpha=1.0 if not pinned else 0.65,
                      label=f"{detector}, {suffix}")
            right.plot(rows.rho, rows.car, style, color=colour,
                       alpha=1.0 if not pinned else 0.65)
    null = table.drop_duplicates("rho").sort_values("rho")
    right.plot(null.rho, null.car_random, ":", color=CONTROL, lw=1.6)
    right.text(0.02, null.car_random.iloc[0] + 0.012, "random-heatmap null",
               fontsize=9, color=CONTROL, va="bottom")

    left.set_ylabel(PEAK)
    left.set_ylim(0.28, 0.86)
    titled(left, "Keep the mark in the training set and the peak stays put",
           f"dashed: P(mark | normal) pinned at {pinned_rate:.2f} while rho moves")

    right.set_ylabel(CAR)
    right.set_ylim(0.0, 0.55)
    titled(right, "So it was the mark going missing, not the label",
           "with the rate pinned, CAR is flat at perfect label correlation")

    for ax in (left, right):
        ax.set_xlabel(RHO)
        ax.set_xticks(sorted(table.rho.unique()))
    _bottom_legend(figure, left, ncol=2)
    figure.tight_layout()
    figure.savefig(out)
    plt.close(figure)
    return out


def _shrink(path: Path) -> Path:
    """Rewrite every frame of a GIF onto one shared palette. Halves the file."""
    src = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(src.convert("RGB"))
            durations.append(src.info.get("duration", 62))
            src.seek(src.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(64, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)
    return path


def anim_sweep(out: Path) -> Path:
    """The two arms of the argument, walked from rho=0 to rho=1.

    Free arms come from real_sweep_summary.csv, which was run at five values of
    rho. Pinned arms come from the pinned rows of real_mechanism_summary.csv,
    run at three. Every vertex is one of those measured rows; between two
    vertices the head slides along the segment the static figures already draw,
    and the readout says it is between stops rather than quoting a value that
    was never measured.

    The free and pinned rows at rho=0 are the same run, which is why the arms
    start on top of each other.
    """
    free = sweep()
    pinned = pd.read_csv(REPORTS / "real_mechanism_summary.csv")
    pinned = pinned[pinned.pinned_train_rate]
    rate = pinned.train_conf_rate.max()
    stops = sorted(free.rho.unique())

    park, move, tail = 6, 9, 18
    # (rho, "from", "to"): parked frames repeat a stop, moving frames ease
    # between two stops with a smoothstep so the head does not jerk.
    schedule = []
    for i, rho in enumerate(stops[:-1]):
        schedule += [(rho, rho, rho)] * park
        for k in range(1, move + 1):
            t = k / (move + 1)
            schedule.append((rho + (stops[i + 1] - rho) * t * t * (3 - 2 * t),
                             rho, stops[i + 1]))
    schedule += [(stops[-1], stops[-1], stops[-1])] * (park + tail)

    figure, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.set_xlim(0.87, 1.015)
    ax.set_ylim(0.30, 0.84)
    ax.set_xlabel(AUROC)
    ax.set_ylabel(PEAK)
    ax.set_title("Pin the mark into the training set and the collapse never "
                 "starts", pad=26)
    # Same layout as style.titled, but the second line changes per frame.
    ticker = ax.text(0.0, 1.012, "", transform=ax.transAxes, fontsize=9.3,
                     color="#5a5a5a", va="bottom", ha="left")
    ax.annotate("ideal corner", xy=(1.006, 0.812), xytext=(0.962, 0.784),
                fontsize=9, color="#8a8a8a", va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=1.0))

    arms, heads = {}, {}
    for detector, colour in DETECTORS.items():
        for arm, style, alpha in (("free", "o-", 1.0), ("pinned", "s--", 0.72)):
            table = free if arm == "free" else pinned
            rows = table[table.detector == detector].sort_values("rho")
            arms[detector, arm] = (rows.rho.values, rows.auroc.values,
                                   rows.peak_on_defect.values)
            line, = ax.plot([], [], style, color=colour, alpha=alpha,
                            label=f"{detector}, training rate {arm}")
            heads[detector, arm] = (line, ax.plot([], [], style[0], color=colour,
                                                  alpha=alpha, markersize=9)[0])
    verdict = ax.text(0.876, 0.315, "", fontsize=9.5, color="#333333", va="bottom")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.13))

    def draw(frame: int):
        rho, low, high = schedule[frame]
        for key, (grid, auroc, peak) in arms.items():
            line, head = heads[key]
            keep = grid <= rho + 1e-9
            x = np.append(auroc[keep], np.interp(rho, grid, auroc))
            y = np.append(peak[keep], np.interp(rho, grid, peak))
            line.set_data(x, y)
            head.set_data(x[-1:], y[-1:])
        where = (f"rho = {rho:.2f}" if low == high
                 else f"rho between {low:.2f} and {high:.2f}")
        ticker.set_text(f"{where}. free arms let P(mark | normal) fall to 0, "
                        f"pinned arms hold it at {rate:.2f}")
        if rho == stops[-1]:
            lo = free[free.rho == stops[-1]].peak_on_defect
            hi = pinned[pinned.rho == stops[-1]].peak_on_defect
            verdict.set_text(f"at rho=1 the free arms sit at {lo.min():.2f} and "
                             f"{lo.max():.2f} on the defect,\nthe pinned arms "
                             f"at {hi.min():.2f} and {hi.max():.2f}")
        else:
            verdict.set_text("")
        return (*(h for pair in heads.values() for h in pair), ticker, verdict)

    anim = FuncAnimation(figure, draw, frames=len(schedule), blit=False)
    # dpi is passed here on purpose: the style saves stills at 170, and these
    # frames at that size would be a multi-megabyte GIF for no extra detail.
    anim.save(out, writer=PillowWriter(fps=15), dpi=100)
    plt.close(figure)
    return _shrink(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for path in (
        dissociation(FIGURES / "dissociation.png"),
        mechanism(FIGURES / "mechanism.png"),
        confound_alone(FIGURES / "confound-alone.png"),
        attribution(FIGURES / "attribution.png"),
        by_category(FIGURES / "by-category.png"),
        backbone(FIGURES / "backbone.png"),
        anim_sweep(FIGURES / "rho-sweep.gif"),
    ):
        print(f"-> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
