"""Standalone figures for the saved synthetic pilot evaluation."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


# Consistent labels, color and line patterns across all figures.
STYLES = {
    "fedavg": ("Standard FL", "#285d95", "-"),
    "self_evolution": ("FL + self-evolution", "#bc5917", "--"),
    "continual": ("FL + continual", "#087a68", "-."),
    "combined": ("Combined", "#864f8f", ":"),
}
STAGES = [(1, 10, "Stable"), (11, 20, "Sudden change"),
          (21, 30, "Gradual change"), (31, 40, "Return")]


def trajectory_figure(curves, metric, title, caption, path):
    fig, (main, spread) = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(left=.10, right=.975, bottom=.15, top=.77, hspace=.20)
    for method, (label, color, style) in STYLES.items():
        rows = [r for r in curves if r["metric"] == metric and r["method"] == method]
        rounds = [r["round"] for r in rows]
        for ax, key in ((main, "mean"), (spread, "sample_sd")):
            ax.plot(rounds, [r[key] for r in rows], color=color, linestyle=style,
                    label=label, linewidth=2.4)

    first, last = rounds[0], rounds[-1]
    for start, end, stage in STAGES:
        if first <= start <= last:
            main.text((start + end) / 2, 1.035, stage, transform=main.get_xaxis_transform(),
                      ha="center", fontsize=10, color="#475569")
            if start > first:
                for ax in (main, spread):
                    ax.axvline(start - .5, color="#929eaa", linestyle=(0, (3, 3)), lw=.85)
    for ax in (main, spread):
        ax.grid(axis="y")
        ax.set_xlim(first - .35, last + .35)
        ax.set_ylim(bottom=0)
    if metric == "forgetting_pp":
        main.set_ylabel("Mean drop (pp)")
    else:
        main.set_ylim(0, 100)
        main.set_ylabel("Mean accuracy (%)")
    spread.set_ylabel("Seed SD (pp)")
    spread.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    spread.set_xticks([first, *range(first + 4, last + 1, 5)])
    spread.set_xlabel("Communication round")
    fig.suptitle(title, x=.10, y=.966, ha="left", fontsize=20, weight="bold")
    fig.text(.10, .897, "Synthetic pilot results  ·  Five paired seeds  ·  Mean and across-seed sample SD",
             fontsize=11, color="#475569")
    fig.legend(*main.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(.095, .868), ncol=4, frameon=False, handlelength=3)
    fig.text(.10, .043, caption, fontsize=10, color="#364452")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def comparison_figure(metrics, seeds, path):
    fig = plt.figure(figsize=(13.2, 7.4))
    grid = fig.add_gridspec(2, 4, left=.145, right=.974, top=.80, bottom=.18,
                            hspace=.64, wspace=.29)
    panels = [
        (grid[0, :2], "recovery_rounds", "Recovery to 70% on the shifted set", "Completed shifted rounds"),
        (grid[0, 2], "adaptation_examples", "Server practice", "Generated examples"),
        (grid[0, 3], "adaptation_steps", "Server updates", "Extra optimizer steps"),
        (grid[1, 0], "replay_presentations", "Client replay", "Example presentations"),
        (grid[1, 1], "teacher_forward_passes", "Teacher inference", "Batch forward calls"),
        (grid[1, 2], "teacher_presentations", "Teacher inputs", "Examples scored"),
        (grid[1, 3], "memory_capacity_per_device", "Memory capacity", "Examples per device"),
    ]
    index = {(r["method"], r["metric"]): r for r in metrics}
    offsets = np.linspace(-.24, .025, len(seeds))
    for location, metric, title, unit in panels:
        ax = fig.add_subplot(location)
        recovery = metric == "recovery_rounds"
        ticklabels = []
        for i, (method, (label, color, _)) in enumerate(STYLES.items()):
            row = index[method, metric]
            ticklabels.append(f"{label}  ({row['n_observed']}/{len(seeds)})" if recovery else label)
            for seed, offset in zip(seeds, offsets):
                x = row[f"seed_{seed}"]
                # NR is a separate categorical column, never included in the numeric summary.
                ax.scatter(12 if x is None else x, i + offset, s=18, color=color,
                           alpha=.72, marker=">" if x is None else "o", linewidth=0, zorder=3)
            if row["mean"] is not None:
                ax.errorbar(row["mean"], i + .20, xerr=row["sample_sd"], fmt="D",
                            markersize=4.5, capsize=3, color=color, lw=1.4, zorder=3)
        if recovery:
            ax.axvspan(11.15, 12.7, color="#eff2f5")
            ax.set_xlim(.4, 12.7)
            ax.set_xticks([1, 3, 5, 7, 9, 10, 12], ["1", "3", "5", "7", "9", "10", "NR"])
        else:
            ax.margins(x=.09)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.set_ylim(3.55, -.55)
        ax.set_yticks(range(4), ticklabels)
        ax.tick_params(axis="y", labelleft=location.is_first_col(), length=0)
        ax.set_title(title, loc="left", weight="bold", fontsize=12, pad=8)
        ax.set_xlabel(unit, fontsize=10)
        ax.grid(axis="x")
    fig.suptitle("Recovery and additional work", x=.065, y=.96, ha="left", fontsize=21, weight="bold")
    fig.text(.065, .89, "Synthetic pilot results  ·  Five paired seeds  ·  Work per 40-round run",
             fontsize=12, color="#475569")
    fig.text(.065, .08, "Dots show individual seeds. Diamonds and whiskers show the mean and one sample SD.",
             fontsize=10)
    fig.text(.065, .048, "NR = not reached within the shifted stage (rounds 11–20). "
             "Counts show how many seeds reached 70%. Recovery means use those seeds.",
             fontsize=10, color="#475569")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_evaluation(metrics, curves, seeds, directory):
    with plt.rc_context({"font.size": 10, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False,
                         "axes.edgecolor": "#9da7b0", "grid.color": "#dee4e9",
                         "grid.linewidth": .7, "text.color": "#213140", "axes.labelcolor": "#213140",
                         "xtick.color": "#475569", "ytick.color": "#475569", "ytick.major.size": 0}):
        trajectory_figure(curves, "current_accuracy_pct", "Accuracy on the current condition",
                          "Rounds 21–30 use a different fixed test set each round. "
                          "Changes reflect both the model and the test set.",
                          directory / "current_accuracy.png")
        trajectory_figure(curves, "original_accuracy_pct", "Retention on the original condition",
                          "Each seed uses its original 256-example test set for all 40 rounds and all four methods.",
                          directory / "original_accuracy.png")
        trajectory_figure(curves, "forgetting_pp", "Forgetting relative to round 10",
                          "Each seed's drop is measured on its fixed original test set. "
                          "Gains above the round-10 score count as zero forgetting.",
                          directory / "forgetting.png")
        comparison_figure(metrics, seeds, directory / "recovery_and_work.png")
