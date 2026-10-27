"""
Plot the outage effect on each system-strength metric (Gate 4 v2, D4).

One histogram per site and metric of the hourly percentage difference

    r = (metric_contingency - metric_intact) / metric_intact

pooled over every N-1 corridor and hour. Rows are the sites (21, 48, 16),
columns SCR, ESCR and SDSCR; WSCR is one group value, so it gets one panel.
All panels share both axes.

Usage
-----
    python -m experiments.glover37.plot_strength_metrics
    python -m experiments.glover37.plot_strength_metrics --png

Input
-----
results/glover37/strength/analysis/d4_hourly.csv

Output
------
figures/glover37/strength_outage_histograms.pdf (and .png with --png)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "glover37" / "strength" / "analysis" / "d4_hourly.csv"
FIGURES_ROOT = ROOT / "figures" / "glover37"

SITES = ("21", "48", "16")
SITE_METRICS = ("SCR", "ESCR", "SDSCR")
BIN_WIDTH_PCT = 1.0

BAR = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def histogram(ax, r_pct, edges, y_max, title):
    weights = np.full(len(r_pct), 100.0 / len(r_pct))  # share of hours, %
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)  # 0 is a bin edge: r < 0 left, r > 0 right
    ax.hist(r_pct, bins=edges, weights=weights, color=BAR, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    stronger = 100.0 * float((r_pct > 0).mean())
    ax.text(0.03, 0.95, f"r > 0: {stronger:.1f} % of hours", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)


def plot(d):
    d = d.assign(r_pct=100.0 * d["r"])
    limit = BIN_WIDTH_PCT * np.ceil(d["r_pct"].abs().max() / BIN_WIDTH_PCT)
    edges = np.arange(-limit, limit + BIN_WIDTH_PCT, BIN_WIDTH_PCT)
    groups = {(m, s): g["r_pct"].to_numpy() for (m, s), g in d.groupby(["metric", "site"])}
    # 30 % headroom keeps the label above the tallest bar
    y_max = 1.3 * max(np.histogram(v, bins=edges)[0].max() * 100.0 / len(v) for v in groups.values())

    fig, axes = plt.subplots(3, 4, figsize=(11, 6.6), sharex=True, sharey=True)
    for i, site in enumerate(SITES):
        for j, metric in enumerate(SITE_METRICS):
            histogram(axes[i, j], groups[metric, site], edges, y_max, f"{metric}, site {site}")
    histogram(axes[0, 3], groups["WSCR", "group"], edges, y_max, "WSCR, group")
    for ax in axes[1:, 3]:
        ax.set_visible(False)
    for ax in axes[-1, :3]:
        ax.set_xlabel("r = (contingency − intact) / intact  [%]", fontsize=8, color=TEXT)
    axes[0, 3].tick_params(labelbottom=True)
    axes[0, 3].set_xlabel("r [%]", fontsize=8, color=TEXT)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of hours [%]", fontsize=8, color=TEXT)
    n_corridors = d["case"].nunique()
    fig.suptitle(f"Outage effect per metric and site: {n_corridors} N-1 corridors × 168 h, "
                 f"{BIN_WIDTH_PCT:g} %-point bins", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--png", action="store_true", help="Also save PNG.")
    args = parser.parse_args()
    fig = plot(pd.read_csv(args.input, dtype={"site": str}))
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    stem = FIGURES_ROOT / "strength_outage_histograms"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    if args.png:
        fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("Saved", stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
