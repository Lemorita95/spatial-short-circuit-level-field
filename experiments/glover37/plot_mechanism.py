"""
Plot a mechanism-oriented coupling figure for the structural scenarios.

The figure relates the magnitude of the driving-point impedance change,
``|ΔZ_ii|``, to the magnitude of the simulated short-circuit-level change,
``|ΔSCL_i|``, for scenarios A2, A3, B1, B2, and B3.

Each scenario is shown with a distinct color on one common scatter plot.

Usage
-----
python experiments/glover37/plot_mechanism.py
python experiments/glover37/plot_mechanism.py --single-column
python experiments/glover37/plot_mechanism.py --linear-x
python experiments/glover37/plot_mechanism.py --svg
python experiments/glover37/plot_mechanism.py --show

Options
-------
--single-column
    Render at IEEE single-column width (3.49 in). Default is 7.14 in.
--linear-x
    Use a linear x-axis. The default is logarithmic because |ΔZ_ii|
    spans multiple orders of magnitude.
--svg
    Also save an SVG copy. PDF is always generated.
--show
    Display the figure interactively.

Output
------
figures/glover37/mechanism_coupling_vs_delta_scl.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ROOT = (
    ROOT
    / "results"
    / "glover37"
    / "mechanism"
    / "analysis"
)

FIGURES_ROOT = (
    ROOT
    / "figures"
    / "glover37"
)

SCENARIOS = ["A2", "A3", "B1", "B2", "B3"]

IEEE_COLUMN_WIDTH_IN = 3.49
IEEE_TEXT_WIDTH_IN = 7.14
FONT_SIZE_PT = 8


def parse_args() -> argparse.Namespace:
    """Parse figure-layout and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot |ΔZ_ii| versus |ΔSCL| for structural "
            "mechanism scenarios A2--B3."
        )
    )

    parser.add_argument(
        "--single-column",
        action="store_true",
        help="Render at IEEE single-column width (3.49 in).",
    )

    parser.add_argument(
        "--linear-x",
        action="store_true",
        help=(
            "Use a linear x-axis. The default is logarithmic "
            "because |ΔZ_ii| spans multiple orders of magnitude."
        ),
    )

    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save an SVG copy. PDF is always generated.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )

    return parser.parse_args()


def load_scenario(scenario: str) -> pd.DataFrame:
    """Load and validate one analyzed mechanism scenario."""
    path = ANALYSIS_ROOT / f"{scenario}_general_coupling.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing mechanism result: {path}"
        )

    data = pd.read_csv(path)

    required = {
        "bus",
        "delta_scl_sim_pct",
        "abs_delta_Zii",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    out = data[
        [
            "bus",
            "delta_scl_sim_pct",
            "abs_delta_Zii",
        ]
    ].copy()

    out["bus"] = pd.to_numeric(
        out["bus"],
        errors="raise",
    ).astype(int)

    out["delta_scl_sim_pct"] = pd.to_numeric(
        out["delta_scl_sim_pct"],
        errors="raise",
    )

    out["abs_delta_Zii"] = pd.to_numeric(
        out["abs_delta_Zii"],
        errors="raise",
    )

    out["abs_delta_scl_pct"] = (
        out["delta_scl_sim_pct"].abs()
    )

    out = out[np.isfinite(out["abs_delta_Zii"])]
    out = out[np.isfinite(out["abs_delta_scl_pct"])]

    return out


def save_figure(
    fig,
    stem: Path,
    *,
    save_svg: bool,
) -> list[Path]:
    """Save the publication PDF and, optionally, an SVG copy."""
    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    written: list[Path] = []

    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    written.append(pdf_path)

    if save_svg:
        svg_path = stem.with_suffix(".svg")
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        written.append(svg_path)

    return written


def figure_size(single_column: bool) -> tuple[float, float]:
    """Return a paper-oriented figure size."""
    if single_column:
        return IEEE_COLUMN_WIDTH_IN, 3.15

    return IEEE_TEXT_WIDTH_IN, 3.55


def compute_limits(
    data: dict[str, pd.DataFrame],
    *,
    linear_x: bool,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits across all scenarios."""
    x = np.concatenate(
        [
            frame["abs_delta_Zii"].to_numpy(dtype=float)
            for frame in data.values()
        ]
    )

    y = np.concatenate(
        [
            frame["abs_delta_scl_pct"].to_numpy(dtype=float)
            for frame in data.values()
        ]
    )

    if linear_x:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        xpad = 0.04 * (xmax - xmin) if xmax > xmin else 1.0
        xlim = (max(0.0, xmin - xpad), xmax + xpad)
    else:
        positive_x = x[x > 0]
        if len(positive_x) == 0:
            raise ValueError(
                "No positive |ΔZ_ii| values are available for log-x plotting."
            )

        xmin = float(np.nanmin(positive_x))
        xmax = float(np.nanmax(positive_x))
        xlim = (0.85 * xmin, 1.15 * xmax)

    ymin = 0.0
    ymax = float(np.nanmax(y))
    ypad = 0.05 * ymax if ymax > 0 else 1.0
    ylim = (ymin, ymax + ypad)

    return xlim, ylim


def plot_mechanism_coupling(
    *,
    single_column: bool,
    linear_x: bool,
    save_svg: bool,
    show: bool,
) -> list[Path]:
    """
    Render one multi-scenario coupling plot:
    |ΔZ_ii| on x, |ΔSCL_i| on y.
    """
    data = {
        scenario: load_scenario(scenario)
        for scenario in SCENARIOS
    }

    xlim, ylim = compute_limits(
        data,
        linear_x=linear_x,
    )

    figsize = figure_size(single_column)

    with plt.rc_context(
        {
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
        }
    ):
        fig, ax = plt.subplots(
            figsize=figsize,
            constrained_layout=True,
        )

        for scenario, frame in data.items():
            ax.scatter(
                frame["abs_delta_Zii"],
                frame["abs_delta_scl_pct"],
                s=18 if single_column else 22,
                alpha=0.82,
                label=scenario,
            )

        if not linear_x:
            ax.set_xscale("log")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xlabel(r"$|\Delta Z_{ii}|$ [p.u.]")
        ax.set_ylabel(r"$|\Delta \mathrm{SCL}_i|$ [%]")

        ax.legend(
            frameon=False,
            ncol=len(SCENARIOS),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            borderaxespad=0.0,
            handletextpad=0.35,
            columnspacing=0.9,
            markerscale=0.9,
        )
        stem = (
            FIGURES_ROOT
            / "mechanism_coupling_vs_delta_scl"
        )

        written = save_figure(
            fig,
            stem,
            save_svg=save_svg,
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

    return written


def main() -> None:
    """Generate the mechanism coupling figure."""
    args = parse_args()

    written = plot_mechanism_coupling(
        single_column=args.single_column,
        linear_x=args.linear_x,
        save_svg=args.svg,
        show=args.show,
    )

    for path in written:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
