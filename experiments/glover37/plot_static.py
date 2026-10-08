"""
Plot the bus-wise static short-circuit-level response for scenarios A1--B3.

The script reads the analyzed static results and produces a heatmap of the
percentage change in short-circuit level relative to the baseline, with buses
on the y-axis and scenarios on the x-axis.

Usage
-----
Generate the default figure:

    python -m experiments.glover37.plot_static

Use a symmetric color scale:

    python -m experiments.glover37.plot_static --scale symmetric

Render at IEEE single-column width:

    python -m experiments.glover37.plot_static --single-column

Also save SVG:

    python -m experiments.glover37.plot_static --svg

Display interactively:

    python -m experiments.glover37.plot_static --show

Inputs
------
results/glover37/static/analysis/A1/delta_from_base.csv
results/glover37/static/analysis/A2/delta_from_base.csv
results/glover37/static/analysis/A3/delta_from_base.csv
results/glover37/static/analysis/B1/delta_from_base.csv
results/glover37/static/analysis/B2/delta_from_base.csv
results/glover37/static/analysis/B3/delta_from_base.csv

Outputs
-------
figures/glover37/static_delta_scl_heatmap.pdf

With --svg:
figures/glover37/static_delta_scl_heatmap.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_ROOT = (
    ROOT
    / "results"
    / "glover37"
    / "static"
    / "analysis"
)

FIGURES_ROOT = (
    ROOT
    / "figures"
    / "glover37"
)

SCENARIOS = [
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
]

IEEE_COLUMN_WIDTH_IN = 3.49
IEEE_TEXT_WIDTH_IN = 7.14
DOUBLE_COLUMN_HEIGHT_IN = 4.45
SINGLE_COLUMN_HEIGHT_IN = 5.20
FONT_SIZE_PT = 8


def parse_args() -> argparse.Namespace:
    """Parse command-line options controlling layout, scale, and output."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot the network-wide static ΔSCL heatmap "
            "for scenarios A1--B3."
        )
    )

    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save an SVG copy. PDF is always written.",
    )

    parser.add_argument(
        "--scale",
        choices=("minmax", "symmetric"),
        default="minmax",
        help=(
            "Color scaling for the heatmap. "
            "'minmax' uses the data minimum and maximum; "
            "'symmetric' uses a zero-centered ±max(|ΔSCL|) scale."
        ),
    )

    parser.add_argument(
        "--single-column",
        action="store_true",
        help=(
            "Render at IEEE single-column width (3.49 in). "
            "Default is full two-column width (7.14 in)."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the matplotlib window.",
    )

    return parser.parse_args()


def load_scenario(scenario: str) -> pd.DataFrame:
    """Load and validate bus-wise ΔSCL results for one static scenario."""
    path = (
        ANALYSIS_ROOT
        / scenario
        / "delta_from_base.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing static-analysis result: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "bus",
        "delta_SCL_pct",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    out = df[
        [
            "bus",
            "delta_SCL_pct",
        ]
    ].copy()

    out["bus"] = pd.to_numeric(
        out["bus"],
        errors="raise",
    ).astype(int)

    out["delta_SCL_pct"] = pd.to_numeric(
        out["delta_SCL_pct"],
        errors="raise",
    )

    if out["bus"].duplicated().any():
        raise ValueError(
            f"{path} contains duplicate bus rows."
        )

    return out.sort_values("bus")


def build_matrix() -> tuple[list[int], np.ndarray]:
    """Assemble the bus-by-scenario ΔSCL matrix in ascending bus order."""
    loaded = {
        scenario: load_scenario(scenario)
        for scenario in SCENARIOS
    }

    bus_sets = {
        scenario: set(df["bus"])
        for scenario, df in loaded.items()
    }

    reference = bus_sets[SCENARIOS[0]]

    for scenario in SCENARIOS[1:]:
        if bus_sets[scenario] != reference:
            raise ValueError(
                "Static scenarios do not contain "
                "the same bus set."
            )

    buses = sorted(reference)

    matrix = np.column_stack(
        [
            (
                loaded[scenario]
                .set_index("bus")
                .loc[buses, "delta_SCL_pct"]
                .to_numpy(dtype=float)
            )
            for scenario in SCENARIOS
        ]
    )

    return buses, matrix


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

    written = []

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


def heatmap_cmap(scale_mode: str):
    """
    Colormap policy:
    - minmax: one-sided blue -> white, so the most negative ΔSCL is blue
      and the least negative / slightly positive end is white.
    - symmetric: diverging red-white-blue centered visually around zero.
    """
    if scale_mode == "minmax":
        return LinearSegmentedColormap.from_list(
            "blue_to_white",
            ["#2166ac", "#ffffff"],
            N=256,
        )

    if scale_mode == "symmetric":
        return "RdBu_r"

    raise ValueError(f"Unsupported scale mode: {scale_mode}")


def plot_static_heatmap(
    *,
    scale_mode: str,
    single_column: bool,
    save_svg: bool,
    show: bool,
) -> list[Path]:
    """Render and save the static ΔSCL heatmap using the selected layout."""
    buses, matrix = build_matrix()

    data_min = float(np.nanmin(matrix))
    data_max = float(np.nanmax(matrix))

    if not np.isfinite(data_min) or not np.isfinite(data_max):
        raise ValueError("Static ΔSCL matrix contains no finite data.")

    if scale_mode == "symmetric":
        limit = max(abs(data_min), abs(data_max))
        if limit == 0:
            limit = 1.0
        vmin = -limit
        vmax = limit
    elif scale_mode == "minmax":
        vmin = data_min
        vmax = data_max
        if vmin == vmax:
            # Degenerate case: keep a visible span.
            vmin -= 1.0
            vmax += 1.0
    else:
        raise ValueError(f"Unsupported scale mode: {scale_mode}")

    if single_column:
        figure_width = IEEE_COLUMN_WIDTH_IN
        figure_height = SINGLE_COLUMN_HEIGHT_IN
    else:
        figure_width = IEEE_TEXT_WIDTH_IN
        figure_height = DOUBLE_COLUMN_HEIGHT_IN

    with plt.rc_context(
        {
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
        }
    ):
        fig, ax = plt.subplots(
            figsize=(
                figure_width,
                figure_height,
            ),
            constrained_layout=True,
        )

        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=heatmap_cmap(scale_mode),
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("Scenario")
        ax.set_ylabel("Bus")

        ax.set_xticks(
            np.arange(
                len(SCENARIOS)
            )
        )
        ax.set_xticklabels(
            SCENARIOS
        )

        ax.set_yticks(
            np.arange(
                len(buses)
            )
        )
        ax.set_yticklabels(
            [
                str(bus)
                for bus in buses
            ]
        )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            pad=0.015 if single_column else 0.02,
            fraction=0.07 if single_column else 0.045,
        )
        colorbar.set_label(
            r"$\Delta$SCL relative to base [%]"
        )
        colorbar.ax.tick_params(
            labelsize=FONT_SIZE_PT
        )

        stem = (
            FIGURES_ROOT
            / "static_delta_scl_heatmap"
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
    """Run the static-figure export from the command line."""
    args = parse_args()

    written = plot_static_heatmap(
        scale_mode=args.scale,
        single_column=args.single_column,
        save_svg=args.svg,
        show=args.show,
    )

    for path in written:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
