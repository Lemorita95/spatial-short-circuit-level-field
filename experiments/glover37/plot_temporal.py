"""
Plot the temporal-spatial results for the Glover 37-bus experiment.

The script produces:
1. a system-wide heatmap of intact SCL variation relative to the independent
   temporal reference state; and
2. a two-panel local-example figure showing selected intact bus trajectories
   and the state-dependent effect of the fixed line 39--47 contingency.

Usage
-----
Generate both figures:

    python -m experiments.glover37.plot_temporal

Generate only the heatmap:

    python -m experiments.glover37.plot_temporal --figure heatmap

Generate only the local examples:

    python -m experiments.glover37.plot_temporal --figure local

Also save SVG:

    python -m experiments.glover37.plot_temporal --svg

Display interactively:

    python -m experiments.glover37.plot_temporal --show

Inputs
------
results/glover37/temporal/analysis/intact_trajectories.csv
results/glover37/temporal/analysis/commitment_state_by_hour.csv
results/glover37/temporal/analysis/contingency_effect.csv

Outputs
-------
figures/glover37/temporal_intact_heatmap.pdf
figures/glover37/temporal_local_examples.pdf

With --svg:
figures/glover37/temporal_intact_heatmap.svg
figures/glover37/temporal_local_examples.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TEMPORAL_ROOT = (
    ROOT
    / "results"
    / "glover37"
    / "temporal"
)

ANALYSIS_ROOT = (
    TEMPORAL_ROOT
    / "analysis"
)

FIGURES_ROOT = (
    ROOT
    / "figures"
    / "glover37"
)

INTACT_FILE = (
    ANALYSIS_ROOT
    / "intact_trajectories.csv"
)

COMMITMENT_FILE = (
    ANALYSIS_ROOT
    / "commitment_state_by_hour.csv"
)

CONTINGENCY_FILE = (
    ANALYSIS_ROOT
    / "contingency_effect.csv"
)

IEEE_TEXT_WIDTH_IN = 7.14
FONT_SIZE_PT = 8

LOCAL_BUSES = [5, 13, 34]
CONTINGENCY_BUS = 44


def parse_args() -> argparse.Namespace:
    """Parse figure-selection and output options."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot temporal-spatial SCL results for the "
            "37-bus conference-paper experiment."
        )
    )

    parser.add_argument(
        "--figure",
        choices=[
            "all",
            "heatmap",
            "local",
        ],
        default="all",
        help=(
            "Select which candidate figure to generate. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--svg",
        action="store_true",
        help=(
            "Also save SVG copies. "
            "PDF is always generated."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Display generated figures interactively."
        ),
    )

    return parser.parse_args()


def require_columns(
    data: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    """Raise a clear error if an input CSV lacks required columns."""
    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{source} is missing columns: "
            f"{sorted(missing)}"
        )


def parse_bool_series(
    series: pd.Series,
    source: Path,
) -> pd.Series:
    values = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
    )

    if values.isna().any():
        raise ValueError(
            "Could not parse commitment_changed "
            f"as boolean in {source}."
        )

    return values.astype(bool)


def load_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load retained temporal results needed by the paper figures."""
    for path in (
        INTACT_FILE,
        COMMITMENT_FILE,
        CONTINGENCY_FILE,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing temporal result: {path}"
            )

    intact = pd.read_csv(
        INTACT_FILE
    )

    commitment = pd.read_csv(
        COMMITMENT_FILE
    )

    contingency = pd.read_csv(
        CONTINGENCY_FILE
    )

    require_columns(
        intact,
        {
            "hour_index",
            "bus",
            "SCL_MVA",
            "SCL_reference_MVA",
            "SCL_deviation_from_reference_pct",
        },
        INTACT_FILE,
    )

    require_columns(
        commitment,
        {
            "hour_index",
            "commitment_changed",
        },
        COMMITMENT_FILE,
    )

    require_columns(
        contingency,
        {
            "hour_index",
            "bus",
            "SCL_intact_MVA",
            "SCL_contingency_MVA",
            "delta_SCL_pct",
        },
        CONTINGENCY_FILE,
    )

    for frame in (
        intact,
        commitment,
        contingency,
    ):
        frame["hour_index"] = (
            pd.to_numeric(
                frame["hour_index"],
                errors="raise",
            )
            .astype(int)
        )

    for frame in (
        intact,
        contingency,
    ):
        frame["bus"] = (
            pd.to_numeric(
                frame["bus"],
                errors="raise",
            )
            .astype(int)
        )

    for column in (
        "SCL_MVA",
        "SCL_reference_MVA",
        "SCL_deviation_from_reference_pct",
    ):
        intact[column] = (
            pd.to_numeric(
                intact[column],
                errors="raise",
            )
        )

    for column in (
        "SCL_intact_MVA",
        "SCL_contingency_MVA",
        "delta_SCL_pct",
    ):
        contingency[column] = (
            pd.to_numeric(
                contingency[column],
                errors="raise",
            )
        )

    commitment[
        "commitment_changed"
    ] = parse_bool_series(
        commitment[
            "commitment_changed"
        ],
        COMMITMENT_FILE,
    )

    intact = (
        intact.sort_values(
            ["bus", "hour_index"]
        )
        .reset_index(drop=True)
    )

    commitment = (
        commitment.sort_values(
            "hour_index"
        )
        .reset_index(drop=True)
    )

    contingency = (
        contingency.sort_values(
            ["bus", "hour_index"]
        )
        .reset_index(drop=True)
    )

    return (
        intact,
        commitment,
        contingency,
    )


def commitment_change_hours(
    commitment: pd.DataFrame,
) -> np.ndarray:
    """Return hours at which synchronous-generator commitment changes."""
    return (
        commitment.loc[
            commitment[
                "commitment_changed"
            ],
            "hour_index",
        ]
        .to_numpy(dtype=int)
    )


def add_commitment_markers(
    ax,
    hours: np.ndarray,
) -> None:
    """Add light vertical lines at commitment-transition hours."""
    for hour in hours:
        ax.axvline(
            hour,
            linestyle="--",
            linewidth=0.45,
            alpha=0.25,
        )


def hour_ticks(
    intact: pd.DataFrame,
    step: int = 24,
) -> list[int]:
    """Return compact regularly spaced hour ticks."""
    maximum = int(
        intact["hour_index"].max()
    )

    ticks = list(
        range(
            0,
            maximum + 1,
            step,
        )
    )

    if maximum not in ticks:
        ticks.append(
            maximum
        )

    return ticks


def save_figure(
    fig,
    stem: Path,
    *,
    save_svg: bool,
) -> list[Path]:
    """Save PDF and, optionally, SVG."""
    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    written = []

    pdf_path = stem.with_suffix(
        ".pdf"
    )

    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    written.append(
        pdf_path
    )

    if save_svg:
        svg_path = stem.with_suffix(
            ".svg"
        )

        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.02,
        )

        written.append(
            svg_path
        )

    return written


def plot_heatmap(
    intact: pd.DataFrame,
    commitment: pd.DataFrame,
    *,
    save_svg: bool,
    show: bool,
) -> list[Path]:
    """
    Plot system-wide intact SCL evolution relative to the reference state.

    Rows are buses in ascending numerical order.
    Columns are successive hourly operating states.
    Color is percent deviation from the independent temporal reference state.
    """
    wide = (
        intact.pivot(
            index="bus",
            columns="hour_index",
            values="SCL_deviation_from_reference_pct",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    values = wide.to_numpy(
        dtype=float
    )

    vmax = float(
        np.nanmax(
            np.abs(values)
        )
    )

    changes = (
        commitment_change_hours(
            commitment
        )
    )

    ticks = hour_ticks(
        intact
    )

    hour_columns = list(
        wide.columns
    )

    tick_positions = [
        hour_columns.index(tick)
        for tick in ticks
        if tick in hour_columns
    ]

    tick_labels = [
        tick
        for tick in ticks
        if tick in hour_columns
    ]

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
                IEEE_TEXT_WIDTH_IN,
                4.35,
            ),
            constrained_layout=True,
        )

        image = ax.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )

        for hour in changes:
            if hour in hour_columns:
                x = hour_columns.index(
                    hour
                )

                ax.axvline(
                    x - 0.5,
                    linestyle="--",
                    linewidth=0.45,
                    alpha=0.25,
                )

        ax.set_xlabel(
            "Hour"
        )

        ax.set_ylabel(
            "Bus"
        )

        ax.set_xticks(
            tick_positions
        )

        ax.set_xticklabels(
            tick_labels
        )

        ax.set_yticks(
            np.arange(
                len(wide.index)
            )
        )

        ax.set_yticklabels(
            wide.index.astype(str)
        )

        colorbar = fig.colorbar(
            image,
            ax=ax,
            pad=0.015,
            fraction=0.025,
        )

        colorbar.set_label(
            "SCL change from reference [%]"
        )

        stem = (
            FIGURES_ROOT
            / "temporal_intact_heatmap"
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


def plot_local_examples(
    intact: pd.DataFrame,
    commitment: pd.DataFrame,
    contingency: pd.DataFrame,
    *,
    save_svg: bool,
    show: bool,
) -> list[Path]:
    """
    Plot two compact local examples.

    Panel (a):
        buses 5, 13, and 34 under intact topology.

        This combines:
        - relative-order reversal between buses 13 and 34;
        - similar absolute SCL but different temporal sensitivity
          between buses 5 and 34.

    Panel (b):
        bus 44 under intact topology and the fixed line 39--47 outage.

        This illustrates operating-state x topology interaction:
        the contingency effect changes across the temporal sequence
        rather than behaving as a fixed offset.
    """
    changes = (
        commitment_change_hours(
            commitment
        )
    )

    ticks = hour_ticks(
        intact,
        step=48,
    )

    local = intact[
        intact["bus"].isin(
            LOCAL_BUSES
        )
    ].copy()

    bus44 = contingency[
        contingency["bus"]
        == CONTINGENCY_BUS
    ].copy()

    missing_local = (
        set(LOCAL_BUSES)
        - set(
            local["bus"].unique()
        )
    )

    if missing_local:
        raise ValueError(
            "Missing local-example buses in temporal results: "
            f"{sorted(missing_local)}"
        )

    if bus44.empty:
        raise ValueError(
            f"Bus {CONTINGENCY_BUS} is missing "
            "from contingency_effect.csv."
        )

    with plt.rc_context(
        {
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": FONT_SIZE_PT,
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(
                IEEE_TEXT_WIDTH_IN,
                3.0,
            ),
            sharex=True,
            constrained_layout=True,
        )

        left, right = axes

        for bus in LOCAL_BUSES:
            group = (
                local[
                    local["bus"] == bus
                ]
                .sort_values(
                    "hour_index"
                )
            )

            left.plot(
                group["hour_index"],
                group["SCL_MVA"],
                linewidth=1.1,
                label=f"Bus {bus}",
            )

        add_commitment_markers(
            left,
            changes,
        )

        left.set_xlabel(
            "Hour"
        )

        left.set_ylabel(
            "SCL [MVA]"
        )

        left.set_xticks(
            ticks
        )

        left.legend(
            frameon=False,
            loc="best",
        )

        left.text(
            0.01,
            0.98,
            "(a)",
            transform=left.transAxes,
            ha="left",
            va="top",
        )

        bus44 = bus44.sort_values(
            "hour_index"
        )

        right.plot(
            bus44[
                "hour_index"
            ],
            bus44[
                "SCL_intact_MVA"
            ],
            linewidth=1.1,
            label="Intact",
        )

        right.plot(
            bus44[
                "hour_index"
            ],
            bus44[
                "SCL_contingency_MVA"
            ],
            linewidth=1.1,
            label="Line 39--47 out",
        )

        add_commitment_markers(
            right,
            changes,
        )

        right.set_xlabel(
            "Hour"
        )

        right.set_ylabel(
            "SCL [MVA]"
        )

        right.set_xticks(
            ticks
        )

        right.legend(
            frameon=False,
            loc="best",
        )

        right.text(
            0.01,
            0.98,
            "(b)",
            transform=right.transAxes,
            ha="left",
            va="top",
        )

        stem = (
            FIGURES_ROOT
            / "temporal_local_examples"
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
    """Generate selected temporal paper-figure candidates."""
    args = parse_args()

    (
        intact,
        commitment,
        contingency,
    ) = load_inputs()

    written = []

    if args.figure in {
        "all",
        "heatmap",
    }:
        written.extend(
            plot_heatmap(
                intact,
                commitment,
                save_svg=args.svg,
                show=args.show,
            )
        )

    if args.figure in {
        "all",
        "local",
    }:
        written.extend(
            plot_local_examples(
                intact,
                commitment,
                contingency,
                save_svg=args.svg,
                show=args.show,
            )
        )

    for path in written:
        print(
            f"Wrote: {path}"
        )


if __name__ == "__main__":
    main()
