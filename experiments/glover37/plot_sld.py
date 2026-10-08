"""
Plot publication-oriented one-line diagrams for the Glover 37-bus study.

The script uses the PowerWorld display geometry to generate the benchmark
overview and, optionally, spatial maps of the analyzed static ΔSCL results.

Usage
-----
Generate the benchmark overview:

    python -m experiments.glover37.plot_sld --overview

Generate one static scenario map:

    python -m experiments.glover37.plot_sld --scenario A2

Generate all available static scenario maps:

    python -m experiments.glover37.plot_sld --all

Also save SVG:

    python -m experiments.glover37.plot_sld --overview --svg

Inputs
------
cases/glover37.json
data/powerworld/glover37/base/DesignCase2_2010.axd

For scenario-result maps:
results/glover37/static/analysis/<scenario>/delta_from_base.csv

Outputs
-------
Overview:
figures/glover37/benchmark_overview_tags.pdf

Scenario maps:
figures/glover37/<scenario>_delta_scl_sld.pdf

With --svg, corresponding SVG files are also written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Rectangle

from tools.axd_grid_viewer import (
    AxdModel,
    Generator,
    _segment_midpoint,
    draw_generator_symbol,
    parse_axd,
)


# =====================================================================
# Paths
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

CASE_FILE = ROOT / "cases" / "glover37.json"

AXD_FILE = (
    ROOT
    / "data"
    / "powerworld"
    / "glover37"
    / "base"
    / "DesignCase2_2010.axd"
)

STATIC_ANALYSIS_ROOT = ROOT / "results" / "glover37" / "static" / "analysis"
FIGURES_ROOT = ROOT / "figures" / "glover37"


# =====================================================================
# Publication settings
# =====================================================================

# Measured from the supplied IEEE conference template.
IEEE_COLUMN_WIDTH_IN = 3.49
IEEE_TEXT_WIDTH_IN = 7.14

# IEEE template guidance for text embedded in figures.
FIGURE_FONT_SIZE_PT = 8.0

# Display-only widening of bus rectangles. This does not modify AXD data.
BUS_WIDTH_SCALE = 2.5

# One restrained highlight style for all study modifications.
HIGHLIGHT_COLOR = "#2166ac"
HIGHLIGHT_LINESTYLE = "--"

# Scenario-result plots use one common symmetric scale.
GLOBAL_SCALE_PERCENTILE = 97.5


# =====================================================================
# Study definitions
# =====================================================================

SCENARIO_ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]

SCENARIOS = {
    "A1": {
        "kind": "generator",
        "bus": 28,
        "gen_id": "1",
        "description": "G28-1: P -> 0 MW",
    },
    "A2": {
        "kind": "generator",
        "bus": 28,
        "gen_id": "1",
        "description": "G28-1 disconnected",
    },
    "A3": {
        "kind": "generator",
        "bus": 14,
        "gen_id": "1",
        "description": "G14-1 disconnected",
    },
    "B1": {
        "kind": "line",
        "from_bus": 14,
        "to_bus": 34,
        "circuit": "1",
        "description": "Line 14-34 disconnected",
    },
    "B2": {
        "kind": "line",
        "from_bus": 21,
        "to_bus": 48,
        "circuit": "1",
        "description": "Line 21-48 ckt 1 disconnected",
    },
    "B3": {
        "kind": "transformer",
        "from_bus": 28,
        "to_bus": 29,
        "circuit": "1",
        "description": "Transformer 28-29 disconnected",
    },
}

# Unique elements shown in the benchmark overview.
# A1 and A2 intentionally share one physical marker.
OVERVIEW_ITEMS = [
    {
        "tag": "A1/A2",
        "kind": "generator",
        "bus": 28,
        "gen_id": "1",
    },
    {
        "tag": "A3",
        "kind": "generator",
        "bus": 14,
        "gen_id": "1",
    },
    {
        "tag": "B1",
        "kind": "line",
        "from_bus": 14,
        "to_bus": 34,
        "circuit": "1",
    },
    {
        "tag": "B2",
        "kind": "line",
        "from_bus": 21,
        "to_bus": 48,
        "circuit": "1",
    },
    {
        "tag": "B3",
        "kind": "transformer",
        "from_bus": 28,
        "to_bus": 29,
        "circuit": "1",
    },
    {
        "tag": "C",
        "kind": "line",
        "from_bus": 39,
        "to_bus": 47,
        "circuit": "1",
    },
]


# =====================================================================
# Data loading
# =====================================================================

def delta_file(scenario: str) -> Path:
    return STATIC_ANALYSIS_ROOT / scenario / "delta_from_base.csv"


def available_scenarios() -> list[str]:
    return [
        scenario
        for scenario in SCENARIO_ORDER
        if delta_file(scenario).is_file()
    ]


def load_delta_scl(scenario: str) -> dict[int, float]:
    path = delta_file(scenario)

    if not path.is_file():
        raise FileNotFoundError(
            f"Scenario results not found: {path}"
        )

    data = pd.read_csv(path)

    required = {"bus", "delta_SCL_pct"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return {
        int(row.bus): float(row.delta_SCL_pct)
        for row in data.itertuples()
    }


def common_scale(
    scenarios: list[str],
    percentile: float | None = GLOBAL_SCALE_PERCENTILE,
) -> float:
    values: list[float] = []

    for scenario in scenarios:
        values.extend(
            abs(value)
            for value in load_delta_scl(scenario).values()
        )

    if not values:
        raise ValueError("No ΔSCL values found.")

    if percentile is None:
        limit = max(values)
    else:
        limit = float(
            pd.Series(values).quantile(
                percentile / 100.0
            )
        )

    return limit if limit > 0 else 1.0


def active_generator_keys() -> set[tuple[int, str]]:
    """
    Return generators that are active in the actual JSON case.

    AXD files may still contain display objects for equipment that is
    out of service, so AXD presence alone is not an activity test.
    """
    case = json.loads(
        CASE_FILE.read_text(encoding="utf-8")
    )

    return {
        (
            int(generator["from_bus"]),
            generator["name"].split("-", 1)[1].strip(),
        )
        for generator in case["elements"]["generators"]
        if int(generator.get("status", 1)) == 1
    }


# =====================================================================
# AXD lookup helpers
# =====================================================================

def find_generator(
    model: AxdModel,
    *,
    bus: int,
    gen_id: str,
) -> Generator:
    gen_id = str(gen_id).strip()

    generator = next(
        (
            item
            for item in model.generators
            if item.bus == bus
            and item.gen_id.strip() == gen_id
        ),
        None,
    )

    if generator is None:
        raise ValueError(
            f"Generator {bus}:{gen_id} not found in AXD."
        )

    return generator


def find_edge(
    model: AxdModel,
    *,
    kind: str,
    from_bus: int,
    to_bus: int,
    circuit: str,
):
    if kind == "line":
        candidates = model.lines
    elif kind == "transformer":
        candidates = model.transformers
    else:
        raise ValueError(
            f"Unsupported edge kind: {kind}"
        )

    edge = next(
        (
            item
            for item in candidates
            if {
                item.from_bus,
                item.to_bus,
            }
            == {
                from_bus,
                to_bus,
            }
            and str(item.circuit).strip()
            == str(circuit).strip()
        ),
        None,
    )

    if edge is None:
        raise ValueError(
            f"{kind} {from_bus}-{to_bus} "
            f"ckt {circuit} not found in AXD."
        )

    return edge


def bus_label_position(
    model: AxdModel,
    bus_number: int,
) -> tuple[float, float] | None:
    """
    Use the original AXD DisplayBusField position.

    This requires the updated axd_grid_viewer.py in which AxdModel
    exposes bus_field_position(). The Name-field anchor is reused for
    the bus number shown in the paper figure.
    """
    getter = getattr(
        model,
        "bus_field_position",
        None,
    )

    if getter is None:
        raise RuntimeError(
            "AxdModel does not expose bus_field_position(). "
            "Use the updated tools/axd_grid_viewer.py that parses "
            "DisplayBusField."
        )

    return getter(bus_number, "Name")


# =====================================================================
# Drawing primitives
# =====================================================================

def draw_network_background(
    ax,
    model: AxdModel,
) -> None:
    """
    Draw passive network geometry and active generators in subdued gray.
    """
    # Lines
    for edge in model.lines:
        xs = [p[0] for p in edge.coordinates]
        ys = [p[1] for p in edge.coordinates]

        ax.plot(
            xs,
            ys,
            color="0.78",
            linewidth=max(
                0.7,
                0.7 * edge.thickness,
            ),
            zorder=1,
        )

    # Transformers
    for edge in model.transformers:
        xs = [p[0] for p in edge.coordinates]
        ys = [p[1] for p in edge.coordinates]

        ax.plot(
            xs,
            ys,
            color="0.70",
            linewidth=max(
                0.8,
                0.8 * edge.thickness,
            ),
            zorder=2,
        )

        mx, my, ux, uy = _segment_midpoint(
            edge.coordinates,
            edge.symbol_segment,
        )

        radius = 0.65
        offset = 0.55

        for sign in (-1.0, 1.0):
            cx = mx + sign * offset * ux
            cy = my + sign * offset * uy

            ax.add_patch(
                Circle(
                    (cx, cy),
                    radius=radius,
                    facecolor="white",
                    edgecolor="0.70",
                    linewidth=1.0,
                    zorder=4,
                )
            )

    # Only generators that are active in the actual case.
    active = active_generator_keys()

    for generator in model.generators:
        key = (
            generator.bus,
            generator.gen_id.strip(),
        )

        if key not in active:
            continue

        draw_generator_symbol(
            ax,
            generator,
            edgecolor="0.55",
            facecolor="white",
            linewidth=1.0,
            zorder=4,
        )


def draw_bus(
    ax,
    model: AxdModel,
    bus,
    *,
    facecolor: str,
    value: float | None = None,
    show_value: bool = False,
    font_size: float = FIGURE_FONT_SIZE_PT,
) -> None:
    """
    Draw one bus rectangle and its paper label.

    Bus geometry comes from DisplayBus.
    Label placement comes from DisplayBusField(Name).
    """
    orientation = bus.orientation.lower()

    length = bus.size
    width = bus.width * BUS_WIDTH_SCALE

    if orientation == "right":
        x0 = bus.x
        y0 = bus.y - width / 2.0
        rect_width = length
        rect_height = width

    elif orientation == "left":
        x0 = bus.x - length
        y0 = bus.y - width / 2.0
        rect_width = length
        rect_height = width

    elif orientation == "up":
        x0 = bus.x - width / 2.0
        y0 = bus.y
        rect_width = width
        rect_height = length

    elif orientation == "down":
        x0 = bus.x - width / 2.0
        y0 = bus.y - length
        rect_width = width
        rect_height = length

    else:
        raise ValueError(
            f"Unsupported bus orientation "
            f"{bus.orientation!r} for bus {bus.number}."
        )

    ax.add_patch(
        Rectangle(
            (x0, y0),
            rect_width,
            rect_height,
            facecolor=facecolor,
            edgecolor="0.35",
            linewidth=0.8,
            zorder=9,
        )
    )

    label_xy = bus_label_position(
        model,
        bus.number,
    )

    if label_xy is None:
        # Defensive fallback only. With the current AXD all displayed
        # buses are expected to have a Name field.
        label_xy = (bus.x, bus.y)

    if show_value and value is not None:
        label = f"{bus.number}\n{value:+.1f}%"
    else:
        label = str(bus.number)

    ax.text(
        label_xy[0],
        label_xy[1],
        label,
        fontsize=font_size,
        ha="left",
        va="bottom",
        color="black",
        clip_on=False,
        zorder=30,
    )


def draw_highlighted_generator(
    ax,
    generator: Generator,
    *,
    tag: str | None,
) -> None:
    draw_generator_symbol(
        ax,
        generator,
        edgecolor=HIGHLIGHT_COLOR,
        facecolor="white",
        linewidth=2.2,
        zorder=15,
    )

    if tag is not None:
        ax.annotate(
            tag,
            xy=(generator.x, generator.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=FIGURE_FONT_SIZE_PT,
            fontweight="bold",
            color=HIGHLIGHT_COLOR,
            ha="left",
            va="bottom",
            annotation_clip=False,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": HIGHLIGHT_COLOR,
                "linewidth": 0.6,
            },
            zorder=40,
        )


def draw_highlighted_edge(
    ax,
    edge,
    *,
    tag: str | None,
) -> None:
    xs = [p[0] for p in edge.coordinates]
    ys = [p[1] for p in edge.coordinates]

    ax.plot(
        xs,
        ys,
        color=HIGHLIGHT_COLOR,
        linewidth=2.0,
        linestyle=HIGHLIGHT_LINESTYLE,
        zorder=15,
    )

    if tag is None:
        return

    mx, my, _, _ = _segment_midpoint(
        edge.coordinates,
        edge.symbol_segment,
    )

    ax.annotate(
        tag,
        xy=(mx, my),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=FIGURE_FONT_SIZE_PT,
        fontweight="bold",
        color=HIGHLIGHT_COLOR,
        ha="left",
        va="bottom",
        annotation_clip=False,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": HIGHLIGHT_COLOR,
            "linewidth": 0.6,
        },
        zorder=40,
    )


# =====================================================================
# Figure sizing / cropping
# =====================================================================

def fit_axes_to_drawn_network(
    fig,
    ax,
    *,
    target_width_in: float,
    data_margin_fraction: float = 0.008,
) -> None:
    """
    Fit the axes tightly around the actual network artists.

    Important:
    - matplotlib line/patch artists determine the primary-element bounds;
    - text/annotations are deliberately NOT used to enlarge the data
      limits, but bbox_inches='tight' includes them on export;
    - figure height is derived from the resulting data aspect ratio;
    - therefore the network fills the requested width without clipping
      the primary elements or leaving avoidable left/right whitespace.
    """
    ax.relim()
    ax.autoscale_view(tight=True)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)

    xpad = data_margin_fraction * xspan
    ypad = data_margin_fraction * yspan

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    xspan = xmax - xmin
    yspan = ymax - ymin

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.0)
    ax.set_axis_off()

    target_height_in = (
        target_width_in
        * yspan
        / xspan
    )

    fig.set_size_inches(
        target_width_in,
        target_height_in,
        forward=True,
    )

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
    )


def save_figure(
    fig,
    stem: Path,
    *,
    save_svg: bool = False,
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
        pad_inches=0.01,
    )
    written.append(pdf_path)

    if save_svg:
        svg_path = stem.with_suffix(".svg")
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            pad_inches=0.01,
        )
        written.append(svg_path)

    return written


# =====================================================================
# Paper overview
# =====================================================================

def render_overview(
    *,
    width_in: float = IEEE_COLUMN_WIDTH_IN,
    save_svg: bool = False,
    show: bool = False,
) -> list[Path]:
    """
    Render the benchmark topology with all study elements highlighted.

    This is the candidate Fig. 1:
      A1/A2  generator at bus 28
      A3     generator at bus 14
      B1     line 14-34
      B2     line 21-48
      B3     transformer 28-29
      C      temporal contingency line 39-47
    """
    model = parse_axd(AXD_FILE)

    fig, ax = plt.subplots()

    draw_network_background(
        ax,
        model,
    )

    for bus in model.buses:
        draw_bus(
            ax,
            model,
            bus,
            facecolor="white",
            font_size=FIGURE_FONT_SIZE_PT,
        )

    for item in OVERVIEW_ITEMS:
        if item["kind"] == "generator":
            generator = find_generator(
                model,
                bus=item["bus"],
                gen_id=item["gen_id"],
            )

            draw_highlighted_generator(
                ax,
                generator,
                tag=item["tag"],
            )

        else:
            edge = find_edge(
                model,
                kind=item["kind"],
                from_bus=item["from_bus"],
                to_bus=item["to_bus"],
                circuit=item["circuit"],
            )

            draw_highlighted_edge(
                ax,
                edge,
                tag=item["tag"],
            )

    fit_axes_to_drawn_network(
        fig,
        ax,
        target_width_in=width_in,
    )

    stem = (
        FIGURES_ROOT
        / "benchmark_overview_tags"
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

    for path in written:
        print(f"Wrote: {path}")

    return written


# =====================================================================
# Static ΔSCL network maps
# =====================================================================

def delta_colormap():
    return LinearSegmentedColormap.from_list(
        "red_white_green",
        [
            "#b2182b",
            "#ffffff",
            "#1a9850",
        ],
        N=256,
    )


def highlight_scenario_element(
    ax,
    model: AxdModel,
    scenario: str,
) -> None:
    item = SCENARIOS[scenario]

    if item["kind"] == "generator":
        generator = find_generator(
            model,
            bus=item["bus"],
            gen_id=item["gen_id"],
        )

        draw_highlighted_generator(
            ax,
            generator,
            tag=scenario,
        )

        return

    edge = find_edge(
        model,
        kind=item["kind"],
        from_bus=item["from_bus"],
        to_bus=item["to_bus"],
        circuit=item["circuit"],
    )

    draw_highlighted_edge(
        ax,
        edge,
        tag=scenario,
    )


def render_scenario(
    scenario: str,
    *,
    limit: float | None = None,
    width_in: float = IEEE_TEXT_WIDTH_IN,
    show_values: bool = True,
    save_svg: bool = False,
    show: bool = False,
) -> list[Path]:
    model = parse_axd(AXD_FILE)
    values = load_delta_scl(scenario)

    if limit is None:
        scenarios = available_scenarios()

        limit = common_scale(
            scenarios,
            percentile=GLOBAL_SCALE_PERCENTILE,
        )

    cmap = delta_colormap()

    norm = Normalize(
        vmin=-limit,
        vmax=limit,
        clip=True,
    )

    fig, ax = plt.subplots()

    draw_network_background(
        ax,
        model,
    )

    for bus in model.buses:
        value = values.get(bus.number)

        if value is None:
            facecolor = "0.85"
            value = 0.0
        else:
            facecolor = cmap(norm(value))

        draw_bus(
            ax,
            model,
            bus,
            facecolor=facecolor,
            value=value,
            show_value=show_values,
            font_size=FIGURE_FONT_SIZE_PT,
        )

    highlight_scenario_element(
        ax,
        model,
        scenario,
    )

    scalar = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    scalar.set_array([])

    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        extend=(
            "both"
            if GLOBAL_SCALE_PERCENTILE is not None
            else "neither"
        ),
    )

    colorbar.set_label(
        "Short-circuit level change, ΔSCL (%)",
        fontsize=FIGURE_FONT_SIZE_PT,
    )

    colorbar.ax.tick_params(
        labelsize=FIGURE_FONT_SIZE_PT,
    )

    fit_axes_to_drawn_network(
        fig,
        ax,
        target_width_in=width_in,
    )

    stem = (
        FIGURES_ROOT
        / f"{scenario}_delta_scl_sld"
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

    for path in written:
        print(f"Wrote: {path}")

    return written


def render_all(
    *,
    width_in: float,
    show_values: bool,
    save_svg: bool,
    show: bool,
) -> None:
    scenarios = available_scenarios()

    if not scenarios:
        raise RuntimeError(
            "No analyzed static scenarios found."
        )

    limit = common_scale(
        scenarios,
        percentile=GLOBAL_SCALE_PERCENTILE,
    )

    print(
        "Scenarios: "
        + ", ".join(scenarios)
    )

    print(
        f"Common ΔSCL scale: "
        f"{-limit:.3f}% to {limit:.3f}%"
    )

    for scenario in scenarios:
        render_scenario(
            scenario,
            limit=limit,
            width_in=width_in,
            show_values=show_values,
            save_svg=save_svg,
            show=show,
        )


# =====================================================================
# CLI
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-oriented Glover-37 "
            "one-line-diagram figures."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--overview",
        action="store_true",
        help=(
            "Render the benchmark topology with "
            "A1/A2, A3, B1, B2, B3 and C tags."
        ),
    )

    group.add_argument(
        "--scenario",
        choices=SCENARIO_ORDER,
        help="Render one static ΔSCL scenario.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Render all available static ΔSCL "
            "scenarios using one common scale."
        ),
    )

    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help=(
            "Output width in inches. Defaults to "
            "3.49 in for --overview and 7.14 in "
            "for scenario-result figures."
        ),
    )

    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help=(
            "Symmetric ΔSCL color limit for "
            "--scenario only."
        ),
    )

    parser.add_argument(
        "--no-values",
        action="store_true",
        help=(
            "For scenario maps, omit numerical "
            "ΔSCL values from bus labels."
        ),
    )

    parser.add_argument(
        "--svg",
        action="store_true",
        help=(
            "Also save an SVG copy. PDF is always written "
            "and is the publication-format output."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the matplotlib window.",
    )

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.overview:
        if args.limit is not None:
            raise ValueError(
                "--limit is not applicable to --overview."
            )

        render_overview(
            width_in=(
                args.width
                if args.width is not None
                else IEEE_COLUMN_WIDTH_IN
            ),
            save_svg=args.svg,
            show=args.show,
        )

        return

    if args.all:
        if args.limit is not None:
            raise ValueError(
                "--limit can only be used with --scenario."
            )

        render_all(
            width_in=(
                args.width
                if args.width is not None
                else IEEE_TEXT_WIDTH_IN
            ),
            show_values=not args.no_values,
            save_svg=args.svg,
            show=args.show,
        )

        return

    render_scenario(
        args.scenario,
        limit=args.limit,
        width_in=(
            args.width
            if args.width is not None
            else IEEE_TEXT_WIDTH_IN
        ),
        show_values=not args.no_values,
        save_svg=args.svg,
        show=args.show,
    )


if __name__ == "__main__":
    main()
