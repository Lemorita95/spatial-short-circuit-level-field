import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import json
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.patches import (
    Circle,
    Rectangle,
)

from tools.axd_grid_viewer import (
    AxdModel,
    Generator,
    _segment_midpoint,
    draw_generator_symbol,
    parse_axd,
)


ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = ROOT / "cases" / "glover37.json"

RESULTS_ROOT = (
    ROOT
    / "results"
    / "glover37"
    / "static"
)

FIGURES_ROOT = (
    ROOT
    / "figures"
    / "glover37"
)

AXD_FILE = (
    ROOT
    / "data"
    / "powerworld"
    / "glover37"
    / "base"
    / "DesignCase2_2010.axd"
)


SCENARIO_ORDER = [
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
]


# ------------------------------------------------------------
# Plot settings
# ------------------------------------------------------------

# Set to None to use the absolute global maximum.
GLOBAL_SCALE_PERCENTILE = 97.5

# Display-only width multiplier for bus rectangles.
BUS_WIDTH_SCALE = 2.5

HIGHLIGHT_COLOR = "#2166ac"


# ------------------------------------------------------------
# Scenario display information
# ------------------------------------------------------------

SCENARIO_HIGHLIGHTS = {
    "A1": {
        "kind": "generator",
        "bus": 28,
        "gen_id": "1",
        "label": "G28-1: P → 0 MW",
    },

    "A2": {
        "kind": "generator",
        "bus": 28,
        "gen_id": "1",
        "label": "G28-1 disconnected",
    },

    "A3": {
        "kind": "generator",
        "bus": 14,
        "gen_id": "1",
        "label": "G14-1 disconnected",
    },

    "B1": {
        "kind": "line",
        "from_bus": 14,
        "to_bus": 34,
        "circuit": "1",
        "label": "Line 14–34 disconnected",
    },

    "B2": {
        "kind": "line",
        "from_bus": 21,
        "to_bus": 48,
        "circuit": "1",
        "label": (
            "Line 21–48 ckt 1 disconnected"
        ),
    },

    "B3": {
        "kind": "transformer",
        "from_bus": 28,
        "to_bus": 29,
        "circuit": "1",
        "label": (
            "Transformer 28–29 disconnected"
        ),
    },
}


def delta_file(
    scenario: str,
) -> Path:

    return (
        RESULTS_ROOT
        / scenario
        / "delta_from_base.csv"
    )


def available_scenarios(
) -> list[str]:

    return [
        scenario
        for scenario
        in SCENARIO_ORDER
        if delta_file(
            scenario
        ).is_file()
    ]


def load_delta_scl(
    scenario: str,
) -> dict[int, float]:

    path = delta_file(
        scenario
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Scenario results not found: "
            f"{path}"
        )

    data = pd.read_csv(
        path
    )

    required = {
        "bus",
        "delta_SCL_pct",
    }

    missing = (
        required
        - set(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing)}"
        )

    return {
        int(row.bus):
            float(
                row.delta_SCL_pct
            )
        for row
        in data.itertuples()
    }


def common_scale(
    scenarios: list[str],
    percentile: float | None = 97.5,
) -> float:
    """
    Determine one symmetric ΔSCL scale shared by all
    scenario figures.

    If percentile is None, use the absolute maximum.
    """

    values = []

    for scenario in scenarios:

        values.extend(
            abs(value)
            for value
            in load_delta_scl(
                scenario
            ).values()
        )

    if not values:
        raise ValueError(
            "No delta-SCL values found."
        )

    if percentile is None:

        limit = max(
            values
        )

    else:

        limit = float(
            pd.Series(
                values
            ).quantile(
                percentile
                / 100.0
            )
        )

    if limit == 0:
        limit = 1.0

    return limit


def active_generator_keys() -> set[tuple[int, str]]:
    """
    Return active generators from the actual case model.

    The AXD may contain display objects for generators that are
    currently out of service, so AXD presence alone is not enough.
    """

    case = json.loads(
        CASE_FILE.read_text(
            encoding="utf-8"
        )
    )

    return {
        (
            int(generator["from_bus"]),
            generator["name"].split("-", 1)[1].strip(),
        )
        for generator in case["elements"]["generators"]
        if int(generator.get("status", 1)) == 1
    }


def draw_network(
    ax,
    model: AxdModel,
) -> None:
    """
    Draw passive network and generator symbols as
    subdued SLD background.
    """

    # ---------------------------------------------------------
    # Transmission lines
    # ---------------------------------------------------------

    for edge in model.lines:

        xs = [
            point[0]
            for point
            in edge.coordinates
        ]

        ys = [
            point[1]
            for point
            in edge.coordinates
        ]

        ax.plot(
            xs,
            ys,
            color="0.78",
            linewidth=max(
                0.7,
                0.7
                * edge.thickness,
            ),
            zorder=1,
        )

    # ---------------------------------------------------------
    # Transformers
    # ---------------------------------------------------------

    for edge in model.transformers:

        xs = [
            point[0]
            for point
            in edge.coordinates
        ]

        ys = [
            point[1]
            for point
            in edge.coordinates
        ]

        ax.plot(
            xs,
            ys,
            color="0.70",
            linewidth=max(
                0.8,
                0.8
                * edge.thickness,
            ),
            zorder=2,
        )

        (
            mx,
            my,
            ux,
            uy,
        ) = (
            _segment_midpoint(
                edge.coordinates,
                edge.symbol_segment,
            )
        )

        radius = 0.65
        offset = 0.55

        for sign in (
            -1.0,
            1.0,
        ):

            cx = (
                mx
                + sign
                * offset
                * ux
            )

            cy = (
                my
                + sign
                * offset
                * uy
            )

            ax.add_patch(
                Circle(
                    (
                        cx,
                        cy,
                    ),
                    radius=radius,
                    facecolor="white",
                    edgecolor="0.70",
                    linewidth=1.0,
                    zorder=4,
                )
            )

    # ---------------------------------------------------------
    # Generators
    # ---------------------------------------------------------

    active_generators = active_generator_keys()

    for generator in model.generators:

        key = (
            generator.bus,
            generator.gen_id.strip(),
        )

        if key not in active_generators:
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
    bus,
    color,
    value: float,
    show_values: bool,
) -> None:
    """
    Draw a colored PowerWorld bus rectangle.

    AXD bus length is preserved exactly.
    Width is visually enlarged by BUS_WIDTH_SCALE.
    """

    orientation = (
        bus.orientation.lower()
    )

    length = (
        bus.size
    )

    width = (
        bus.width
        * BUS_WIDTH_SCALE
    )

    if orientation == "right":

        x0 = (
            bus.x
        )

        y0 = (
            bus.y
            - width / 2.0
        )

        rect_width = length
        rect_height = width

        label_x = (
            bus.x
            + length / 2.0
        )

        label_y = (
            bus.y
            + width / 2.0
            + 0.7
        )

        ha = "center"
        va = "bottom"

    elif orientation == "left":

        x0 = (
            bus.x
            - length
        )

        y0 = (
            bus.y
            - width / 2.0
        )

        rect_width = length
        rect_height = width

        label_x = (
            bus.x
            - length / 2.0
        )

        label_y = (
            bus.y
            + width / 2.0
            + 0.7
        )

        ha = "center"
        va = "bottom"

    elif orientation == "up":

        x0 = (
            bus.x
            - width / 2.0
        )

        y0 = (
            bus.y
        )

        rect_width = width
        rect_height = length

        label_x = (
            bus.x
            + width / 2.0
            + 0.7
        )

        label_y = (
            bus.y
            + length / 2.0
        )

        ha = "left"
        va = "center"

    elif orientation == "down":

        x0 = (
            bus.x
            - width / 2.0
        )

        y0 = (
            bus.y
            - length
        )

        rect_width = width
        rect_height = length

        label_x = (
            bus.x
            + width / 2.0
            + 0.7
        )

        label_y = (
            bus.y
            - length / 2.0
        )

        ha = "left"
        va = "center"

    else:
        raise ValueError(
            f"Unsupported bus orientation "
            f"{bus.orientation!r} "
            f"for bus {bus.number}"
        )

    ax.add_patch(
        Rectangle(
            (
                x0,
                y0,
            ),
            rect_width,
            rect_height,
            facecolor=color,
            edgecolor="0.45",
            linewidth=1.0,
            zorder=9,
        )
    )

    if show_values:

        label = (
            f"{bus.number}\n"
            f"{value:+.1f}%"
        )

    else:

        label = str(
            bus.number
        )

    ax.text(
        label_x,
        label_y,
        label,
        fontsize=7,
        ha=ha,
        va=va,
        color="black",
        zorder=30,
    )

def find_generator(
    model: AxdModel,
    bus: int,
    gen_id: str,
) -> Generator:
    """
    Find an exact DisplayGen object by bus and generator ID.
    """

    gen_id = (
        str(gen_id)
        .strip()
    )

    generator = next(
        (
            generator
            for generator
            in model.generators
            if generator.bus
            == bus
            and generator.gen_id.strip()
            == gen_id
        ),
        None,
    )

    if generator is None:
        raise ValueError(
            f"Generator {bus}:{gen_id} "
            f"not found in AXD."
        )

    return generator


def highlight_generator(
    ax,
    generator: Generator,
    label: str,
) -> None:
    """
    Highlight the actual DisplayGen symbol from the AXD.
    """

    # First draw a thicker blue symbol underneath.
    draw_generator_symbol(
        ax,
        generator,
        edgecolor=HIGHLIGHT_COLOR,
        facecolor="white",
        linewidth=3.2,
        zorder=10,
    )

    # Then redraw a slightly thinner inner symbol.
    draw_generator_symbol(
        ax,
        generator,
        edgecolor=HIGHLIGHT_COLOR,
        facecolor="white",
        linewidth=1.8,
        zorder=11,
    )

    ax.annotate(
        label,
        xy=(
            generator.x,
            generator.y,
        ),
        xytext=(
            8,
            8,
        ),
        textcoords=(
            "offset points"
        ),
        fontsize=8,
        color=HIGHLIGHT_COLOR,
        fontweight="bold",
        zorder=12,
    )


def highlight_edge(
    ax,
    edge,
    label: str,
) -> None:
    """
    Highlight an exact AXD line or transformer path.
    """

    xs = [
        point[0]
        for point
        in edge.coordinates
    ]

    ys = [
        point[1]
        for point
        in edge.coordinates
    ]

    ax.plot(
        xs,
        ys,
        color=HIGHLIGHT_COLOR,
        linewidth=2.8,
        linestyle="--",
        zorder=9,
    )

    (
        mx,
        my,
        _,
        _,
    ) = (
        _segment_midpoint(
            edge.coordinates,
            edge.symbol_segment,
        )
    )

    ax.annotate(
        label,
        xy=(
            mx,
            my,
        ),
        xytext=(
            8,
            8,
        ),
        textcoords=(
            "offset points"
        ),
        fontsize=8,
        color=HIGHLIGHT_COLOR,
        fontweight="bold",
        zorder=12,
    )


def highlight_scenario_change(
    ax,
    model: AxdModel,
    scenario: str,
) -> None:

    change = (
        SCENARIO_HIGHLIGHTS.get(
            scenario
        )
    )

    if change is None:
        return

    # ---------------------------------------------------------
    # Generator
    # ---------------------------------------------------------

    if change["kind"] == "generator":

        generator = (
            find_generator(
                model,
                bus=change["bus"],
                gen_id=change[
                    "gen_id"
                ],
            )
        )

        highlight_generator(
            ax,
            generator,
            change["label"],
        )

        return

    # ---------------------------------------------------------
    # Line
    # ---------------------------------------------------------

    if change["kind"] == "line":

        candidates = (
            model.lines
        )

    # ---------------------------------------------------------
    # Transformer
    # ---------------------------------------------------------

    elif (
        change["kind"]
        == "transformer"
    ):

        candidates = (
            model.transformers
        )

    else:
        raise ValueError(
            f"Unknown highlight kind: "
            f"{change['kind']}"
        )

    edge = next(
        (
            edge
            for edge
            in candidates

            if {
                edge.from_bus,
                edge.to_bus,
            }
            == {
                change[
                    "from_bus"
                ],
                change[
                    "to_bus"
                ],
            }

            and str(
                edge.circuit
            ).strip()
            == str(
                change[
                    "circuit"
                ]
            ).strip()
        ),
        None,
    )

    if edge is None:
        raise ValueError(
            "Scenario element not "
            f"found in AXD: {change}"
        )

    highlight_edge(
        ax,
        edge,
        change["label"],
    )


def render_scenario(
    model: AxdModel,
    scenario: str,
    limit: float,
    show: bool = False,
    show_values: bool = True,
) -> Path:

    values = load_delta_scl(
        scenario
    )

    FIGURES_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FIGURES_ROOT
        / (
            f"{scenario}"
            "_delta_scl_sld.svg"
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            14,
            9,
        )
    )

    draw_network(
        ax,
        model,
    )

    # ---------------------------------------------------------
    # ΔSCL colormap
    #
    # negative = red
    # zero     = white
    # positive = green
    # ---------------------------------------------------------

    cmap = (
        LinearSegmentedColormap
        .from_list(
            "red_white_green",
            [
                "#b2182b",
                "#ffffff",
                "#1a9850",
            ],
            N=256,
        )
    )

    norm = Normalize(
        vmin=-limit,
        vmax=limit,
        clip=True,
    )

    # ---------------------------------------------------------
    # Bus ΔSCL values
    # ---------------------------------------------------------

    for bus in (
        model.buses
    ):

        value = (
            values.get(
                bus.number
            )
        )

        if value is None:

            color = (
                "0.85"
            )

            value = 0.0

        else:

            color = cmap(
                norm(
                    value
                )
            )

        draw_bus(
            ax,
            bus,
            color,
            value,
            show_values,
        )

    # ---------------------------------------------------------
    # Experimental modification
    # ---------------------------------------------------------

    highlight_scenario_change(
        ax,
        model,
        scenario,
    )

    # ---------------------------------------------------------
    # Color bar
    # ---------------------------------------------------------

    scalar = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar.set_array(
        []
    )

    colorbar = fig.colorbar(
        scalar,
        ax=ax,
        fraction=0.035,
        pad=0.02,
        extend=(
            "both"
            if (
                GLOBAL_SCALE_PERCENTILE
                is not None
            )
            else "neither"
        ),
    )

    colorbar.set_label(
        "Short-circuit level change, ΔSCL (%)"
    )

    # ---------------------------------------------------------
    # Figure formatting
    # ---------------------------------------------------------

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    ax.autoscale(
        enable=True,
        axis="both",
        tight=False,
    )

    ax.margins(
        0.04
    )

    ax.set_axis_off()

    ax.set_title(
        f"{scenario} — "
        f"short-circuit level change"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        format="svg",
        bbox_inches="tight",
    )

    if show:
        plt.show()

    else:
        plt.close(
            fig
        )

    print(
        f"Wrote: {output}"
    )

    return output


def render_all(
    show: bool,
    show_values: bool,
) -> None:

    scenarios = (
        available_scenarios()
    )

    if not scenarios:
        raise RuntimeError(
            "No analyzed scenario "
            "results found."
        )

    model = parse_axd(
        AXD_FILE
    )

    limit = common_scale(
        scenarios,
        percentile=(
            GLOBAL_SCALE_PERCENTILE
        ),
    )

    if (
        GLOBAL_SCALE_PERCENTILE
        is None
    ):

        scale_description = (
            "absolute maximum"
        )

    else:

        scale_description = (
            f"{GLOBAL_SCALE_PERCENTILE}"
            "th percentile"
        )

    print(
        "Scenarios: "
        + ", ".join(
            scenarios
        )
    )

    print(
        f"Common ΔSCL scale "
        f"({scale_description}): "
        f"{-limit:.3f}% to "
        f"{limit:.3f}%"
    )

    for scenario in (
        scenarios
    ):

        render_scenario(
            model=model,
            scenario=scenario,
            limit=limit,
            show=show,
            show_values=show_values,
        )


def render_one(
    scenario: str,
    show: bool,
    show_values: bool,
    limit: float | None,
) -> None:

    model = parse_axd(
        AXD_FILE
    )

    if limit is None:

        scenarios = (
            available_scenarios()
        )

        limit = common_scale(
            scenarios,
            percentile=(
                GLOBAL_SCALE_PERCENTILE
            ),
        )

    render_scenario(
        model=model,
        scenario=scenario,
        limit=limit,
        show=show,
        show_values=show_values,
    )


def main(
) -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Plot bus-level ΔSCL on the "
            "PowerWorld Glover-37 one-line diagram."
        )
    )

    group = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--scenario",
        choices=SCENARIO_ORDER,
        help=(
            "Plot one analyzed scenario."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Plot every currently available "
            "scenario using one common scale."
        ),
    )

    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help=(
            "Override the symmetric color "
            "limit for a single scenario."
        ),
    )

    parser.add_argument(
        "--no-values",
        action="store_true",
        help=(
            "Show bus numbers without "
            "numerical ΔSCL labels."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Open matplotlib windows in "
            "addition to saving SVG figures."
        ),
    )

    args = (
        parser.parse_args()
    )

    show_values = (
        not args.no_values
    )

    if args.all:

        if (
            args.limit
            is not None
        ):
            raise ValueError(
                "--limit can only be "
                "used with --scenario."
            )

        render_all(
            show=args.show,
            show_values=show_values,
        )

    else:

        render_one(
            scenario=args.scenario,
            show=args.show,
            show_values=show_values,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()