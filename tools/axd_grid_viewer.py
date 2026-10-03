#!/usr/bin/env python3
"""
Parse a PowerWorld-style .axd display file and render its network layout.

Supported display objects:
    * DisplayBus
    * DisplayGen
    * DisplayTransmissionLine
    * DisplayTransformer

For lines and transformers, <SUBDATA Line> coordinates are preserved so
routed bends from the original one-line diagram remain visible.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


POINT_RE = re.compile(
    r"\[\s*([-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?)\s*\]"
)

GENERATOR_BODY_SIZE = 3.96

def pw_float(
    text: str,
) -> float:
    """Convert PowerWorld numeric text, including decimal commas."""
    return float(
        text.strip().replace(",", ".")
    )


def _tokenize(
    line: str,
) -> list[str]:
    """Split one AXD record while preserving quoted fields."""
    return shlex.split(
        line,
        comments=False,
        posix=True,
    )


@dataclass
class Bus:
    number: int
    auxiliary_id: str
    x: float
    y: float
    thickness: float
    size: float
    width: float
    orientation: str
    style: str


@dataclass
class Generator:
    bus: int
    gen_id: str
    auxiliary_id: str
    x: float
    y: float
    thickness: float
    size: float
    width: float
    orientation: str
    style: str


@dataclass
class Edge:
    kind: str
    from_bus: int
    to_bus: int
    circuit: str
    auxiliary_id: str
    x: float
    y: float
    thickness: float
    coordinates: list[tuple[float, float]]
    symbol_segment: Optional[int] = None


@dataclass
class AxdModel:
    buses: list[Bus]
    generators: list[Generator]
    lines: list[Edge]
    transformers: list[Edge]

    @property
    def edges(
        self,
    ) -> list[Edge]:
        return (
            self.lines
            + self.transformers
        )


def _find_block(
    lines: list[str],
    name: str,
) -> tuple[int, int]:
    """
    Return inclusive content bounds inside a named top-level AXD block.
    """

    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.lstrip().startswith(
                name + " ("
            )
        ),
        None,
    )

    if start is None:
        raise ValueError(
            f"Block {name!r} not found"
        )

    i = start

    while (
        i < len(lines)
        and "{"
        not in lines[i]
    ):
        i += 1

    if i >= len(lines):
        raise ValueError(
            f"Opening '{{' for block "
            f"{name!r} not found"
        )

    content_start = (
        i + 1
    )

    depth = 1
    i = content_start

    while i < len(lines):

        depth += (
            lines[i].count("{")
        )

        depth -= (
            lines[i].count("}")
        )

        if depth == 0:
            return (
                content_start,
                i - 1,
            )

        i += 1

    raise ValueError(
        f"Closing '}}' for block "
        f"{name!r} not found"
    )


def _parse_bus_record(
    line: str,
) -> Bus:

    t = _tokenize(
        line
    )

    if len(t) < 16:
        raise ValueError(
            "Unexpected DisplayBus record: "
            f"{line.rstrip()}"
        )

    return Bus(
        number=int(t[0]),
        auxiliary_id=t[1].strip(),
        x=pw_float(t[2]),
        y=pw_float(t[3]),
        thickness=pw_float(t[4]),
        size=pw_float(t[8]),
        width=pw_float(t[9]),
        orientation=t[10].strip(),
        style=t[15].strip(),
    )


def _parse_generator_record(
    line: str,
) -> Generator:

    t = _tokenize(
        line
    )

    if len(t) < 20:
        raise ValueError(
            "Unexpected DisplayGen record: "
            f"{line.rstrip()}"
        )

    return Generator(
        bus=int(t[0]),
        gen_id=t[1].strip(),
        auxiliary_id=t[2].strip(),
        x=pw_float(t[3]),
        y=pw_float(t[4]),
        thickness=pw_float(t[5]),
        size=pw_float(t[11]),
        width=pw_float(t[12]),
        orientation=t[14].strip(),
        style=t[19].strip(),
    )


def _parse_edge_record(
    line: str,
    kind: str,
) -> Edge:

    t = _tokenize(
        line
    )

    if len(t) < 8:
        raise ValueError(
            "Unexpected edge record: "
            f"{line.rstrip()}"
        )

    symbol_segment = None

    if kind == "transformer":
        try:
            symbol_segment = int(
                t[-1]
            )
        except (
            ValueError,
            TypeError,
        ):
            symbol_segment = None

    return Edge(
        kind=kind,
        from_bus=int(t[0]),
        to_bus=int(t[1]),
        circuit=t[2].strip(),
        auxiliary_id=t[3].strip(),
        x=pw_float(t[4]),
        y=pw_float(t[5]),
        thickness=1.0,
        coordinates=[],
        symbol_segment=symbol_segment,
    )


def _parse_display_bus(
    lines: list[str],
) -> list[Bus]:

    a, b = _find_block(
        lines,
        "DisplayBus",
    )

    buses = []

    for raw in lines[
        a : b + 1
    ]:

        s = raw.strip()

        if (
            not s
            or s.startswith("//")
            or s.startswith("<")
        ):
            continue

        buses.append(
            _parse_bus_record(
                raw
            )
        )

    return buses


def _parse_display_gen(
    lines: list[str],
) -> list[Generator]:

    a, b = _find_block(
        lines,
        "DisplayGen",
    )

    generators = []

    for raw in lines[
        a : b + 1
    ]:

        s = raw.strip()

        if (
            not s
            or s.startswith("//")
            or s.startswith("<")
        ):
            continue

        generators.append(
            _parse_generator_record(
                raw
            )
        )

    return generators


def _parse_edge_block(
    lines: list[str],
    block_name: str,
    kind: str,
) -> list[Edge]:

    a, b = _find_block(
        lines,
        block_name,
    )

    edges = []

    i = a

    while i <= b:

        s = lines[i].strip()

        if (
            not s
            or s.startswith("//")
        ):
            i += 1
            continue

        if (
            s.startswith("<SUBDATA")
            or s.startswith("</SUBDATA")
        ):
            i += 1
            continue

        edge = _parse_edge_record(
            lines[i],
            kind,
        )

        i += 1

        while (
            i <= b
            and not lines[i].strip()
        ):
            i += 1

        if (
            i <= b
            and lines[i]
            .strip()
            .lower()
            .startswith(
                "<subdata line"
            )
        ):

            i += 1

            coords = []

            while (
                i <= b
                and not lines[i]
                .strip()
                .lower()
                .startswith(
                    "</subdata"
                )
            ):

                for mx, my in (
                    POINT_RE.findall(
                        lines[i]
                    )
                ):
                    coords.append(
                        (
                            pw_float(mx),
                            pw_float(my),
                        )
                    )

                i += 1

            edge.coordinates = (
                coords
            )

            if i <= b:
                i += 1

        edges.append(
            edge
        )

    return edges


def parse_axd(
    path: str | Path,
) -> AxdModel:

    path = Path(
        path
    )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = (
        text.splitlines()
    )

    buses = (
        _parse_display_bus(
            lines
        )
    )

    generators = (
        _parse_display_gen(
            lines
        )
    )

    lines_out = (
        _parse_edge_block(
            lines,
            "DisplayTransmissionLine",
            "line",
        )
    )

    transformers = (
        _parse_edge_block(
            lines,
            "DisplayTransformer",
            "transformer",
        )
    )

    bus_xy = {
        bus.number: (
            bus.x,
            bus.y,
        )
        for bus in buses
    }

    for edge in (
        lines_out
        + transformers
    ):

        if not edge.coordinates:

            p0 = bus_xy.get(
                edge.from_bus,
                (
                    edge.x,
                    edge.y,
                ),
            )

            p1 = bus_xy.get(
                edge.to_bus,
                p0,
            )

            edge.coordinates = [
                p0,
                p1,
            ]

    return AxdModel(
        buses=buses,
        generators=generators,
        lines=lines_out,
        transformers=transformers,
    )


def _segment_midpoint(
    points: list[
        tuple[float, float]
    ],
    one_based_segment: Optional[int],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Return midpoint and unit direction of a selected polyline segment.
    """

    if len(points) < 2:

        x, y = (
            points[0]
            if points
            else (
                0.0,
                0.0,
            )
        )

        return (
            x,
            y,
            1.0,
            0.0,
        )

    if (
        one_based_segment
        is not None
        and 1
        <= one_based_segment
        < len(points)
    ):

        i = (
            one_based_segment
            - 1
        )

        (
            x0,
            y0,
        ), (
            x1,
            y1,
        ) = (
            points[i],
            points[i + 1],
        )

    else:

        lengths = [
            math.hypot(
                x1 - x0,
                y1 - y0,
            )
            for (
                x0,
                y0,
            ), (
                x1,
                y1,
            )
            in zip(
                points,
                points[1:],
            )
        ]

        total = sum(
            lengths
        )

        target = (
            total / 2.0
        )

        acc = 0.0
        i = 0

        for i, seg_len in enumerate(
            lengths
        ):

            if (
                acc
                + seg_len
                >= target
            ):
                break

            acc += seg_len

        (
            x0,
            y0,
        ), (
            x1,
            y1,
        ) = (
            points[i],
            points[i + 1],
        )

    dx = (
        x1 - x0
    )

    dy = (
        y1 - y0
    )

    norm = (
        math.hypot(
            dx,
            dy,
        )
        or 1.0
    )

    return (
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        dx / norm,
        dy / norm,
    )


def draw_generator_symbol(
    ax,
    generator: Generator,
    edgecolor="0.35",
    facecolor="white",
    linewidth: float = 1.2,
    zorder: int = 5,
) -> None:
    """
    Draw a PowerWorld DisplayGen using its AXD anchor,
    orientation, and total size.

    All generator bodies use the same diameter as G28
    for visual consistency.
    """

    from matplotlib.patches import Circle

    orientation = generator.orientation.lower()

    x = generator.x
    y = generator.y

    total_length = generator.size
    body_size = GENERATOR_BODY_SIZE
    radius = body_size / 2.0

    # Place the body at the far end of the AXD generator extent.
    center_offset = (
        total_length
        - radius
    )

    stem_length = max(
        total_length
        - body_size,
        0.0,
    )

    if orientation == "right":

        cx = x + center_offset
        cy = y

        ax.plot(
            [
                x,
                x + stem_length,
            ],
            [
                y,
                y,
            ],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    elif orientation == "left":

        cx = x - center_offset
        cy = y

        ax.plot(
            [
                x,
                x - stem_length,
            ],
            [
                y,
                y,
            ],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    elif orientation == "up":

        cx = x
        cy = y + center_offset

        ax.plot(
            [
                x,
                x,
            ],
            [
                y,
                y + stem_length,
            ],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    elif orientation == "down":

        cx = x
        cy = y - center_offset

        ax.plot(
            [
                x,
                x,
            ],
            [
                y,
                y - stem_length,
            ],
            color=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )

    else:

        raise ValueError(
            f"Unsupported generator orientation "
            f"{generator.orientation!r} for "
            f"generator {generator.bus}:{generator.gen_id}"
        )

    ax.add_patch(
        Circle(
            (
                cx,
                cy,
            ),
            radius=radius,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=zorder,
        )
    )

def render_model(
    model: AxdModel,
    output: Optional[
        str | Path
    ] = None,
    show: bool = True,
    labels: bool = True,
    circuit_labels: bool = False,
    title: Optional[str] = None,
    dpi: int = 160,
) -> None:

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required "
            "for rendering"
        ) from exc

    fig, ax = plt.subplots(
        figsize=(
            14,
            9,
        )
    )

    for edge in model.lines:

        xs = [
            p[0]
            for p in edge.coordinates
        ]

        ys = [
            p[1]
            for p in edge.coordinates
        ]

        ax.plot(
            xs,
            ys,
            color="#59636e",
            linewidth=max(
                0.8,
                0.8
                * edge.thickness,
            ),
            zorder=1,
        )

        if circuit_labels:

            mid = (
                len(xs) // 2
            )

            ax.text(
                xs[mid],
                ys[mid],
                (
                    f"{edge.from_bus}-"
                    f"{edge.to_bus}/"
                    f"{edge.circuit}"
                ),
                fontsize=6,
                zorder=5,
            )

    for edge in model.transformers:

        xs = [
            p[0]
            for p in edge.coordinates
        ]

        ys = [
            p[1]
            for p in edge.coordinates
        ]

        ax.plot(
            xs,
            ys,
            color="#a05a1c",
            linewidth=max(
                1.0,
                0.9
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
                    edgecolor="#a05a1c",
                    linewidth=1.3,
                    zorder=4,
                )
            )

    for generator in (
        model.generators
    ):
        draw_generator_symbol(
            ax,
            generator,
        )

    for bus in model.buses:

        orientation = (
            bus.orientation.lower()
        )

        lw = max(
            1.8,
            1.5
            * bus.thickness,
        )

        if orientation == "up":

            ax.plot(
                [
                    bus.x,
                    bus.x,
                ],
                [
                    bus.y,
                    bus.y
                    + bus.size,
                ],
                color="#20262d",
                linewidth=lw,
                zorder=6,
            )

        elif orientation == "down":

            ax.plot(
                [
                    bus.x,
                    bus.x,
                ],
                [
                    bus.y,
                    bus.y
                    - bus.size,
                ],
                color="#20262d",
                linewidth=lw,
                zorder=6,
            )

        elif orientation == "right":

            ax.plot(
                [
                    bus.x,
                    bus.x
                    + bus.size,
                ],
                [
                    bus.y,
                    bus.y,
                ],
                color="#20262d",
                linewidth=lw,
                zorder=6,
            )

        else:

            ax.plot(
                [
                    bus.x,
                    bus.x
                    - bus.size,
                ],
                [
                    bus.y,
                    bus.y,
                ],
                color="#20262d",
                linewidth=lw,
                zorder=6,
            )

        if labels:
            ax.text(
                bus.x,
                bus.y,
                str(
                    bus.number
                ),
                fontsize=8,
                zorder=7,
            )

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

    ax.set_title(
        title
        or (
            f"AXD network: "
            f"{len(model.buses)} buses, "
            f"{len(model.generators)} generators, "
            f"{len(model.lines)} lines, "
            f"{len(model.transformers)} transformers"
        )
    )

    ax.grid(
        False
    )

    fig.tight_layout()

    if output:

        fig.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    else:
        plt.close(
            fig
        )


def export_json(
    model: AxdModel,
    path: str | Path,
) -> None:

    payload = {
        "buses": [
            asdict(bus)
            for bus
            in model.buses
        ],
        "generators": [
            asdict(generator)
            for generator
            in model.generators
        ],
        "lines": [
            asdict(edge)
            for edge
            in model.lines
        ],
        "transformers": [
            asdict(edge)
            for edge
            in model.transformers
        ],
    }

    Path(
        path
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def print_summary(
    model: AxdModel,
) -> None:

    print(
        f"Buses:        "
        f"{len(model.buses)}"
    )

    print(
        f"Generators:   "
        f"{len(model.generators)}"
    )

    print(
        f"Lines:        "
        f"{len(model.lines)}"
    )

    print(
        f"Transformers: "
        f"{len(model.transformers)}"
    )

    routed = sum(
        1
        for edge
        in model.edges
        if len(
            edge.coordinates
        ) >= 2
    )

    print(
        f"Routed edges: "
        f"{routed}/"
        f"{len(model.edges)}"
    )


def build_arg_parser(
) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Render an AXD one-line network "
            "using its stored Display* coordinates."
        )
    )

    parser.add_argument(
        "axd",
        help="Input .axd text file",
    )

    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Save figure. If omitted, only "
            "the interactive window is shown."
        ),
    )

    parser.add_argument(
        "--json",
        dest="json_output",
        help=(
            "Export parsed display geometry "
            "as JSON."
        ),
    )

    parser.add_argument(
        "--no-labels",
        action="store_true",
    )

    parser.add_argument(
        "--circuit-labels",
        action="store_true",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
    )

    parser.add_argument(
        "--title",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
    )

    return parser


def main(
) -> None:

    args = (
        build_arg_parser()
        .parse_args()
    )

    model = parse_axd(
        args.axd
    )

    print_summary(
        model
    )

    if args.json_output:
        export_json(
            model,
            args.json_output,
        )

    if (
        args.output
        or not args.no_show
    ):

        render_model(
            model,
            output=args.output,
            show=not args.no_show,
            labels=not args.no_labels,
            circuit_labels=args.circuit_labels,
            title=args.title,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()