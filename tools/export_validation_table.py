"""
    Export the compact PowerWorld-reference-agreement table used by the paper.

    Usage:
    python -m tools.export_validation_table
    python -m tools.export_validation_table --bottom
"""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SUMMARY = (
    ROOT
    / "results"
    / "glover37"
    / "validation"
    / "summary.csv"
)

OUTPUT = (
    ROOT
    / "IEEE-conference-template-062824"
    / "validation_table.tex"
)


def fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def build_rows(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    states = df[df["comparison"].eq("state")]

    changes = df[
        df["comparison"].eq("change_field")
        & df["case"].isin(
            [
                "A1 vs base",
                "A2 vs base",
                "A3 vs base",
                "B1 vs base",
                "B2 vs base",
                "B3 vs base",
            ]
        )
    ]

    a2_a1 = df[
        df["comparison"].eq("change_field")
        & df["case"].eq("A2 vs A1")
    ].iloc[0]

    return [
        (
            "Fault-current magnitude",
            fmt(states["I_mean_abs_error_pct"].max(), 3) + r"\%",
            fmt(states["I_max_abs_error_pct"].max(), 3) + r"\%",
        ),
        (
            r"$\Delta$SCL from base",
            fmt(changes["delta_SCL_MAE_pct_points"].max(), 4) + " pp",
            fmt(changes["delta_SCL_max_abs_error_pct_points"].max(), 4) + " pp",
        ),
        (
            r"A2--A1 $\Delta$SCL",
            fmt(a2_a1["delta_SCL_MAE_pct_points"], 6) + " pp",
            fmt(a2_a1["delta_SCL_max_abs_error_pct_points"], 6) + " pp",
        ),
    ]


def render_table(
    rows: list[tuple[str, str, str]],
    placement: str,
) -> str:
    body = "\n".join(
        f"{name} & {mean_err} & {max_err} \\\\"
        for name, mean_err, max_err in rows
    )

    return rf"""\begin{{table}}[{placement}]
\caption{{Agreement with the PowerWorld reference calculation.}}
\label{{tab:validation}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{lcc}}
\hline
Comparison & Worst mean $|e|$ & Max bus $|e|$ \\
\hline
{body}
\hline
\end{{tabular}}
\vspace{{1mm}}

\begin{{minipage}}{{\columnwidth}}
\scriptsize
Errors are evaluated over all 37 buses.
The first row reports the worst case across the base state and A1--B3;
the second reports the worst case across A1--B3 relative to the base state.
\end{{minipage}}
\end{{table}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the compact validation table for IEEEtran."
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--top",
        action="store_true",
        help="Place the table at the top of a column (default).",
    )

    group.add_argument(
        "--bottom",
        action="store_true",
        help="Place the table at the bottom of a column.",
    )

    args = parser.parse_args()

    placement = "b" if args.bottom else "t"

    df = pd.read_csv(SUMMARY)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        render_table(
            build_rows(df),
            placement,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {OUTPUT}")
    print(f"Placement: [{placement}]")


if __name__ == "__main__":
    main()
