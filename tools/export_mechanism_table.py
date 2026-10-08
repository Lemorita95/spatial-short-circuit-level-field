"""Export the compact mechanism-verification table used by the paper.

Usage:
  python -m tools.export_mechanism_table
  python -m tools.export_mechanism_table --bottom
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "glover37" / "mechanism" / "analysis" / "general_outage_coupling_summary.csv"
OUTPUT = ROOT / "IEEE-conference-template-062824" / "mechanism_table.tex"
SCENARIO_ORDER = ["A2", "A3", "B1", "B2", "B3"]


def parse_args() -> argparse.Namespace:
    """Parse top/bottom float placement; top is the default."""
    p = argparse.ArgumentParser(description="Export the mechanism-verification table.")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--top", action="store_true", help="Place table at top (default).")
    g.add_argument("--bottom", action="store_true", help="Place table at bottom.")
    return p.parse_args()


def load_summary() -> pd.DataFrame:
    """Load and validate the mechanism summary."""
    df = pd.read_csv(SUMMARY)
    required = {"scenario", "rank_Yet", "MAE_pp", "max_error_pp", "correlation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{SUMMARY} is missing columns: {sorted(missing)}")

    df = df[df["scenario"].isin(SCENARIO_ORDER)].copy()
    df["scenario"] = pd.Categorical(df["scenario"], SCENARIO_ORDER, ordered=True)
    return df.sort_values("scenario")


def render_table(df: pd.DataFrame, placement: str) -> str:
    """Render a compact single-column IEEE-style LaTeX table."""
    rows = "\n".join(
        f"{r.scenario} & {int(r.rank_Yet)} & {r.MAE_pp:.4f} & "
        f"{r.max_error_pp:.4f} & {r.correlation:.6f} \\\\"
        for r in df.itertuples()
    )

    return rf"""\begin{{table}}[{placement}]
\caption{{Agreement between the low-rank structural prediction and simulated $\Delta$SCL.}}
\label{{tab:mechanism}}
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\renewcommand{{\arraystretch}}{{1.05}}
\begin{{tabular}}{{ccccc}}
\hline
Case & Rank & MAE [pp] & Max. [pp] & Corr. \\
\hline
{rows}
\hline
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    """Export the table to the conference-paper directory."""
    args = parse_args()
    placement = "b" if args.bottom else "t"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render_table(load_summary(), placement), encoding="utf-8")
    print(f"Wrote: {OUTPUT}")
    print(f"Placement: [{placement}]")


if __name__ == "__main__":
    main()
