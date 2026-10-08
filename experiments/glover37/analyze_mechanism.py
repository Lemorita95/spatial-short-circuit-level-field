"""
Analyze the raw 37-bus structural-mechanism experiment.

This stage performs no network or short-circuit simulation. It reads the raw
outputs produced by ``run_mechanism.py``, computes prediction errors, and
writes the per-bus and aggregate CSV files consumed by the paper table and
mechanism plot.

Examples
--------
python -m experiments.glover37.analyze_mechanism --all
python -m experiments.glover37.analyze_mechanism --scenario A2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results" / "glover37" / "mechanism"
RAW_ROOT = RESULTS_ROOT / "raw"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"

SCENARIO_ORDER = [
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
]


def raw_results_file(
    scenario: str,
) -> Path:
    """Return the raw per-bus result path for one scenario."""
    return (
        RAW_ROOT
        / scenario
        / "results.csv"
    )


def raw_metadata_file(
    scenario: str,
) -> Path:
    """Return the raw metadata path for one scenario."""
    return (
        RAW_ROOT
        / scenario
        / "metadata.json"
    )


def load_raw(
    scenario: str,
) -> tuple[pd.DataFrame, dict]:
    """Load and validate one raw mechanism scenario."""
    results_path = raw_results_file(
        scenario
    )

    metadata_path = raw_metadata_file(
        scenario
    )

    if not results_path.is_file():
        raise FileNotFoundError(
            f"Missing mechanism results: {results_path}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing mechanism metadata: {metadata_path}"
        )

    data = pd.read_csv(
        results_path
    )

    required = {
        "bus",
        "delta_scl_sim_pct",
        "delta_scl_pred_pct",
        "delta_Zii_real",
        "delta_Zii_imag",
        "abs_delta_Zii",
    }

    missing = required - set(
        data.columns
    )

    if missing:
        raise ValueError(
            f"{results_path} is missing columns: "
            f"{sorted(missing)}"
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    return data, metadata


def analyze_scenario(
    scenario: str,
) -> tuple[pd.DataFrame, dict]:
    """Compute bus-wise errors and summary metrics for one scenario."""
    data, metadata = load_raw(
        scenario
    )

    out = data.copy()

    out["error_pp"] = (
        out["delta_scl_pred_pct"]
        - out["delta_scl_sim_pct"]
    )

    # Preserve the existing analyzed-column order used downstream.
    out = out[
        [
            "bus",
            "delta_scl_sim_pct",
            "delta_scl_pred_pct",
            "error_pp",
            "delta_Zii_real",
            "delta_Zii_imag",
            "abs_delta_Zii",
        ]
    ]

    error = out[
        "error_pp"
    ].to_numpy(dtype=float)

    simulated = out[
        "delta_scl_sim_pct"
    ].to_numpy(dtype=float)

    predicted = out[
        "delta_scl_pred_pct"
    ].to_numpy(dtype=float)

    summary = {
        "scenario": scenario,
        "element": metadata["element"],
        "kind": metadata["kind"],
        "terminals": "-".join(
            str(value)
            for value in metadata["terminals"]
        ),
        "rank_Yet": int(
            metadata["rank_Yet"]
        ),
        "cond_middle": float(
            metadata["cond_middle"]
        ),
        "max_Z_algorithm_error": float(
            metadata["max_Z_algorithm_error"]
        ),
        "max_secondary_Y": float(
            metadata["max_secondary_Y"]
        ),
        "MAE_pp": float(
            np.mean(
                np.abs(error)
            )
        ),
        "RMSE_pp": float(
            np.sqrt(
                np.mean(
                    error**2
                )
            )
        ),
        "max_error_pp": float(
            np.max(
                np.abs(error)
            )
        ),
        "correlation": float(
            np.corrcoef(
                predicted,
                simulated,
            )[0, 1]
        ),
    }

    return out, summary


def save_scenario(
    scenario: str,
    data: pd.DataFrame,
) -> Path:
    """Write one analyzed per-bus coupling CSV."""
    path = (
        ANALYSIS_ROOT
        / f"{scenario}_general_coupling.csv"
    )

    data.to_csv(
        path,
        index=False,
    )

    return path


def print_summary(
    summary: dict,
) -> None:
    """Print compact mechanism-comparison metrics."""
    print()
    print(
        f"{summary['scenario']} mechanism agreement"
    )
    print("-" * 48)
    print(
        f"Rank              : "
        f"{summary['rank_Yet']}"
    )
    print(
        f"MAE               : "
        f"{summary['MAE_pp']:.8f} pp"
    )
    print(
        f"RMSE              : "
        f"{summary['RMSE_pp']:.8f} pp"
    )
    print(
        f"Maximum error     : "
        f"{summary['max_error_pp']:.8f} pp"
    )
    print(
        f"Correlation       : "
        f"{summary['correlation']:.10f}"
    )
    print(
        f"Max Z update error: "
        f"{summary['max_Z_algorithm_error']:.3e}"
    )


def analyze_one(
    scenario: str,
) -> dict:
    """Analyze and save one mechanism scenario."""
    data, summary = analyze_scenario(
        scenario
    )

    path = save_scenario(
        scenario,
        data,
    )

    print_summary(
        summary
    )

    print(
        f"Wrote: {path}"
    )

    return summary


def save_aggregate_summary(
    summaries: list[dict],
) -> Path:
    """Write the aggregate mechanism summary used by the paper table."""
    path = (
        ANALYSIS_ROOT
        / "general_outage_coupling_summary.csv"
    )

    pd.DataFrame(
        summaries
    ).to_csv(
        path,
        index=False,
    )

    return path


def analyze_all() -> None:
    """Analyze all structural scenarios and write the aggregate summary."""
    summaries = [
        analyze_one(
            scenario
        )
        for scenario in SCENARIO_ORDER
    ]

    path = save_aggregate_summary(
        summaries
    )

    print()
    print(
        f"Wrote: {path}"
    )


def main() -> None:

    ANALYSIS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw 37-bus structural-mechanism results."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--scenario",
        choices=SCENARIO_ORDER,
        help=(
            "Analyze one mechanism scenario. "
            "Use --all to regenerate the aggregate summary."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Analyze all scenarios and regenerate the aggregate summary."
        ),
    )

    args = parser.parse_args()

    if args.all:
        analyze_all()
    else:
        analyze_one(
            args.scenario
        )


if __name__ == "__main__":
    main()
