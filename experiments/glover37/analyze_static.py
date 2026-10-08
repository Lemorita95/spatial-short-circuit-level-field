"""
Analyze the Glover 37-bus static experiment relative to the baseline.

The script reads completed static runs and computes the bus-wise changes in
short-circuit current, short-circuit level, and pre-fault voltage relative
to the validated baseline case.

Usage
-----
Analyze all available scenarios:

    python -m experiments.glover37.analyze_static --all

Analyze one scenario:

    python -m experiments.glover37.analyze_static --scenario A2

Inputs
------
results/glover37/static/raw/base/results.csv
results/glover37/static/raw/base/metadata.json
results/glover37/static/raw/<scenario>/results.csv
results/glover37/static/raw/<scenario>/metadata.json

Outputs
-------
results/glover37/static/analysis/<scenario>/delta_from_base.csv

The analyzed scenarios are:
    A1, A2, A3, B1, B2, B3
"""


import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "results" / "glover37" / "static"
RAW_ROOT = STATIC_ROOT / "raw"
ANALYSIS_ROOT = STATIC_ROOT / "analysis"

S_BASE_MVA = 100.0


SCENARIO_ORDER = [
    "A1",
    "A2",
    "A3",
    "B1",
    "B2",
    "B3",
]


def result_file(
    name: str,
) -> Path:
    return (
        RAW_ROOT
        / name
        / "results.csv"
    )


def metadata_file(
    name: str,
) -> Path:
    return (
        RAW_ROOT
        / name
        / "metadata.json"
    )


def load_metadata(
    name: str,
) -> dict:
    path = metadata_file(
        name
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Metadata not found for '{name}': {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(
            f
        )


def load_results(
    name: str,
) -> pd.DataFrame:
    path = result_file(
        name
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Results not found for '{name}': {path}"
        )

    return pd.read_csv(
        path
    )


def is_valid_result(
    name: str,
) -> tuple[bool, str]:
    """
    Determine whether one result is complete enough
    for static sensitivity analysis.
    """

    if not metadata_file(
        name
    ).is_file():
        return (
            False,
            "metadata.json missing",
        )

    metadata = load_metadata(
        name
    )

    if not metadata.get(
        "power_flow_converged",
        False,
    ):
        return (
            False,
            "power flow did not converge",
        )

    if not metadata.get(
        "short_circuit_attempted",
        False,
    ):
        return (
            False,
            "short-circuit calculation was not attempted",
        )

    if not metadata.get(
        "short_circuit_converged",
        False,
    ):
        failed = metadata.get(
            "short_circuit_failed_buses",
            [],
        )

        return (
            False,
            "short-circuit calculation failed "
            f"for bus(es): {failed}",
        )

    if not result_file(
        name
    ).is_file():
        return (
            False,
            "results.csv missing",
        )

    return (
        True,
        "ok",
    )


def available_scenarios() -> list[str]:
    """
    Return valid available scenarios while preserving
    the predefined static-scenario order.
    """

    available = []

    for name in SCENARIO_ORDER:
        valid, _ = is_valid_result(
            name
        )

        if valid:
            available.append(
                name
            )

    return available


def validate_base() -> None:
    valid, reason = is_valid_result(
        "base"
    )

    if not valid:
        raise RuntimeError(
            "Baseline results are not valid for analysis: "
            f"{reason}"
        )


def validate_required_columns(
    name: str,
    data: pd.DataFrame,
) -> None:
    required = {
        "bus",
        "pre_fault_V_mag",
        "I_SCC_mag_pu",
        "SCL_pu",
        "SCL_MVA",
    }

    missing = (
        required
        - set(data.columns)
    )

    if missing:
        raise ValueError(
            f"{name} results are missing columns: "
            f"{sorted(missing)}. "
            "Re-run the case with the current run.py."
        )


def compare_to_base(
    scenario_name: str,
) -> pd.DataFrame:

    validate_base()

    valid, reason = is_valid_result(
        scenario_name
    )

    if not valid:
        raise RuntimeError(
            f"Scenario {scenario_name} is not valid "
            f"for analysis: {reason}"
        )

    base = load_results(
        "base"
    )

    scenario = load_results(
        scenario_name
    )

    validate_required_columns(
        "base",
        base,
    )

    validate_required_columns(
        scenario_name,
        scenario,
    )

    base = base.set_index(
        "bus"
    )

    scenario = scenario.set_index(
        "bus"
    )

    if set(base.index) != set(
        scenario.index
    ):
        raise ValueError(
            f"Bus sets differ between base "
            f"and {scenario_name}."
        )

    # Preserve baseline bus order.
    scenario = scenario.loc[
        base.index
    ]

    result = pd.DataFrame(
        index=base.index
    )

    # ---------------------------------------------------------
    # Short-circuit current
    # ---------------------------------------------------------

    result[
        "I_SCC_base_pu"
    ] = base[
        "I_SCC_mag_pu"
    ]

    result[
        "I_SCC_scenario_pu"
    ] = scenario[
        "I_SCC_mag_pu"
    ]

    result[
        "delta_I_SCC_pu"
    ] = (
        result[
            "I_SCC_scenario_pu"
        ]
        - result[
            "I_SCC_base_pu"
        ]
    )

    result[
        "delta_I_SCC_pct"
    ] = (
        100.0
        * result[
            "delta_I_SCC_pu"
        ]
        / result[
            "I_SCC_base_pu"
        ]
    )

    # ---------------------------------------------------------
    # Short-circuit level
    #
    # Current definition:
    #
    #   SCL_pu = I_SCC_pu
    #
    # because V_nominal_pu = 1.0.
    #
    # Keep the SCL columns explicit so the definition can be
    # changed later without changing the rest of the analysis.
    # ---------------------------------------------------------

    result[
        "SCL_base_pu"
    ] = base[
        "SCL_pu"
    ]

    result[
        "SCL_scenario_pu"
    ] = scenario[
        "SCL_pu"
    ]

    result[
        "delta_SCL_pu"
    ] = (
        result[
            "SCL_scenario_pu"
        ]
        - result[
            "SCL_base_pu"
        ]
    )

    result[
        "delta_SCL_pct"
    ] = (
        100.0
        * result[
            "delta_SCL_pu"
        ]
        / result[
            "SCL_base_pu"
        ]
    )

    result[
        "SCL_base_MVA"
    ] = base[
        "SCL_MVA"
    ]

    result[
        "SCL_scenario_MVA"
    ] = scenario[
        "SCL_MVA"
    ]

    result[
        "delta_SCL_MVA"
    ] = (
        result[
            "SCL_scenario_MVA"
        ]
        - result[
            "SCL_base_MVA"
        ]
    )

    # ---------------------------------------------------------
    # Pre-fault voltage diagnostics
    # ---------------------------------------------------------

    result[
        "V_base_pu"
    ] = base[
        "pre_fault_V_mag"
    ]

    result[
        "V_scenario_pu"
    ] = scenario[
        "pre_fault_V_mag"
    ]

    result[
        "delta_V_pu"
    ] = (
        result[
            "V_scenario_pu"
        ]
        - result[
            "V_base_pu"
        ]
    )

    result[
        "delta_V_pct"
    ] = (
        100.0
        * result[
            "delta_V_pu"
        ]
        / result[
            "V_base_pu"
        ]
    )

    result.index.name = (
        "bus"
    )

    return result.reset_index()


def summarize(
    scenario_name: str,
    comparison: pd.DataFrame,
) -> None:
    """
    The primary summary uses short-circuit level.

    I_SCC remains available in the output CSV as a
    supporting diagnostic.
    """

    delta = comparison[
        "delta_SCL_pct"
    ]

    absolute = delta.abs()

    max_abs_idx = (
        absolute.idxmax()
    )

    max_increase_idx = (
        delta.idxmax()
    )

    max_decrease_idx = (
        delta.idxmin()
    )

    print()
    print(
        f"Scenario {scenario_name}"
    )

    print(
        "-" * 52
    )

    print(
        f"Buses compared       : "
        f"{len(comparison)}"
    )

    # print(
    #     f"Mean SCL delta       : "
    #     f"{delta.mean():+.4f} %"
    # )

    # print(
    #     f"Mean |SCL delta|     : "
    #     f"{absolute.mean():.4f} %"
    # )

    print(
        f"Maximum increase     : "
        f"{delta.loc[max_increase_idx]:+.4f} % "
        f"(bus "
        f"{comparison.loc[max_increase_idx, 'bus']})"
    )

    print(
        f"Maximum decrease     : "
        f"{delta.loc[max_decrease_idx]:+.4f} % "
        f"(bus "
        f"{comparison.loc[max_decrease_idx, 'bus']})"
    )

    print(
        f"Maximum |SCL delta|  : "
        f"{absolute.loc[max_abs_idx]:.4f} % "
        f"(bus "
        f"{comparison.loc[max_abs_idx, 'bus']})"
    )


def analyze_one(
    scenario_name: str,
) -> None:

    comparison = compare_to_base(
        scenario_name
    )

    output = (
        ANALYSIS_ROOT
        / scenario_name
        / "delta_from_base.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output,
        index=False,
    )

    summarize(
        scenario_name,
        comparison,
    )

    print(
        f"Wrote: {output}"
    )


def print_scenario_status() -> None:
    print()
    print(
        "Scenario availability"
    )

    print(
        "-" * 52
    )

    for name in SCENARIO_ORDER:
        valid, reason = (
            is_valid_result(
                name
            )
        )

        if valid:
            print(
                f"{name}: available"
            )

        else:
            print(
                f"{name}: skipped ({reason})"
            )


def analyze_all() -> None:

    validate_base()

    print_scenario_status()

    scenarios = (
        available_scenarios()
    )

    if not scenarios:
        print()
        print(
            "No valid scenario results "
            "available for analysis."
        )

        return

    print()
    print(
        "Analyzing: "
        + ", ".join(
            scenarios
        )
    )

    for name in scenarios:
        analyze_one(
            name
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze available 37-bus "
            "conference-paper scenarios "
            "relative to the validated baseline."
        )
    )

    group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--scenario",
        choices=SCENARIO_ORDER,
        help=(
            "Analyze one scenario."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Analyze every currently available "
            "scenario with valid PF and SCC convergence."
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