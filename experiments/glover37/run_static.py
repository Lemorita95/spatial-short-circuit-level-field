import argparse
import cmath
import csv
import json
from pathlib import Path

import numpy as np

from models.CustomNetwork import CustomNetwork
from models.PowerFlow import PowerFlow
from models.ShortCircuit import ShortCircuit as SC

from .scenarios import (
    Action,
    CASE_FILE,
    SCENARIOS,
    Change,
    Scenario,
)


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "results" / "glover37" / "static" / "raw"

S_BASE_MVA = 100.0


CSV_HEADER = [
    "bus",
    "pre_fault_V_mag",
    "pre_fault_V_ang_deg",
    "P_net",
    "Q_net",
    "P_gen",
    "Q_gen",
    "P_conv",
    "Q_conv",
    "P_load",
    "Q_load",
    "P_shunt",
    "Q_shunt",
    "I_SCC_mag_pu",
    "I_SCC_ang_deg",
    "SCL_pu",
    "SCL_MVA",
]


class PowerFlowConvergenceError(RuntimeError):
    pass


def apply_change(
    network: CustomNetwork,
    change: Change,
) -> None:
    """
    Apply one predefined experimental change.
    """

    if change.action == Action.SET_GENERATOR_P:
        key = change.element.lower()

        if key not in network.generators:
            raise KeyError(
                f"Generator not found: {change.element}"
            )

        if change.value is None:
            raise ValueError(
                f"{change.action.value} requires a numerical value."
            )

        network.generators[key].P = float(change.value)
        return

    if change.action == Action.DISCONNECT:
        network.change_element_status(
            change.element,
            False,
        )
        return

    raise ValueError(
        f"Unsupported scenario action: {change.action}"
    )


def apply_scenario(
    network: CustomNetwork,
    scenario: Scenario | None,
) -> None:
    if scenario is None:
        return

    for change in scenario.changes:
        apply_change(
            network,
            change,
        )


def scenario_changes_metadata(
    scenario: Scenario | None,
) -> list[dict]:
    if scenario is None:
        return []

    return [
        {
            "action": change.action.value,
            "element": change.element,
            "value": change.value,
        }
        for change in scenario.changes
    ]


def scenario_description(
    scenario: Scenario | None,
) -> str:
    if scenario is None:
        return "Validated 37-bus baseline."

    return scenario.description


def solve_case(
    scenario: Scenario | None = None,
) -> tuple[list[list], dict]:
    """
    Load a fresh 37-bus case, apply one scenario,
    solve the pre-fault power flow, and calculate
    short-circuit current for a fault at every bus.
    """

    case_path = ROOT / CASE_FILE

    network = CustomNetwork(
        str(case_path)
    )

    network.create_network()

    # ---------------------------------------------------------
    # Scenario modification
    # ---------------------------------------------------------

    apply_scenario(
        network,
        scenario,
    )

    # Build passive network after any topology changes.
    network.build_ybus()

    passive_ybus = network.YBus

    # ---------------------------------------------------------
    # Power flow
    # ---------------------------------------------------------

    pf_solver = PowerFlow()

    pf_result = pf_solver.solve_power_flow(
        network
    )

    if pf_result is None:
        raise PowerFlowConvergenceError(
            "Power flow did not converge."
        )

    theta, voltage = pf_result

    state = np.concatenate(
        (
            theta,
            voltage,
        )
    )

    v_complex = np.array(
        [
            v * np.exp(1j * angle)
            for angle, v in zip(
                theta,
                voltage,
            )
        ]
    )

    active_element_flow = (
        pf_solver.get_active_elements_flow(
            network,
            state,
        )
    )

    active_element_status = (
        pf_solver.get_active_element_status(
            network
        )
    )

    active_elements = (
        network.get_active_elements(
            v_complex,
            active_element_flow,
            active_element_status,
        )
    )

    case_data, totals = (
        pf_solver.power_flow_summary(
            network,
            state,
        )
    )

    # ---------------------------------------------------------
    # Short-circuit calculation
    # ---------------------------------------------------------

    i_scc_bus = []
    failed_buses = []

    print(
        "    Solving short-circuit using "
        "Newton-Raphson method..."
    )

    for bus in network.buses.keys():
        try:
            i_scc = SC.SCC_NR(
                network.idx[bus],
                v_complex,
                passive_ybus,
                active_elements,
            )

            i_scc_bus.append(
                i_scc
            )

        except RuntimeError:
            failed_buses.append(
                bus
            )

            i_scc_bus.append(
                None
            )

    # ---------------------------------------------------------
    # Append SCC and SCL quantities
    #
    # SCL definition:
    #
    #   SCL_pu = V_nominal_pu * I_SCC_pu
    #
    # with V_nominal_pu = 1.0:
    #
    #   SCL_pu = I_SCC_pu
    #
    # and:
    #
    #   SCL_MVA = S_BASE_MVA * SCL_pu
    # ---------------------------------------------------------

    for i, row in enumerate(case_data):
        i_scc = i_scc_bus[i]

        if i_scc is None:
            row.extend(
                [
                    None,  # I_SCC_mag_pu
                    None,  # I_SCC_ang_deg
                    None,  # SCL_pu
                    None,  # SCL_MVA
                ]
            )

            continue

        i_scc_mag_pu = abs(
            i_scc
        )

        i_scc_ang_deg = (
            cmath.phase(i_scc)
            * 180.0
            / cmath.pi
        )

        scl_pu = (
            1.0
            * i_scc_mag_pu
        )

        scl_mva = (
            S_BASE_MVA
            * scl_pu
        )

        row.extend(
            [
                i_scc_mag_pu,
                i_scc_ang_deg,
                scl_pu,
                scl_mva,
            ]
        )

    metadata = {
        "scenario": (
            "base"
            if scenario is None
            else scenario.name
        ),

        "description":
            scenario_description(
                scenario
            ),

        "case_file":
            CASE_FILE,

        "system_base_MVA":
            S_BASE_MVA,

        "scl_definition": (
            "SCL_pu = V_nominal_pu * I_SCC_pu, "
            "with V_nominal_pu = 1.0"
        ),

        "changes":
            scenario_changes_metadata(
                scenario
            ),

        "power_flow_converged":
            True,

        "short_circuit_attempted":
            True,

        "short_circuit_converged":
            len(failed_buses) == 0,

        "short_circuit_failed_buses":
            failed_buses,

        "totals":
            totals,
    }

    return (
        case_data,
        metadata,
    )


def save_metadata(
    scenario_name: str,
    metadata: dict,
) -> None:
    output_dir = (
        RAW_ROOT
        / scenario_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_file = (
        output_dir
        / "metadata.json"
    )

    with metadata_file.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"    Wrote {metadata_file}"
    )


def save_case(
    scenario_name: str,
    case_data: list[list],
    metadata: dict,
) -> None:
    output_dir = (
        RAW_ROOT
        / scenario_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_file = (
        output_dir
        / "results.csv"
    )

    with results_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            CSV_HEADER
        )

        writer.writerows(
            case_data
        )

    print(
        f"    Wrote {results_file}"
    )

    save_metadata(
        scenario_name,
        metadata,
    )


def run_one(
    name: str,
) -> None:
    if name == "base":
        scenario = None

    else:
        if name not in SCENARIOS:
            raise KeyError(
                f"Unknown scenario '{name}'. "
                f"Available scenarios: "
                f"{', '.join(SCENARIOS.keys())}"
            )

        scenario = SCENARIOS[name]

    description = (
        scenario_description(
            scenario
        )
    )

    print()
    print(
        f"Running {name}: "
        f"{description}"
    )

    try:
        case_data, metadata = (
            solve_case(
                scenario
            )
        )

    except PowerFlowConvergenceError:
        metadata = {
            "scenario":
                name,

            "description":
                description,

            "case_file":
                CASE_FILE,

            "system_base_MVA":
                S_BASE_MVA,

            "scl_definition": (
                "SCL_pu = V_nominal_pu * I_SCC_pu, "
                "with V_nominal_pu = 1.0"
            ),

            "changes":
                scenario_changes_metadata(
                    scenario
                ),

            "power_flow_converged":
                False,

            "short_circuit_attempted":
                False,

            "short_circuit_converged":
                False,

            "short_circuit_failed_buses":
                [],
        }

        save_metadata(
            name,
            metadata,
        )

        print(
            "    FAILED: power flow "
            "did not converge."
        )

        return

    save_case(
        name,
        case_data,
        metadata,
    )

    if metadata[
        "short_circuit_converged"
    ]:
        print(
            "    SCC convergence: OK"
        )

    else:
        failed = ", ".join(
            str(bus)
            for bus in metadata[
                "short_circuit_failed_buses"
            ]
        )

        print(
            "    WARNING: SCC did not converge "
            f"for bus(es): {failed}"
        )


def run_all() -> None:
    run_one(
        "base"
    )

    for name in SCENARIOS:
        run_one(
            name
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predefined 37-bus "
            "conference-paper short-circuit "
            "experiments."
        )
    )

    group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    group.add_argument(
        "--scenario",
        choices=[
            "base",
            *SCENARIOS.keys(),
        ],
        help=(
            "Run one predefined scenario."
        ),
    )

    group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run the baseline and all "
            "predefined scenarios."
        ),
    )

    args = parser.parse_args()

    if args.all:
        run_all()

    else:
        run_one(
            args.scenario
        )


if __name__ == "__main__":
    main()