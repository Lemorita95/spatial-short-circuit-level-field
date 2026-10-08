"""
Run the Glover 37-bus structural-mechanism experiment.

For each structural scenario A2, A3, B1, B2, and B3, the script evaluates
the exact low-rank admittance-matrix update, computes the corresponding
driving-point-impedance prediction, and solves the modified network for
comparison.

This stage performs the numerical experiment only. Error metrics and
aggregate summaries are computed by analyze_mechanism.py.

Usage
-----
Run all mechanism scenarios:

    python -m experiments.glover37.run_mechanism --all

Run one scenario:

    python -m experiments.glover37.run_mechanism --scenario A2

Inputs
------
cases/glover37.json

Outputs
-------
results/glover37/mechanism/raw/<scenario>/results.csv
results/glover37/mechanism/raw/<scenario>/metadata.json

The mechanism scenarios are:
    A2, A3, B1, B2, B3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.CustomNetwork import CustomNetwork
from models.PowerFlow import PowerFlow
from models.ShortCircuit import ShortCircuit as SC
from experiments.glover37.scenarios import CASE_FILE


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "results" / "glover37" / "mechanism" / "raw"

SCENARIOS = {
    "A2": ("generator.g28-1", "generator", "G28-1 disconnected"),
    "A3": ("generator.g14-1", "generator", "G14-1 disconnected"),
    "B1": ("line.line 14-34 ckt 1", "line", "Line 14-34 ckt 1 disconnected"),
    "B2": ("line.line 21-48 ckt 1", "line", "Line 21-48 ckt 1 disconnected"),
    "B3": (
        "transformer.transformer 28-29 ckt 1",
        "transformer",
        "Transformer 28-29 ckt 1 disconnected",
    ),
}


def prepare_case(disconnect_element: str | None = None):
    """Build one case, solve its power flow, and recover SCC source data."""
    network = CustomNetwork(str(ROOT / CASE_FILE))
    network.create_network()

    if disconnect_element is not None:
        network.change_element_status(disconnect_element, False)

    network.build_ybus()
    passive_ybus = network.YBus

    pf = PowerFlow()
    result = pf.solve_power_flow(network)
    if result is None:
        raise RuntimeError(
            f"Power flow did not converge: {disconnect_element}"
        )

    theta, voltage = result
    state = np.concatenate((theta, voltage))
    v_complex = np.array(
        [v * np.exp(1j * a) for a, v in zip(theta, voltage)],
        dtype=complex,
    )

    flow = pf.get_active_elements_flow(network, state)
    status = pf.get_active_element_status(network)
    active = network.get_active_elements(
        v_complex,
        flow,
        status,
    )

    return network, passive_ybus, v_complex, active


def run_all_faults(
    network,
    passive_ybus,
    v_complex,
    active,
):
    """Run a fault at every bus and retain the common SCC admittance matrix."""
    buses = list(network.buses.keys())
    currents = np.zeros(len(buses), dtype=complex)

    y_ref = None
    max_y_variation = 0.0

    for k, bus in enumerate(buses):
        result = SC.SCC_NR(
            network.idx[bus],
            v_complex,
            passive_ybus,
            active,
            return_details=True,
        )

        currents[k] = result["I_fault"]
        y_scc = result["Y_SCC"]

        if y_ref is None:
            y_ref = y_scc.copy()
        else:
            max_y_variation = max(
                max_y_variation,
                float(
                    np.max(
                        np.abs(
                            (y_scc - y_ref).toarray()
                        )
                    )
                ),
            )

    return currents, y_ref, max_y_variation


def get_element(network, full_name: str, kind: str):
    """Return the active modified element from the baseline network."""
    groups = {
        "generator": network.generators,
        "line": network.lines,
        "transformer": network.transformers,
    }

    element = groups[kind][full_name]

    if not element.status:
        raise RuntimeError(
            f"{full_name} is not active in baseline."
        )

    return element


def build_terminal_stamp(network, element, kind: str):
    """Build the terminal selector E and local element stamp Y_et."""
    terminals = (
        [element.bus_1]
        if kind == "generator"
        else [element.bus_1, element.bus_2]
    )

    local = {
        bus: i
        for i, bus in enumerate(terminals)
    }

    y_et = np.zeros(
        (len(terminals), len(terminals)),
        dtype=complex,
    )

    rows, cols, data = element.get_Y()

    for row, col, value in zip(rows, cols, data):
        y_et[local[row], local[col]] += complex(value)

    e = np.zeros(
        (len(network.buses), len(terminals)),
        dtype=complex,
    )

    for j, bus in enumerate(terminals):
        e[network.idx[bus], j] = 1.0

    return terminals, e, y_et


def general_update(z, e, y_et):
    """Apply the exact low-rank inverse update for element removal."""
    middle = (
        np.eye(y_et.shape[0], dtype=complex)
        - e.T @ z @ e @ y_et
    )

    correction = (z @ e @ y_et @ np.linalg.inv(middle) @ e.T @ z)

    return z + correction, middle


def build_baseline():
    """Solve the common baseline once for one or more mechanism scenarios."""
    network, passive_ybus, v_complex, active = prepare_case()

    currents, y_sparse, y_variation = run_all_faults(
        network,
        passive_ybus,
        v_complex,
        active,
    )

    y = y_sparse.toarray()
    z = np.linalg.inv(y)
    buses = list(network.buses.keys())

    return {
        "network": network,
        "currents": currents,
        "Y": y,
        "Z": z,
        "buses": buses,
        "max_y_variation": y_variation,
    }


def run_scenario(
    name: str,
    baseline: dict,
) -> tuple[pd.DataFrame, dict]:
    """Run one structural scenario and return raw bus-wise results + metadata."""
    full_name, kind, description = SCENARIOS[name]

    net0 = baseline["network"]
    i0 = baseline["currents"]
    y0 = baseline["Y"]
    z0 = baseline["Z"]
    buses = baseline["buses"]

    element = get_element(
        net0,
        full_name,
        kind,
    )

    terminals, e, y_et = build_terminal_stamp(
        net0,
        element,
        kind,
    )

    z_pred, middle = general_update(
        z0,
        e,
        y_et,
    )

    y_element = e @ y_et @ e.T
    y_struct = y0 - y_element
    z_direct = np.linalg.inv(y_struct)
    z_algorithm_error = z_pred - z_direct

    zii0 = np.diag(z0)
    zii_pred = np.diag(z_pred)
    delta_zii = zii_pred - zii0

    delta_scl_pred = 100.0 * (
        np.abs(zii0) / np.abs(zii_pred)
        - 1.0
    )

    net1, yp1, v1, active1 = prepare_case(
        full_name
    )

    i1, y1_sparse, scenario_y_variation = run_all_faults(
        net1,
        yp1,
        v1,
        active1,
    )

    y1 = y1_sparse.toarray()

    delta_scl_sim = 100.0 * (
        np.abs(i1) / np.abs(i0)
        - 1.0
    )

    results = pd.DataFrame(
        {
            "bus": buses,
            "delta_scl_sim_pct": delta_scl_sim,
            "delta_scl_pred_pct": delta_scl_pred,
            "delta_Zii_real": delta_zii.real,
            "delta_Zii_imag": delta_zii.imag,
            "abs_delta_Zii": np.abs(delta_zii),
        }
    )

    metadata = {
        "scenario": name,
        "description": description,
        "element": full_name,
        "kind": kind,
        "terminals": terminals,
        "rank_Yet": int(
            np.linalg.matrix_rank(y_et)
        ),
        "cond_middle": float(
            np.linalg.cond(middle)
        ),
        "max_Z_algorithm_error": float(
            np.max(
                np.abs(z_algorithm_error)
            )
        ),
        "mean_Z_algorithm_error": float(
            np.mean(
                np.abs(z_algorithm_error)
            )
        ),
        "max_secondary_Y": float(
            np.max(
                np.abs(y1 - y_struct)
            )
        ),
        "mean_secondary_Y": float(
            np.mean(
                np.abs(y1 - y_struct)
            )
        ),
        "baseline_max_Y_variation": float(
            baseline["max_y_variation"]
        ),
        "scenario_max_Y_variation": float(
            scenario_y_variation
        ),
    }

    return results, metadata


def save_scenario(
    name: str,
    results: pd.DataFrame,
    metadata: dict,
) -> None:
    """Write one scenario's raw mechanism outputs."""
    output_dir = RAW_ROOT / name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = output_dir / "results.csv"
    metadata_path = output_dir / "metadata.json"

    results.to_csv(
        results_path,
        index=False,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {results_path}")
    print(f"Wrote: {metadata_path}")


def run_one(
    name: str,
    baseline: dict | None = None,
) -> None:
    """Run and save one mechanism scenario."""
    if baseline is None:
        baseline = build_baseline()

    print()
    print(
        f"Running mechanism {name}: "
        f"{SCENARIOS[name][2]}"
    )

    results, metadata = run_scenario(
        name,
        baseline,
    )

    save_scenario(
        name,
        results,
        metadata,
    )


def run_all() -> None:
    """Run all predefined structural mechanism scenarios."""
    baseline = build_baseline()

    for name in SCENARIOS:
        run_one(
            name,
            baseline,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the 37-bus structural-mechanism experiment."
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--scenario",
        choices=list(SCENARIOS),
        help="Run one mechanism scenario.",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Run all mechanism scenarios.",
    )

    args = parser.parse_args()

    if args.all:
        run_all()
    else:
        run_one(args.scenario)


if __name__ == "__main__":
    main()
