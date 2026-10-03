
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

from models.CustomNetwork import CustomNetwork
from models.PowerFlow import PowerFlow
from models.ShortCircuit import ShortCircuit as SC
from experiments.glover37.scenarios import CASE_FILE

ROOT = Path(__file__).resolve().parents[2]

SCENARIOS = {
    "A2": ("generator.g28-1", "generator", "G28-1 disconnected"),
    "A3": ("generator.g14-1", "generator", "G14-1 disconnected"),
    "B1": ("line.line 14-34 ckt 1", "line", "Line 14-34 ckt 1 disconnected"),
    "B2": ("line.line 21-48 ckt 1", "line", "Line 21-48 ckt 1 disconnected"),
    "B3": ("transformer.transformer 28-29 ckt 1", "transformer", "Transformer 28-29 ckt 1 disconnected"),
}


def prepare_case(disconnect_element=None):
    network = CustomNetwork(str(ROOT / CASE_FILE))
    network.create_network()

    if disconnect_element is not None:
        network.change_element_status(disconnect_element, False)

    network.build_ybus()
    passive_ybus = network.YBus

    pf = PowerFlow()
    result = pf.solve_power_flow(network)
    if result is None:
        raise RuntimeError(f"Power flow did not converge: {disconnect_element}")

    theta, voltage = result
    state = np.concatenate((theta, voltage))
    v_complex = np.array(
        [v * np.exp(1j * a) for a, v in zip(theta, voltage)],
        dtype=complex,
    )

    flow = pf.get_active_elements_flow(network, state)
    status = pf.get_active_element_status(network)
    active = network.get_active_elements(v_complex, flow, status)

    return network, passive_ybus, v_complex, active


def run_all_faults(network, passive_ybus, v_complex, active):
    buses = list(network.buses.keys())
    currents = np.zeros(len(buses), dtype=complex)

    Y_ref = None
    max_y_variation = 0.0

    for k, bus in enumerate(buses):
        r = SC.SCC_NR(
            network.idx[bus],
            v_complex,
            passive_ybus,
            active,
            return_details=True,
        )

        currents[k] = r["I_fault"]
        Y = r["Y_SCC"]

        if Y_ref is None:
            Y_ref = Y.copy()
        else:
            max_y_variation = max(
                max_y_variation,
                float(np.max(np.abs((Y - Y_ref).toarray()))),
            )

    return currents, Y_ref, max_y_variation


def get_element(network, full_name, kind):
    groups = {
        "generator": network.generators,
        "line": network.lines,
        "transformer": network.transformers,
    }
    element = groups[kind][full_name]
    if not element.status:
        raise RuntimeError(f"{full_name} is not active in baseline.")
    return element


def build_terminal_stamp(network, element, kind):
    terminals = [element.bus_1] if kind == "generator" else [element.bus_1, element.bus_2]
    local = {bus: i for i, bus in enumerate(terminals)}

    Y_et = np.zeros((len(terminals), len(terminals)), dtype=complex)
    rows, cols, data = element.get_Y()

    for r, c, y in zip(rows, cols, data):
        Y_et[local[r], local[c]] += complex(y)

    E = np.zeros((len(network.buses), len(terminals)), dtype=complex)
    for j, bus in enumerate(terminals):
        E[network.idx[bus], j] = 1.0

    return terminals, E, Y_et


def general_update(Z, E, Y_et):
    middle = np.eye(Y_et.shape[0], dtype=complex) - E.T @ Z @ E @ Y_et
    correction = Z @ E @ Y_et @ np.linalg.inv(middle) @ E.T @ Z
    return Z + correction, middle, correction


def evaluate(name, full_name, kind, description, baseline):
    net0, I0, Y0, Z0, buses = baseline

    element = get_element(net0, full_name, kind)
    terminals, E, Y_et = build_terminal_stamp(net0, element, kind)

    Z_pred, middle, correction = general_update(Z0, E, Y_et)

    Y_element = E @ Y_et @ E.T
    Y_struct = Y0 - Y_element
    Z_direct = np.linalg.inv(Y_struct)

    alg_error = Z_pred - Z_direct

    Zii0 = np.diag(Z0)
    Zii_pred = np.diag(Z_pred)
    delta_Zii = Zii_pred - Zii0

    delta_scl_pred = 100.0 * (
        np.abs(Zii0) / np.abs(Zii_pred) - 1.0
    )

    net1, Yp1, V1, active1 = prepare_case(full_name)
    I1, Y1_sparse, yvar1 = run_all_faults(net1, Yp1, V1, active1)
    Y1 = Y1_sparse.toarray()

    delta_scl_sim = 100.0 * (
        np.abs(I1) / np.abs(I0) - 1.0
    )

    error = delta_scl_pred - delta_scl_sim
    order = np.argsort(np.abs(delta_scl_sim))[::-1]

    return {
        "name": name,
        "description": description,
        "element": full_name,
        "kind": kind,
        "terminals": terminals,
        "Y_et": Y_et,
        "rank": int(np.linalg.matrix_rank(Y_et)),
        "middle": middle,
        "cond": float(np.linalg.cond(middle)),
        "max_alg_error": float(np.max(np.abs(alg_error))),
        "mean_alg_error": float(np.mean(np.abs(alg_error))),
        "max_secondary_Y": float(np.max(np.abs(Y1 - Y_struct))),
        "mean_secondary_Y": float(np.mean(np.abs(Y1 - Y_struct))),
        "scenario_yvar": yvar1,
        "delta_Zii": delta_Zii,
        "delta_scl_pred": delta_scl_pred,
        "delta_scl_sim": delta_scl_sim,
        "error": error,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_error": float(np.max(np.abs(error))),
        "corr": float(np.corrcoef(delta_scl_pred, delta_scl_sim)[0, 1]),
        "order": order,
        "buses": buses,
    }


def print_result(r):
    print()
    print("=" * 72)
    print(f"{r['name']} — {r['description']}")
    print("=" * 72)
    print("Element:", r["element"])
    print("Terminals:", r["terminals"])
    print("Y_et:")
    print(r["Y_et"])
    print("rank(Y_et):", r["rank"])
    print("cond(I - E^T Z E Y_et):", f"{r['cond']:.8e}")
    print()

    print("General update vs direct structural inverse")
    print("-------------------------------------------")
    print("max |Z_update-Z_direct| =", f"{r['max_alg_error']:.16e}")
    print("mean|Z_update-Z_direct| =", f"{r['mean_alg_error']:.16e}")
    print()

    print("Actual scenario Y_SCC vs pure element removal")
    print("---------------------------------------------")
    print("max secondary |dY| =", f"{r['max_secondary_Y']:.16e}")
    print("mean secondary|dY| =", f"{r['mean_secondary_Y']:.16e}")
    print()

    print("Predicted vs simulated delta SCL")
    print("--------------------------------")
    print(f"MAE       : {r['mae']:.8f} percentage points")
    print(f"RMSE      : {r['rmse']:.8f} percentage points")
    print(f"max error : {r['max_error']:.8f} percentage points")
    print(f"corr      : {r['corr']:.10f}")
    print()

    print("Top 10 buses by actual |delta SCL|")
    print("----------------------------------")
    print(
        f"{'bus':>6} {'sim [%]':>12} {'pred [%]':>12} "
        f"{'error [pp]':>14} {'|dZii|':>12}"
    )
    for idx in r["order"][:10]:
        print(
            f"{r['buses'][idx]:>6} "
            f"{r['delta_scl_sim'][idx]:>12.6f} "
            f"{r['delta_scl_pred'][idx]:>12.6f} "
            f"{r['error'][idx]:>14.6f} "
            f"{abs(r['delta_Zii'][idx]):>12.6e}"
        )


def save_csv(r, out_dir):
    path = out_dir / f"{r['name']}_general_coupling.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "bus",
            "delta_scl_sim_pct",
            "delta_scl_pred_pct",
            "error_pp",
            "delta_Zii_real",
            "delta_Zii_imag",
            "abs_delta_Zii",
        ])
        for i, bus in enumerate(r["buses"]):
            dz = r["delta_Zii"][i]
            w.writerow([
                bus,
                r["delta_scl_sim"][i],
                r["delta_scl_pred"][i],
                r["error"][i],
                dz.real,
                dz.imag,
                abs(dz),
            ])
    return path


def save_plots(r, out_dir):
    order = r["order"]
    labels = [str(r["buses"][i]) for i in order]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.plot(x, r["delta_scl_sim"][order], marker="o", label="Simulated")
    plt.plot(
        x,
        r["delta_scl_pred"][order],
        marker="s",
        linestyle="--",
        label="General low-rank prediction",
    )
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(x, labels, rotation=90)
    plt.xlabel("Bus (ordered by actual |delta SCL|)")
    plt.ylabel("delta SCL [%]")
    plt.title(f"{r['name']}: simulated vs general prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path1 = out_dir / f"{r['name']}_general_response.png"
    plt.savefig(path1, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(r["delta_scl_sim"], r["delta_scl_pred"])

    lo = min(np.min(r["delta_scl_sim"]), np.min(r["delta_scl_pred"]))
    hi = max(np.max(r["delta_scl_sim"]), np.max(r["delta_scl_pred"]))
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="Perfect agreement")

    for i, bus in enumerate(r["buses"]):
        plt.annotate(
            str(bus),
            (r["delta_scl_sim"][i], r["delta_scl_pred"][i]),
            fontsize=8,
        )

    plt.xlabel("Simulated delta SCL [%]")
    plt.ylabel("Predicted delta SCL [%]")
    plt.title(f"{r['name']}: predicted vs simulated delta SCL")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path2 = out_dir / f"{r['name']}_general_scatter.png"
    plt.savefig(path2, dpi=200)
    plt.close()

    return path1, path2


def save_summary(results, out_dir):
    path = out_dir / "general_outage_coupling_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario",
            "element",
            "kind",
            "terminals",
            "rank_Yet",
            "cond_middle",
            "max_Z_algorithm_error",
            "max_secondary_Y",
            "MAE_pp",
            "RMSE_pp",
            "max_error_pp",
            "correlation",
        ])
        for r in results:
            w.writerow([
                r["name"],
                r["element"],
                r["kind"],
                "-".join(str(x) for x in r["terminals"]),
                r["rank"],
                r["cond"],
                r["max_alg_error"],
                r["max_secondary_Y"],
                r["mae"],
                r["rmse"],
                r["max_error"],
                r["corr"],
            ])
    return path


def main():
    net0, Yp0, V0, active0 = prepare_case()
    I0, Y0_sparse, yvar0 = run_all_faults(net0, Yp0, V0, active0)

    Y0 = Y0_sparse.toarray()
    Z0 = np.linalg.inv(Y0)
    buses = list(net0.buses.keys())

    baseline = (net0, I0, Y0, Z0, buses)

    print()
    print("General low-rank outage coupling diagnostic")
    print("===========================================")
    print("Baseline max recovered Y_SCC variation:", f"{yvar0:.16e}")

    out_dir = ROOT / "results" / "glover37" / "mechanism"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for name, (element, kind, description) in SCENARIOS.items():
        r = evaluate(
            name,
            element,
            kind,
            description,
            baseline,
        )
        results.append(r)

        print_result(r)
        print("Saved:", save_csv(r, out_dir))
        p1, p2 = save_plots(r, out_dir)
        print("Saved:", p1)
        print("Saved:", p2)

    print()
    print("=" * 108)
    print("SUMMARY")
    print("=" * 108)
    print(
        f"{'case':>5} {'rank':>5} {'MAE [pp]':>12} "
        f"{'RMSE [pp]':>12} {'max err [pp]':>14} "
        f"{'corr':>12} {'max Z alg err':>16} {'cond(middle)':>16}"
    )

    for r in results:
        print(
            f"{r['name']:>5} {r['rank']:>5} "
            f"{r['mae']:>12.6f} {r['rmse']:>12.6f} "
            f"{r['max_error']:>14.6f} {r['corr']:>12.8f} "
            f"{r['max_alg_error']:>16.3e} {r['cond']:>16.3e}"
        )

    print()
    print("Saved summary:", save_summary(results, out_dir))


if __name__ == "__main__":
    main()
