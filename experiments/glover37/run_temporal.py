"""
37-bus temporal-spatial short-circuit-level experiment.

Run from the repository root, e.g.:

    python -m experiments.glover37.run_temporal \
        --se3-file data/se3_2025.csv \
        --timestamp-column timestamp \
        --load-column SE3

Experiment design:
  • use the 2025 hourly SE3 demand series as a temporal scaling signal;
  • select the continuous 168-hour window with maximum Pmax-Pmin;
  • map the selected-week mean to the validated 37-bus baseline load;
  • scale every benchmark load P and Q by the same hourly factor;
  • use synchronous generators only in the temporal experiment;
  • rank non-slack generators by descending MWMax and dispatch sequentially
    within MWMin/MWMax;
  • keep slack bus 31 connected to balance residual demand and losses;
  • evaluate outage of line 39-47 circuit 1 at every hourly state;
  • freeze intact commitment and non-slack dispatch before applying the
    contingency;
  • evaluate 168 intact and 168 contingency states.

Outputs:
  selected_week.csv
  generator_priority.csv
  dispatch_schedule.csv
  run_summary.csv
  scl_timeseries.csv
  contingency_effect.csv
  metadata.json
"""

from __future__ import annotations

import argparse
import cmath
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. Paths and locked configuration
# =============================================================================

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None

from models.CustomNetwork import CustomNetwork  # noqa: E402
from models.PowerFlow import PowerFlow  # noqa: E402
from models.ShortCircuit import ShortCircuit as SC  # noqa: E402

CASE_FILE = ROOT / "cases" / "glover37.json"
GEN_LIMITS_FILE = ROOT / "data/powerworld/glover37/base/generators.csv"
OUTPUT_DIR = ROOT / "results/glover37/temporal"

S_BASE_MVA = 100.0
YEAR = 2025
WINDOW_HOURS = 168
SLACK_BUS = 31
CONTINGENCY = "line.line 39-47 ckt 1"
CONTINGENCY_CASE = "line_39_47_out"


@dataclass(frozen=True)
class GenLimit:
    key: str
    bus: int
    unit_id: str
    mw_min: float
    mw_max: float
    priority: int


# =============================================================================
# 2. Public SE3 load profile
# =============================================================================

def num(value) -> float:
    """Parse ordinary or PowerWorld-style decimal-comma numbers."""
    return float(str(value).strip().replace(" ", "").replace(",", "."))


def norm_name(value: str) -> str:
    return (str(value).strip().lower()
            .replace("å", "a").replace("ä", "a").replace("ö", "o")
            .replace("_", " ").replace("-", " "))


def choose_column(columns, explicit, kind):
    columns = list(columns)
    if explicit:
        if explicit not in columns:
            raise KeyError(f"{kind} column '{explicit}' not found. Columns: {columns}")
        return explicit

    normalized = {c: norm_name(c) for c in columns}
    if kind == "load":
        se3 = [c for c, n in normalized.items() if "se3" in n]
        if len(se3) == 1:
            return se3[0]
        tokens = ("consumption", "load", "demand", "forbrukning")
    else:
        tokens = ("timestamp", "datetime", "date time", "datum", "period", "time")

    matches = [c for c, n in normalized.items() if any(t in n for t in tokens)]
    if len(matches) != 1:
        raise ValueError(
            f"Could not identify one unique {kind} column. Candidates: {matches}. "
            f"Pass --{kind}-column explicitly."
        )
    return matches[0]


def read_se3(path: Path, timestamp_col=None, load_col=None) -> pd.DataFrame:
    """Read the exact public SE3 source file retained with the experiment."""
    if not path.exists():
        raise FileNotFoundError(f"SE3 file not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(path, sep=None, engine="python")
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            raw = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                "Excel input needs an appropriate pandas engine. "
                "Easiest reproducible option: export the official file to CSV."
            ) from exc
    else:
        raise ValueError("SE3 input must be CSV, XLS, or XLSX.")

    tcol = choose_column(raw.columns, timestamp_col, "timestamp")
    lcol = choose_column(raw.columns, load_col, "load")

    df = raw[[tcol, lcol]].copy()
    df.columns = ["timestamp", "se3_mw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["se3_mw"] = df["se3_mw"].map(
        lambda x: num(x) if not pd.isna(x) else np.nan
    )
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"].dt.year == YEAR].copy()

    if df["timestamp"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. Resolve timezone/DST representation first."
        )
    if len(df) < WINDOW_HOURS:
        raise ValueError(f"Need >= {WINDOW_HOURS} usable hourly observations.")

    return df


def select_week(profile: pd.DataFrame) -> pd.DataFrame:
    """Pick max-range continuous 168 h window using load data only."""
    best_start, best_range = None, -np.inf

    for start in range(len(profile) - WINDOW_HOURS + 1):
        w = profile.iloc[start:start + WINDOW_HOURS]
        steps = w["timestamp"].diff().dropna().dt.total_seconds() / 3600
        if not np.allclose(steps.to_numpy(), 1.0):
            continue

        spread = float(w["se3_mw"].max() - w["se3_mw"].min())
        if spread > best_range:
            best_start, best_range = start, spread

    if best_start is None:
        raise ValueError("No complete continuous 168-hour window found.")

    w = profile.iloc[best_start:best_start + WINDOW_HOURS].copy().reset_index(drop=True)
    w["hour_index"] = np.arange(WINDOW_HOURS)
    w["load_scale"] = w["se3_mw"] / w["se3_mw"].mean()
    return w[["hour_index", "timestamp", "se3_mw", "load_scale"]]


# =============================================================================
# 3. Generator priority and sequential dispatch
# =============================================================================

def read_gen_limits(path: Path) -> list[GenLimit]:
    """Read MWMin/MWMax from the existing PowerWorld generator export."""
    raw = pd.read_csv(path, skiprows=1, dtype=str)
    need = {"BusNum", "ID", "Status", "MWMin", "MWMax"}
    if missing := need.difference(raw.columns):
        raise KeyError(f"Missing columns in generators.csv: {sorted(missing)}")

    rows = []
    for _, r in raw.iterrows():
        bus, uid = int(r["BusNum"].strip()), r["ID"].strip()
        pmin, pmax = num(r["MWMin"]), num(r["MWMax"])
        if bus == SLACK_BUS or r["Status"].strip().lower() != "closed" or pmax <= 0:
            continue
        rows.append((bus, uid, pmin, pmax))

    # Locked rule: descending MWMax. Deterministic tie-break independent of SCL.
    rows.sort(key=lambda x: (-x[3], x[0], x[1]))
    return [
        GenLimit(f"generator.g{bus}-{uid}".lower(), bus, uid, pmin, pmax, i + 1)
        for i, (bus, uid, pmin, pmax) in enumerate(rows)
    ]


def sequential_dispatch(target_load_mw: float, limits: list[GenLimit]) -> dict[str, float]:
    """
    Fill non-slack generators in priority order.

    If remaining demand is below the next generator MWMin, stop; slack supplies
    that residual. If non-slack MWMax is exhausted, slack supplies the remainder.
    """
    remaining = max(float(target_load_mw), 0.0)
    dispatch = {g.key: 0.0 for g in limits}

    for g in limits:
        if remaining <= 0 or remaining < g.mw_min:
            break
        p = min(g.mw_max, remaining)
        dispatch[g.key] = max(g.mw_min, p)
        remaining -= dispatch[g.key]

    return dispatch


def baseline_load_mw() -> float:
    net = CustomNetwork(str(CASE_FILE))
    net.create_network()
    return S_BASE_MVA * sum(load.P for load in net.loads.values())


def build_schedule(week, limits):
    base_mw = baseline_load_mw()
    records, dispatch_by_hour = [], {}

    for row in week.itertuples(index=False):
        target = base_mw * float(row.load_scale)
        dispatch = sequential_dispatch(target, limits)
        dispatch_by_hour[int(row.hour_index)] = dispatch

        for g in limits:
            p = dispatch[g.key]
            records.append({
                "hour_index": row.hour_index, "timestamp": row.timestamp,
                "generator": g.key, "bus": g.bus, "priority": g.priority,
                "MWMin": g.mw_min, "MWMax": g.mw_max,
                "committed": p > 0, "dispatch_MW": p,
                "target_load_MW": target,
            })

    return pd.DataFrame(records), dispatch_by_hour


# =============================================================================
# 4. One quasi-static snapshot
# =============================================================================

def prepare_network(load_scale, dispatch, limits, contingency=False):
    net = CustomNetwork(str(CASE_FILE))
    net.create_network()

    for load in net.loads.values():
        load.P *= load_scale
        load.Q *= load_scale

    for g in limits:
        if g.key not in net.generators:
            raise KeyError(f"{g.key} not found in glover37.json")
        p_mw = dispatch[g.key]
        net.change_element_status(g.key, p_mw > 0)
        if p_mw > 0:
            net.generators[g.key].P = p_mw / S_BASE_MVA

    # Slack generator is deliberately untouched: always connected/balancing.
    if contingency:
        net.change_element_status(CONTINGENCY, False)

    net.build_ybus()
    return net


def solve_snapshot(load_scale, dispatch, limits, contingency=False):
    net = prepare_network(load_scale, dispatch, limits, contingency)
    pf = PowerFlow()
    solved = pf.solve_power_flow(net)

    if solved is None:
        return [], [], False

    theta, voltage = solved
    state = np.concatenate((theta, voltage))
    v_complex = np.array([v * np.exp(1j * a) for a, v in zip(theta, voltage)])

    flows = pf.get_active_elements_flow(net, state)
    statuses = pf.get_active_element_status(net)
    active = net.get_active_elements(v_complex, flows, statuses)
    case_data, _ = pf.power_flow_summary(net, state)

    failed, rows = [], []
    for i, bus in enumerate(net.buses.keys()):
        try:
            current = SC.SCC_NR(net.idx[bus], v_complex, net.YBus, active)
        except RuntimeError:
            current = None
            failed.append(bus)

        if current is None:
            imag = iang = scl_pu = scl_mva = np.nan
        else:
            imag = abs(current)
            iang = cmath.phase(current) * 180 / cmath.pi
            scl_pu = imag
            scl_mva = S_BASE_MVA * scl_pu

        rows.append({
            "bus": case_data[i][0],
            "pre_fault_V_mag": case_data[i][1],
            "pre_fault_V_ang_deg": case_data[i][2],
            "I_SCC_mag_pu": imag,
            "I_SCC_ang_deg": iang,
            "SCL_pu": scl_pu,
            "SCL_MVA": scl_mva,
        })

    return rows, failed, True


# =============================================================================
# 5. Run intact + fixed contingency for every hour
# =============================================================================

def run_timeseries(week, limits, dispatch_by_hour):
    scl_rows, summary = [], []

    for row in week.itertuples(index=False):
        h = int(row.hour_index)
        dispatch = dispatch_by_hour[h]
        committed = sum(p > 0 for p in dispatch.values())

        print(
            f"[{h + 1:03d}/{WINDOW_HOURS}] {row.timestamp} "
            f"scale={row.load_scale:.4f} committed={committed}"
        )

        for case, outage in (("intact", False), (CONTINGENCY_CASE, True)):
            buses, failed, pf_ok = solve_snapshot(
                float(row.load_scale), dispatch, limits, outage
            )

            summary.append({
                "hour_index": h, "timestamp": row.timestamp, "case": case,
                "load_scale": row.load_scale, "se3_mw": row.se3_mw,
                "pf_converged": pf_ok, "scc_failed_bus_count": len(failed),
                "scc_failed_buses": ";".join(map(str, failed)),
                "committed_non_slack": committed,
                "non_slack_dispatch_MW": sum(dispatch.values()),
            })

            for b in buses:
                scl_rows.append({
                    "hour_index": h, "timestamp": row.timestamp, "case": case,
                    "load_scale": row.load_scale, "se3_mw": row.se3_mw, **b,
                })

    return pd.DataFrame(scl_rows), pd.DataFrame(summary)


def contingency_effect(scl):
    keys = ["hour_index", "timestamp", "bus"]
    a = (scl[scl["case"] == "intact"][keys + ["SCL_MVA"]]
         .rename(columns={"SCL_MVA": "SCL_intact_MVA"}))
    b = (scl[scl["case"] == CONTINGENCY_CASE][keys + ["SCL_MVA"]]
         .rename(columns={"SCL_MVA": "SCL_contingency_MVA"}))
    out = a.merge(b, on=keys, how="outer", validate="one_to_one")
    out["delta_SCL_MVA"] = out["SCL_contingency_MVA"] - out["SCL_intact_MVA"]
    out["delta_SCL_pct"] = 100 * out["delta_SCL_MVA"] / out["SCL_intact_MVA"]
    return out.sort_values(keys).reset_index(drop=True)


# =============================================================================
# 6. Reproducibility outputs
# =============================================================================

def save_outputs(source, week, limits, schedule, scl, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    effect = contingency_effect(scl)

    week.to_csv(OUTPUT_DIR / "selected_week.csv", index=False)
    schedule.to_csv(OUTPUT_DIR / "dispatch_schedule.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)
    scl.to_csv(OUTPUT_DIR / "scl_timeseries.csv", index=False)
    effect.to_csv(OUTPUT_DIR / "contingency_effect.csv", index=False)

    pd.DataFrame([{
        "priority": g.priority, "generator": g.key, "bus": g.bus,
        "unit_id": g.unit_id, "MWMin": g.mw_min, "MWMax": g.mw_max,
    } for g in limits]).to_csv(OUTPUT_DIR / "generator_priority.csv", index=False)

    metadata = {
        "experiment": "37-bus temporal-spatial short-circuit-level assessment",
        "network": CASE_FILE.resolve().relative_to(ROOT.resolve()).as_posix(),
        "SE3": {
            "year": YEAR,
            "source_file": source.resolve().relative_to(ROOT.resolve()).as_posix(),
            "role": "temporal scaling only; 37-bus benchmark is not SE3",
        },
        "week_selection": {
            "criterion": "largest Pmax-Pmin over continuous 168-hour windows",
            "uses_SCL_results": False,
            "start": str(week.timestamp.iloc[0]),
            "end": str(week.timestamp.iloc[-1]),
            "min_MW": float(week.se3_mw.min()),
            "max_MW": float(week.se3_mw.max()),
            "mean_MW": float(week.se3_mw.mean()),
        },
        "load_mapping": {
            "baseline_total_load_MW": baseline_load_mw(),
            "formula": "lambda=SE3/weekly_mean; P_i=lambda P_i0; Q_i=lambda Q_i0",
        },
        "dispatch": {
            "slack_bus": SLACK_BUS,
            "slack_in_commitment_order": False,
            "priority": "descending MWMax",
            "tie_break": "ascending BusNum then ID",
            "allocation": "sequential loading within MWMin/MWMax",
            "limits_source": GEN_LIMITS_FILE.resolve().relative_to(ROOT.resolve()).as_posix(),
        },
        "contingency": {
            "element": CONTINGENCY,
            "selection_basis": (
                "Selected a priori from PowerWorld contingency analysis as "
                "the critical case according to that analysis."
            ),
            "intact_commitment_and_non_slack_dispatch_frozen": True,
        },
        "states": {"intact": 168, "contingency": 168, "total": 336},
        "additional_temporal_PowerWorld_SCC_validation": False,
    }

    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Results: {OUTPUT_DIR}")


# =============================================================================
# 7. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run the temporal-spatial SCL experiment on the Glover 37-bus system."
    )
    parser.add_argument("--se3-file", type=Path, required=True)
    parser.add_argument("--timestamp-column", default=None)
    parser.add_argument("--load-column", default=None)
    args = parser.parse_args()

    profile = read_se3(args.se3_file, args.timestamp_column, args.load_column)
    week = select_week(profile)

    print(
        f"Selected week: {week.timestamp.iloc[0]} -> {week.timestamp.iloc[-1]}\n"
        f"SE3: {week.se3_mw.min():.2f} -> {week.se3_mw.max():.2f} MW "
        f"(range {week.se3_mw.max() - week.se3_mw.min():.2f} MW)\n"
    )

    limits = read_gen_limits(GEN_LIMITS_FILE)
    print("Non-slack priority:")
    for g in limits:
        print(
            f"  {g.priority}. {g.key:<18} "
            f"MWMin={g.mw_min:6.1f} MWMax={g.mw_max:6.1f}"
        )

    schedule, dispatch_by_hour = build_schedule(week, limits)
    scl, summary = run_timeseries(week, limits, dispatch_by_hour)
    save_outputs(args.se3_file, week, limits, schedule, scl, summary)


if __name__ == "__main__":
    main()
