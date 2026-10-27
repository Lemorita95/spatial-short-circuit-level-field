"""
Analyze the system-strength metrics experiment: Gate 4, evidence plan v2.

Plan: current_research_handoff/05_GATE3.5_LITERATURE_EXTRACTION.md, Part E.
Every claim is an observation, a mechanism (the formula term responsible) and a
confirmation: that term is isolated, and the effect must disappear or be
accounted for. No network, power-flow or short-circuit calculation is performed,
except the topology-only update of the stored intact Z (D4).

Common basis (file 02 §4), Z = Y_SC^-1, SCMVA_i = 100 |V_i| / |Z_ii|:

    SCR_i   = SCMVA_i / P_i                          37-bus, 69 kV POI
    WSCR    = Σ SCMVA_i P_i / (Σ P_i)²               37-bus, 69 kV POI
    ESCR_i  = SCMVA_i / (P_i + Σ IF_ji P_j)          46-bus, 34.5 kV MV
    SDSCR_i = |V_i|² / (|Z_ii| |P_i + Σ w_ij P_j|)   46-bus, 0.69 kV LV

Isolating terms. Each metric m is recomputed from its recorded terms as

    m_noV   |V| = 1 pu, coupling kept
    m_noC   coupling removed (IF = 0, w = 0), |V| kept
    m_Z     both removed: Z_ii only

so a change in ln m between two states (hours or scenarios) splits exactly into

    Δln m = (Δln m - Δln m_noV) + (Δln m_noV - Δln m_Z) + Δln m_Z
             |V| part              coupling part           Z_ii part

SCR and WSCR have no coupling term (m_noC = m, m_noV = m_Z).

D4 topology. Removing line a-b adds ΔY = U C U^T to Y_SC (U = [e_a, e_b],
C = minus the line's π stamp). Its effect on the intact Z is exact (Woodbury):

    Z' = Z - Z U (I + C U^T Z U)^-1 C U^T Z

This is the topology-only outage effect; the rest of the actual change comes
from load admittances at the changed voltages. It splits into two opposing parts:
removing the series branch raises |Z_ii|; removing the charging shunts (B/2 at
each end, capacitive) lowers it, since jX in parallel with jB gives jX/(1 - XB).
The series part is computed with the series branch alone; the shunt part is the rest.

Coupling load, the interaction a site carries relative to its own power:
    ESCR  k_i = Σ IF_ji P_j / P_i          SDSCR  k_i = |P_i + Σ w_ij P_j| / P_i - 1
An outage that decouples the sites lowers k, which raises ESCR and SDSCR even
while the network weakens.

Claims (Part E §4) and tables
-----------------------------
D1/RQ2  The weakest hour is a minimum-commitment hour (B_count); exceptions are
        traced to |V| or coupling                       -> d1_weakest_hour
D4      In network terms an outage weakens a site or leaves it practically
        unaffected; the topology-only ΔZ_ii explains the change  -> d4_outage
D4      Measured per hour, contingency and site: metric_contingency > metric_intact
        ("stronger"). The coupling metrics are stronger when the outage decouples
        the sites (coupling load k falls)                -> d4_outage
        r per hour, corridor, metric and site (histograms)  -> d4_hourly
D5      The light-load snapshot almost finds the weekly minimum
                                                         -> d5_snapshots, d5_dispatch
A3      Rated, output and MVA denominators change no conclusion
                                                         -> a3_denominators
check   Parallel circuits agree with circuit 1          -> parallel_circuits
check   ESCR conclusions under other IF probe steps (Gate 3 limitation 2;
        with --if-step-run)                              -> if_step_robustness
summary.csv: one line per result, traced to its claim.

Scenarios: intact plus one outage per corridor; circuit 1 represents a corridor
of parallel circuits (Part E §5). D4 counts corridors.

Usage
-----
    python -m experiments.glover37.analyze_strength_metrics
    python -m experiments.glover37.analyze_strength_metrics \
        --input-dir results/glover37/strength/raw \
        --output-dir results/glover37/strength/analysis \
        --if-step-run results/glover37/strength/raw_if_step_0.005 \
        --if-step-run results/glover37/strength/raw_if_step_0.02
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glover37.run_temporal import (
    GEN_LIMITS_FILE, ROOT, S_BASE_MVA, prepare_network, read_gen_limits, sequential_dispatch,
)
from experiments.glover37.strength_config import SITES

STRENGTH_ROOT = ROOT / "results" / "glover37" / "strength"
DEFAULT_INPUT = STRENGTH_ROOT / "raw"
DEFAULT_OUTPUT = STRENGTH_ROOT / "analysis"
INPUT_FILES = ("metrics_timeseries.csv", "metric_variables.csv", "run_summary.csv",
               "dispatch_schedule.csv", "contingency_set.csv", "z_full_intact.npz")

INTACT = "intact"
METRICS = ("SCR", "WSCR", "ESCR", "SDSCR")
SITE_IDS = tuple(str(s.host_bus) for s in SITES)
GROUP = "group"
UNITS = tuple((m, s) for m in METRICS for s in ((GROUP,) if m == "WSCR" else SITE_IDS))
VARIANTS = ("m", "noV", "noC", "Z")

EVAL_BUS = {
    "SCR": {str(s.host_bus): s.poi_bus for s in SITES},
    "WSCR": {str(s.host_bus): s.poi_bus for s in SITES},
    "ESCR": {str(s.host_bus): s.escr_bus for s in SITES},
    "SDSCR": {str(s.host_bus): s.sdscr_bus for s in SITES},
}
NETWORK = {"SCR": "base", "WSCR": "base", "ESCR": "ibr", "SDSCR": "ibr"}
N_COMMITTED_COLUMN = {"base": "base_n_committed", "ibr": "ibr_case_n_committed"}
P_MW = {
    "rated": {str(s.host_bus): s.pmax_mw for s in SITES},
    "output": {str(s.host_bus): s.operating_mw for s in SITES},
    "mva": {str(s.host_bus): s.converter_mva for s in SITES},
}
AS_RUN = {"SCR": "rated", "WSCR": "output", "ESCR": "rated", "SDSCR": "output"}

REL_TIE = 1e-9
IF_ZERO = 3e-5  # IF ≈ 0 state: every IF < 3e-5 (Gate 3 record, file 02 §5)
RECONSTRUCTION_TOL = 1e-9


# =============================================================================
# 1. Inputs
# =============================================================================

def _bus_id(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return str(x)


def _join(items):
    return ";".join(str(x) for x in items)


def load_inputs(input_dir, reference=None):
    """A run's files. An IF-step rerun (reference given) reuses the reference run's
    contingency set and intact Z, so it needs only the four files the runner writes."""
    d = Path(input_dir)
    files = INPUT_FILES if reference is None else INPUT_FILES[:4]
    if missing := [f for f in files if not (d / f).is_file()]:
        raise FileNotFoundError(f"Missing inputs in {d}: {missing}")
    metrics = pd.read_csv(d / "metrics_timeseries.csv", dtype={"bus": str})
    metrics["bus"] = metrics["bus"].map(_bus_id)
    variables = pd.read_csv(d / "metric_variables.csv", dtype={"bus": str, "related_bus": str})
    variables["bus"] = variables["bus"].map(_bus_id)
    variables["related_bus"] = variables["related_bus"].map(_bus_id)
    if reference is None:
        z = np.load(d / "z_full_intact.npz")
        contingencies, z = pd.read_csv(d / "contingency_set.csv"), {k: z[k] for k in z.files}
    else:
        contingencies, z = reference["contingencies"], reference["z"]
    return {
        "metrics": metrics,
        "variables": dict(tuple(variables.groupby(["metric", "variable"]))),
        "summary": pd.read_csv(d / "run_summary.csv"),
        "dispatch": pd.read_csv(d / "dispatch_schedule.csv"),
        "contingencies": contingencies,
        "z": z,
        "sha256": {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in files},
    }


def scenarios(contingencies):
    """Representative case per corridor (circuit 1), and {case: representative}."""
    inc = contingencies[contingencies["included"].astype(bool)]
    representative = {}
    for _, grp in inc.groupby(inc.apply(lambda r: tuple(sorted((r.from_bus, r.to_bus))), axis=1),
                              sort=False):
        rep = next((r.case for r in grp.itertuples() if r.line.endswith("ckt 1")), grp["case"].iloc[0])
        representative.update({c: rep for c in grp["case"]})
    corridors = [c for c in inc["case"] if representative[c] == c]
    return corridors, representative


def check_inputs(data):
    """Fail on anything incomplete; never treat a missing state as valid."""
    summary, metrics = data["summary"], data["metrics"]
    cases = [INTACT] + list(data["contingencies"].query("included")["case"])
    if set(summary["case"]) != set(cases):
        raise ValueError(f"run_summary cases differ from contingency_set: "
                         f"{sorted(set(cases) ^ set(summary['case']))}")
    hours = sorted(summary["hour_index"].unique())
    for case, s in summary.groupby("case"):
        if sorted(s["hour_index"]) != hours:
            raise ValueError(f"{case}: hours differ from the other cases or repeat")
    if not (summary["base_pf_ok"].astype(bool).all() and summary["ibr_pf_ok"].astype(bool).all()):
        raise ValueError("Unconverged power flow in run_summary.csv")
    if summary["error"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("Errors recorded in run_summary.csv")
    if len(metrics) != len(hours) * len(cases) * len(UNITS):
        raise ValueError(f"metrics_timeseries.csv has {len(metrics)} rows, expected "
                         f"{len(hours) * len(cases) * len(UNITS)}")
    v = metrics["value"].to_numpy(float)
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Non-finite or non-positive metric values")
    for column in N_COMMITTED_COLUMN.values():
        if summary.groupby("hour_index")[column].nunique().max() > 1:
            raise ValueError(f"{column}: commitment differs between scenarios")
    if sorted(data["z"]["hour_index"]) != hours:
        raise ValueError("z_full_intact.npz hours differ from run_summary.csv")
    return np.asarray(hours, dtype=int), cases


# =============================================================================
# 2. Metrics and their isolated-term variants
# =============================================================================

def _wide(groups, metric, variable, part="real", pair=False):
    rows = groups.get((metric, variable))
    if rows is None:
        raise ValueError(f"No {metric} {variable} rows in metric_variables.csv")
    if part == "complex":
        vals = rows["real"].to_numpy(float) + 1j * rows["imag"].to_numpy(float)
    else:
        vals = rows[part].to_numpy(float)
    keys = ["case", "hour_index", "bus"] + (["related_bus"] if pair else [])
    s = pd.Series(vals, index=pd.MultiIndex.from_frame(rows[keys]))
    if s.index.duplicated().any():
        raise ValueError(f"Duplicate {metric} {variable} rows")
    return s.unstack(["bus", "related_bus"] if pair else "bus").sort_index()


def _by_site(frame, metric):
    site_of = {b: s for s, b in EVAL_BUS[metric].items()}
    return frame.rename(columns=site_of)[list(SITE_IDS)]


def variant_values(groups, p_mw):
    """{variant: {metric: DataFrame(index (case, hour_index), columns site)}}, P per metric."""
    out = {v: {} for v in VARIANTS}

    p = pd.Series(p_mw["SCR"])[list(SITE_IDS)]
    v_abs = _by_site(_wide(groups, "SCR", "v_pf", "complex"), "SCR").abs()
    z_abs = _by_site(_wide(groups, "SCR", "z_ii", "complex"), "SCR").abs()
    out["m"]["SCR"] = out["noC"]["SCR"] = S_BASE_MVA * v_abs / z_abs / p
    out["noV"]["SCR"] = out["Z"]["SCR"] = S_BASE_MVA / z_abs / p

    p = pd.Series(p_mw["WSCR"])[list(SITE_IDS)]
    z_abs = _by_site(_wide(groups, "WSCR", "z_ii", "complex"), "WSCR").abs()
    v_abs = _by_site(_wide(groups, "WSCR", "scmva"), "WSCR") * z_abs / S_BASE_MVA

    def wscr(v):
        return (S_BASE_MVA * v / z_abs * p).sum(axis=1).to_frame(GROUP) / p.sum() ** 2
    out["m"]["WSCR"] = out["noC"]["WSCR"] = wscr(v_abs)
    out["noV"]["WSCR"] = out["Z"]["WSCR"] = wscr(1.0)

    p = p_mw["ESCR"]
    v_abs = _by_site(_wide(groups, "ESCR", "v_pf", "complex"), "ESCR").abs()
    z_abs = _by_site(_wide(groups, "ESCR", "z_ii", "complex"), "ESCR").abs()
    interaction = _wide(groups, "ESCR", "if_magnitude", pair=True)  # columns (j, i): IF_ji
    bus = EVAL_BUS["ESCR"]
    den = pd.DataFrame({i: p[i] + sum(interaction[(bus[j], bus[i])] * p[j]
                                      for j in SITE_IDS if j != i) for i in SITE_IDS})
    p_own = pd.Series(p)[list(SITE_IDS)]
    out["m"]["ESCR"] = S_BASE_MVA * v_abs / z_abs / den
    out["noV"]["ESCR"] = S_BASE_MVA / z_abs / den
    out["noC"]["ESCR"] = S_BASE_MVA * v_abs / z_abs / p_own
    out["Z"]["ESCR"] = S_BASE_MVA / z_abs / p_own

    p = {s: mw / S_BASE_MVA for s, mw in p_mw["SDSCR"].items()}
    bus = EVAL_BUS["SDSCR"]
    v_abs = _by_site(_wide(groups, "SDSCR", "v_pf", "complex"), "SDSCR").abs()
    z_rr = _wide(groups, "SDSCR", "z_rr", "complex", pair=True)
    w = _wide(groups, "SDSCR", "w_complex", "complex", pair=True)  # columns (i, j): w_ij
    z_abs = pd.DataFrame({i: z_rr[(bus[i], bus[i])].abs() for i in SITE_IDS})
    eff = pd.DataFrame({i: (p[i] + sum(w[(bus[i], bus[j])] * p[j]
                                       for j in SITE_IDS if j != i)).abs() for i in SITE_IDS})
    p_own = pd.Series(p)[list(SITE_IDS)]
    out["m"]["SDSCR"] = v_abs ** 2 / (z_abs * eff)
    out["noV"]["SDSCR"] = 1 / (z_abs * eff)
    out["noC"]["SDSCR"] = v_abs ** 2 / (z_abs * p_own)
    out["Z"]["SDSCR"] = 1 / (z_abs * p_own)
    return out


def check_reconstruction(metrics, values):
    """The recomputed m must equal the recorded metric (guards the schema reading)."""
    rows = []
    for metric in METRICS:
        rec = metrics[metrics["metric"] == metric]
        site = (rec["bus"].map({b: s for s, b in EVAL_BUS[metric].items()})
                if metric != "WSCR" else pd.Series(GROUP, index=rec.index))
        recorded = pd.Series(rec["value"].to_numpy(float),
                             index=pd.MultiIndex.from_arrays([rec["case"], rec["hour_index"], site]))
        computed = values["m"][metric].stack()
        computed.index = computed.index.set_names(recorded.index.names)
        rel = (computed.reindex(recorded.index) / recorded - 1).abs()
        if rel.isna().any():
            raise ValueError(f"{metric}: recorded and recomputed rows do not match")
        rows.append({"metric": metric, "max_rel_diff": float(rel.max())})
    out = pd.DataFrame(rows)
    if (out["max_rel_diff"] > RECONSTRUCTION_TOL).any():
        raise ValueError(f"Recomputed metrics differ from the recorded values:\n{out}")
    return out


def series(values, variant, metric, site, case):
    col = GROUP if metric == "WSCR" else site
    return values[variant][metric][col].xs(case, level="case")


def coupling_load(values, metric, site, case):
    """Interaction load relative to the site's own power, over hours:
    ESCR  k = Σ IF_ji P_j / P_i;   SDSCR  k = |P_i + Σ w_ij P_j| / P_i - 1;   SCR, WSCR  0.
    (m_Z / m_noV = denominator / P_i, so k = m_Z / m_noV - 1.)"""
    return series(values, "Z", metric, site, case) / series(values, "noV", metric, site, case) - 1


def split(values, metric, site, frm, to):
    """Δln m from state frm to state to, split into |V|, coupling and Z_ii parts.
    A state is (case, hour_index)."""
    col = GROUP if metric == "WSCR" else site
    d = {v: float(np.log(values[v][metric].loc[to, col] / values[v][metric].loc[frm, col]))
         for v in ("m", "noV", "Z")}
    return {"dln_total": d["m"], "dln_V": d["m"] - d["noV"],
            "dln_coupling": d["noV"] - d["Z"], "dln_Z_ii": d["Z"]}


# =============================================================================
# 3. Context: load scale, commitment, IF ≈ 0 states, topology-only outage Z
# =============================================================================

def lowest(s, k):
    threshold = s.sort_values(kind="mergesort").iloc[k - 1]
    return set(s.index[s <= threshold + REL_TIE * abs(threshold)])


def t_star(s):
    return sorted(lowest(s, 1))


def sign0(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= REL_TIE, 0.0, np.sign(x))


def build_context(data, hours):
    s = data["summary"].set_index(["case", "hour_index"]).sort_index()
    intact = s.loc[INTACT]
    b_count, n_committed = {}, {}
    for network, column in N_COMMITTED_COLUMN.items():
        n = intact[column].astype(int)
        n_committed[network] = n
        b_count[network] = set(n.index[n == n.min()])
    if_max = _wide(data["variables"], "ESCR", "if_magnitude", pair=True).max(axis=1)
    return {"hours": hours, "load_scale": intact["load_scale"], "n_committed": n_committed,
            "b_count": b_count, "if_zero": if_max < IF_ZERO}


def branch_update(contingencies, corridors):
    """{case: (bus_a, bus_b, C)} with C = minus the line's π stamp (2x2)."""
    limits = read_gen_limits(GEN_LIMITS_FILE)
    with contextlib.redirect_stdout(io.StringIO()):
        net = prepare_network(1.0, sequential_dispatch(0.0, limits), limits, False)
    out = {}
    for row in contingencies[contingencies["case"].isin(corridors)].itertuples():
        line = net.lines[row.line]
        y, ys = 1 / line.Z_series, line.Y_shunt
        out[row.case] = (line.bus_1, line.bus_2,
                         -np.array([[y + ys / 2, -y], [-y, y + ys / 2]], dtype=complex))
    return out


def topology_z_ii(z, network, eval_buses, bus_a, bus_b, c):
    """|Z'_ii| (hours x buses) after the branch update, from the intact Z of every hour."""
    buses = [int(b) for b in z[f"{network}_buses"]]
    order = np.argsort(z["hour_index"])
    zz = z[f"z_{network}"][order]
    u = [buses.index(bus_a), buses.index(bus_b)]
    r = [buses.index(b) for b in eval_buses]
    z_uu = zz[:, u][:, :, u]
    z_ru = zz[:, r][:, :, u]
    z_ur = zz[:, u][:, :, r]
    m = np.eye(2) + c @ z_uu
    dz = z_ru @ np.linalg.solve(m, c @ z_ur)
    z_rr = zz[:, r][:, :, r]
    return np.abs(np.diagonal(z_rr - dz, axis1=1, axis2=2)), np.abs(np.diagonal(z_rr, axis1=1, axis2=2))


def series_only(c):
    """The update for the line's series branch alone (charging shunts kept in service)."""
    y = c[0, 1]  # C = minus the π stamp, whose off-diagonal is -y
    return -np.array([[y, -y], [-y, y]], dtype=complex)


def topology_parts(data, corridors, ctx):
    """{(case, metric, site): DataFrame over hours} of the topology-only Δln m_Z:
    'full' (series branch and charging shunts removed) and 'series' (series branch only).
    The shunt part is full - series: removing a capacitive shunt lowers |Z_ii|."""
    updates = branch_update(data["contingencies"], corridors)
    hours = ctx["hours"]
    output = pd.Series(P_MW["output"])[list(SITE_IDS)].to_numpy()
    out = {}
    for case, (a, b, c) in updates.items():
        for metric in METRICS:
            buses = [EVAL_BUS[metric][s] for s in SITE_IDS]
            parts = {}
            for name, update in (("full", c), ("series", series_only(c))):
                z_out, z_int = topology_z_ii(data["z"], NETWORK[metric], buses, a, b, update)
                if metric == "WSCR":
                    ratio = (output / z_out).sum(axis=1) / (output / z_int).sum(axis=1)
                    parts[name] = {GROUP: np.log(ratio)}
                else:
                    parts[name] = {s: np.log(z_int[:, k] / z_out[:, k]) for k, s in enumerate(SITE_IDS)}
            for site in parts["full"]:
                out[case, metric, site] = pd.DataFrame(
                    {name: parts[name][site] for name in parts}, index=hours)
    return out


# =============================================================================
# 4. Claims
# =============================================================================

def _explain(in_b):
    """Why the weakest hour is (or is not) a minimum-commitment hour."""
    if not in_b["Z"]:
        return "commitment not confirmed (Z_ii-only weakest hour outside B_count)"
    if in_b["m"]:
        return "commitment"
    if in_b["noV"] and in_b["noC"]:
        return "|V| or coupling"
    if in_b["noV"]:
        return "|V|"
    if in_b["noC"]:
        return "coupling"
    return "|V| and coupling"


def d1_weakest_hour(values, ctx, cases):
    rows = []
    lam = ctx["load_scale"]
    for case in cases:
        for metric, site in UNITS:
            b = ctx["b_count"][NETWORK[metric]]
            t = {v: t_star(series(values, v, metric, site, case)) for v in VARIANTS}
            in_b = {v: bool(set(t[v]) & b) for v in VARIANTS}
            rows.append({
                "case": case, "metric": metric, "site": site, "network": NETWORK[metric],
                "B_count": _join(sorted(b)),
                "t_star": _join(t["m"]), "t_star_in_B_count": in_b["m"],
                "t_star_V_1pu": _join(t["noV"]), "t_star_coupling_removed": _join(t["noC"]),
                "t_star_Z_ii_only": _join(t["Z"]), "Z_ii_only_t_star_in_B_count": in_b["Z"],
                "t_star_equals_Z_ii_only": t["m"] == t["Z"],
                "explanation": _explain(in_b),
                "load_scale_at_t_star": float(lam.loc[t["m"][0]]),
                # from the Z_ii-only weakest hour to the actual weakest hour
                **split(values, metric, site, (case, t["Z"][0]), (case, t["m"][0])),
            })
    return pd.DataFrame(rows)


def d4_outage(values, ctx, corridors, topology):
    rows = []
    if_zero = ctx["if_zero"]
    for case in corridors:
        for metric, site in UNITS:
            ln = {v: np.log(series(values, v, metric, site, case)
                            / series(values, v, metric, site, INTACT)) for v in VARIANTS}
            part_v, part_c, part_z = ln["m"] - ln["noV"], ln["noV"] - ln["Z"], ln["Z"]
            topo = topology[case, metric, site]["full"]
            topo_series = topology[case, metric, site]["series"]
            # stronger: metric_contingency > metric_intact in that hour
            stronger = sign0(ln["m"]) > 0
            if_out = if_zero.xs(case, level="case").reindex(ln["m"].index).to_numpy()
            k_int, k_out = coupling_load(values, metric, site, INTACT), coupling_load(values, metric, site, case)
            r = np.expm1(ln["m"])
            rows.append({
                "case": case, "metric": metric, "site": site,
                "median_r": float(r.median()),
                "share_hours_stronger": float(stronger.mean()),
                "share_hours_stronger_V_1pu": float((sign0(ln["noV"]) > 0).mean()),
                "share_hours_stronger_coupling_removed": float((sign0(ln["noC"]) > 0).mean()),
                "share_hours_stronger_Z_ii_only": float((sign0(ln["Z"]) > 0).mean()),
                "max_r_Z_ii_only": float(np.expm1(ln["Z"]).max()),
                "share_hours_stronger_topology": float((sign0(topo) > 0).mean()),
                "share_hours_stronger_topology_series": float((sign0(topo_series) > 0).mean()),
                "median_dln_total": float(ln["m"].median()),
                "median_dln_V": float(part_v.median()),
                "median_dln_coupling": float(part_c.median()),
                "median_dln_Z_ii": float(part_z.median()),
                "median_dln_Z_ii_topology": float(topo.median()),
                "median_dln_Z_ii_topology_series": float(topo_series.median()),
                "median_dln_Z_ii_topology_shunt": float((topo - topo_series).median()),
                "max_abs_Z_ii_minus_topology": float((part_z - topo).abs().max()),
                "median_coupling_load_intact": float(k_int.median()),
                "median_coupling_load_outage": float(k_out.median()),
                "n_hours_stronger": int(stronger.sum()),
                "median_r_stronger": float(r[stronger].median()) if stronger.any() else float("nan"),
                "max_r_stronger": float(r[stronger].max()) if stronger.any() else float("nan"),
                "median_dln_Z_ii_stronger": float(part_z[stronger].median()) if stronger.any() else float("nan"),
                "n_stronger_coupling_load_falls": int((stronger & (sign0(k_out - k_int) < 0)).sum()),
                "n_stronger_IF_zero_outage": int((stronger & if_out).sum()),
            })
    return pd.DataFrame(rows)


def d5_snapshots(values, ctx, cases):
    lam = ctx["load_scale"]
    peak, light = int(lam.idxmax()), int(lam.idxmin())
    rows = []
    for case in cases:
        for metric, site in UNITS:
            s = series(values, "m", metric, site, case)
            t = t_star(s)
            snap = peak if s.loc[peak] <= s.loc[light] else light
            rows.append({
                "case": case, "metric": metric, "site": site, "t_peak": peak, "t_light": light,
                "t_light_in_B_count": light in ctx["b_count"][NETWORK[metric]],
                "t_snapshot": snap, "t_star": _join(t), "snapshot_is_weekly_min": snap in t,
                "snapshot_value": float(s.loc[snap]), "weekly_min_value": float(s.min()),
                "g": float(s.loc[snap] / s.min() - 1),
                # from the weekly minimum to the snapshot
                **split(values, metric, site, (case, t[0]), (case, snap)),
            })
    return pd.DataFrame(rows)


def d5_dispatch(ctx):
    """Sequential dispatch is monotone in λ: committed count never falls as λ rises."""
    lam = ctx["load_scale"].sort_values(kind="mergesort")
    rows = []
    for network, n in ctx["n_committed"].items():
        ordered = n.loc[lam.index].to_numpy()
        rows.append({"network": network, "n_hours": len(ordered),
                     "n_decreases_as_load_rises": int((np.diff(ordered) < 0).sum()),
                     "min_n_committed": int(n.min()),
                     "B_count": _join(sorted(ctx["b_count"][network]))})
    return pd.DataFrame(rows)


def d4_hourly(values, corridors):
    """r = (metric_contingency - metric_intact) / metric_intact, per hour, corridor and site."""
    frames = []
    for case in corridors:
        for metric, site in UNITS:
            r = (series(values, "m", metric, site, case) / series(values, "m", metric, site, INTACT) - 1)
            frames.append(pd.DataFrame({"case": case, "metric": metric, "site": site,
                                        "hour_index": r.index, "r": r.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def core_tables(values, ctx, cases, corridors, topology):
    return {
        "d1_weakest_hour": d1_weakest_hour(values, ctx, cases),
        "d4_outage": d4_outage(values, ctx, corridors, topology),
        "d4_hourly": d4_hourly(values, corridors),
        "d5_snapshots": d5_snapshots(values, ctx, cases),
    }


def compare_tables(reference, other, rtol=1e-9):
    """A3: every column without 'value' in its name must be unchanged."""
    rows = []
    for name, ref in reference.items():
        alt = other[name]
        cols = [c for c in ref.columns if "value" not in c]
        differing = []
        for c in cols:
            x, y = ref[c].reset_index(drop=True), alt[c].reset_index(drop=True)
            if pd.api.types.is_float_dtype(x):
                same = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=rtol, atol=1e-12,
                                   equal_nan=True)
            else:
                same = x.astype(str).equals(y.astype(str))
            if not same:
                differing.append(c)
        rows.append({"table": name, "n_rows": len(ref), "n_columns_compared": len(cols),
                     "unchanged": not differing, "differing_columns": _join(differing)})
    return pd.DataFrame(rows)


def infer_if_step(groups):
    """The ESCR probe's relative MV voltage step, from the recorded target voltages."""
    target = _wide(groups, "ESCR", "target_voltage_pu")
    v = _wide(groups, "ESCR", "v_pf", "complex").abs()
    return round(float(np.median((target / v - 1).to_numpy())), 6)


def if_step_robustness(runs, cases, corridors):
    """Gate 3 limitation 2: ESCR conclusions under other IF probe steps.
    runs: [(step, values)], the first is the results of record."""
    rows = []
    for step, values in runs:
        for case in cases:
            for site in SITE_IDS:
                row = {"if_step": step, "case": case, "site": site,
                       "t_star": _join(t_star(series(values, "m", "ESCR", site, case)))}
                if case != INTACT:
                    ln = np.log(series(values, "m", "ESCR", site, case)
                                / series(values, "m", "ESCR", site, INTACT))
                    row.update({"share_hours_stronger": float((sign0(ln) > 0).mean()),
                                "median_r": float(np.expm1(ln).median())})
                rows.append(row)
    out = pd.DataFrame(rows)
    ref = runs[0][1]
    unaffected = max(float((values["m"][m] / ref["m"][m] - 1).abs().max().max())
                     for _, values in runs[1:] for m in ("SCR", "WSCR", "SDSCR")) if len(runs) > 1 else 0.0
    return out, unaffected


def parallel_circuits(values, representative):
    rows = []
    for case, rep in representative.items():
        if case == rep:
            continue
        for metric, site in UNITS:
            a = series(values, "m", metric, site, case)
            b = series(values, "m", metric, site, rep)
            rows.append({"case": case, "representative": rep, "metric": metric, "site": site,
                         "same_t_star": t_star(a) == t_star(b),
                         "max_abs_dln": float(np.log(a / b).abs().max())})
    return pd.DataFrame(rows)


# =============================================================================
# 5. Summary and driver
# =============================================================================

def _count(mask):
    mask = pd.Series(mask)
    return f"{int(mask.sum())}/{len(mask)}"


def build_summary(data, ctx, out, cases, corridors):
    rows = []

    def add(claim, item, result, table=""):
        rows.append({"claim": claim, "item": item, "result": result, "table": table})

    add("inputs", "hours per scenario", len(ctx["hours"]))
    add("inputs", "scenarios (intact + one per corridor)", len(cases))
    for f, h in data["sha256"].items():
        add("inputs", f"SHA-256 {f}", h)
    for network, b in ctx["b_count"].items():
        add("inputs", f"minimum-commitment hours B_count [{network}]", _join(sorted(b)))

    d1 = out["d1_weakest_hour"]
    c = "D1/RQ2 weakest hour is a minimum-commitment hour"
    add(c, "mechanism: Z_ii-only weakest hour in B_count", _count(d1["Z_ii_only_t_star_in_B_count"]),
        "d1_weakest_hour")
    add(c, "observation: weakest hour in B_count", _count(d1["t_star_in_B_count"]), "d1_weakest_hour")
    add(c, "weakest hour equals the Z_ii-only weakest hour", _count(d1["t_star_equals_Z_ii_only"]),
        "d1_weakest_hour")
    for explanation, n in d1["explanation"].value_counts().items():
        add(c, f"explanation: {explanation}", int(n), "d1_weakest_hour")
    outside = d1[~d1["t_star_in_B_count"]]
    add(c, "exceptions (case/metric/site: explanation)", _join(
        f"{r.case}/{r.metric}/{r.site}: {r.explanation}" for r in outside.itertuples()) or "none",
        "d1_weakest_hour")

    d4 = out["d4_outage"]

    def pct(x):
        return f"{100 * float(x):.1f} %"

    # every corridor-site has the same number of hours, so the mean of shares is the pooled share
    c = "D4 in network terms an outage weakens a site or leaves it practically unaffected"
    add(c, "Z_ii part matches the topology-only ΔZ_ii: max |difference| in Δln",
        float(d4["max_abs_Z_ii_minus_topology"].max()), "d4_outage")
    add(c, "network (Z_ii only): largest change toward stronger",
        pct(d4["max_r_Z_ii_only"].max()), "d4_outage")
    add(c, "topology alone: hours with contingency > intact",
        pct(d4["share_hours_stronger_topology"].mean()), "d4_outage")
    add(c, "  series branch alone: hours with contingency > intact",
        pct(d4["share_hours_stronger_topology_series"].mean()), "d4_outage")
    add(c, "  charging-shunt part strengthens (median Δln > 0)",
        _count(d4["median_dln_Z_ii_topology_shunt"] > 0), "d4_outage")

    c = "D4 hours with metric_contingency > metric_intact"
    for metric in METRICS:
        part = d4[d4["metric"] == metric]
        add(c, f"{metric}: as computed / |V| = 1 pu / coupling removed / Z_ii only",
            " / ".join(pct(part[col].mean()) for col in (
                "share_hours_stronger", "share_hours_stronger_V_1pu",
                "share_hours_stronger_coupling_removed", "share_hours_stronger_Z_ii_only")), "d4_outage")
        if part["n_hours_stronger"].sum():
            add(c, f"  {metric}: size r when stronger, median / largest",
                f"{part['median_r_stronger'].median():.4g} / {part['max_r_stronger'].max():.4g}", "d4_outage")

    c = "D4 coupling metrics move against the network when the outage decouples the sites"
    for metric in ("ESCR", "SDSCR"):
        part = d4[d4["metric"] == metric]
        add(c, f"{metric}: stronger hours where the coupling load k falls",
            f"{int(part['n_stronger_coupling_load_falls'].sum())}/{int(part['n_hours_stronger'].sum())}",
            "d4_outage")
    escr = d4[d4["metric"] == "ESCR"]
    add(c, "ESCR: stronger hours in IF ≈ 0 outage states",
        f"{int(escr['n_stronger_IF_zero_outage'].sum())}/{int(escr['n_hours_stronger'].sum())}", "d4_outage")

    d5, dispatch = out["d5_snapshots"], out["d5_dispatch"]
    c = "D5 light-load snapshot almost finds the weekly minimum"
    for r in dispatch.itertuples():
        add(c, f"mechanism: committed count falls as load rises [{r.network}]",
            f"{r.n_decreases_as_load_rises} times", "d5_dispatch")
    light = int(ctx["load_scale"].idxmin())
    for network, b in ctx["b_count"].items():
        add(c, f"light-load hour {light} in B_count [{network}]", light in b, "d5_snapshots")
    add(c, "snapshot is the weekly minimum", _count(d5["snapshot_is_weekly_min"]), "d5_snapshots")
    worst = d5.loc[d5["g"].idxmax()]
    add(c, "largest gap g", float(worst["g"]), "d5_snapshots")
    add(c, "largest gap at", f"{worst['case']}/{worst['metric']}/{worst['site']}", "d5_snapshots")
    add(c, "largest gap: Δln from |V| / coupling / Z_ii",
        f"{worst['dln_V']:.4g} / {worst['dln_coupling']:.4g} / {worst['dln_Z_ii']:.4g}", "d5_snapshots")

    add("A3 denominators change no conclusion", "unchanged under rated, output and MVA",
        bool(out["a3_denominators"]["unchanged"].all()), "a3_denominators")
    add("A3 denominators change no conclusion", "max relative reconstruction difference",
        float(out["reconstruction_check"]["max_rel_diff"].max()), "reconstruction_check")

    if "if_step_robustness" in out:
        rob = out["if_step_robustness"]
        c = "check IF probe step (Gate 3 limitation 2)"
        ref_step = rob["if_step"].iloc[0]
        ref = rob[rob["if_step"] == ref_step].set_index(["case", "site"])
        steps = sorted(rob["if_step"].unique())
        add(c, "ESCR hours with contingency > intact, per step", " | ".join(
            f"{100 * s:g} %: {pct(rob[(rob['if_step'] == s) & (rob['case'] != INTACT)]['share_hours_stronger'].mean())}"
            for s in steps), "if_step_robustness")
        add(c, f"ESCR weakest hour unchanged from the {100 * ref_step:g} % step", " | ".join(
            f"{100 * s:g} %: "
            f"{_count(rob[rob['if_step'] == s].set_index(['case', 'site'])['t_star'] == ref['t_star'])}"
            for s in steps if s != ref_step), "if_step_robustness")
        add(c, "SCR, WSCR, SDSCR unchanged by the step (max relative difference)",
            out["if_step_unaffected"], "if_step_robustness")

    par = out["parallel_circuits"]
    if not par.empty:
        add("check parallel circuits", "same weakest hour as circuit 1", _count(par["same_t_star"]),
            "parallel_circuits")
        add("check parallel circuits", "max |Δln| against circuit 1", float(par["max_abs_dln"].max()),
            "parallel_circuits")
    return pd.DataFrame(rows)


def run(input_dir=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, if_step_runs=()):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir or output_dir in {Path(d).resolve() for d in if_step_runs}:
        raise ValueError("Output directory must differ from every input directory")
    data = load_inputs(input_dir)
    hours, _ = check_inputs(data)
    corridors, representative = scenarios(data["contingencies"])
    cases = [INTACT] + corridors
    ctx = build_context(data, hours)
    topology = topology_parts(data, corridors, ctx)

    values = variant_values(data["variables"], {m: P_MW[AS_RUN[m]] for m in METRICS})
    out = {"reconstruction_check": check_reconstruction(data["metrics"], values)}
    core = core_tables(values, ctx, cases, corridors, topology)
    out.update(core)
    out["d5_dispatch"] = d5_dispatch(ctx)
    out["a3_denominators"] = pd.concat([
        compare_tables(core, core_tables(variant_values(data["variables"], {m: P_MW[kind] for m in METRICS}),
                                         ctx, cases, corridors, topology)).assign(denominator=kind)
        for kind in ("rated", "output", "mva")
    ], ignore_index=True)
    out["parallel_circuits"] = parallel_circuits(values, representative)
    unaffected = None
    if if_step_runs:
        runs = [(infer_if_step(data["variables"]), values)]
        for d in if_step_runs:
            alt = load_inputs(d, reference=data)
            check_inputs(alt)  # same scenarios and hours as the results of record
            alt_values = variant_values(alt["variables"], {m: P_MW[AS_RUN[m]] for m in METRICS})
            check_reconstruction(alt["metrics"], alt_values)
            runs.append((infer_if_step(alt["variables"]), alt_values))
        out["if_step_robustness"], unaffected = if_step_robustness(runs, cases, corridors)
    out["summary"] = build_summary(data, ctx, {**out, "if_step_unaffected": unaffected}, cases, corridors)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--if-step-run", type=Path, action="append", default=[], dest="if_step_runs",
                        help="A rerun with another ESCR probe step (repeatable), for the robustness check.")
    args = parser.parse_args()
    out = run(args.input_dir, args.output_dir, args.if_step_runs)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 100,
                           "display.width", 200):
        print(out["summary"].drop(columns="table").to_string(index=False))
    print(f"\nSaved {len(out)} tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
