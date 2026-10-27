"""
Analyze the Glover system-strength metrics experiment (Gate 4, criteria v1.0).
Record of the v1.0 run (25 Oct 2026) on results c263896 (intact + line 39-47 only);
superseded by evidence plan v2 (handoff file 05, Part E) and analyze_strength_metrics.py.

Compares SCR, WSCR, ESCR and SDSCR under the criteria frozen on 24 Oct 2026
(current_research_handoff/05_GATE3.5_LITERATURE_EXTRACTION.md, Part A §4).
Notation follows v1.0: for metrics m and n,

    t*_m      hour of the minimum (relative ties within 1e-9)
    K_m       bottom 10 % of hours (17 of 168)
    J(A, B)   Jaccard overlap
    r_m(t)  = m_out(t) / m_int(t) - 1                                 (D4, C4)
    g       = (snapshot minimum - weekly minimum) / weekly minimum     (D5)
    Δ_m(t)  = m(t) / m(t-1) - 1,  E_m = top 5 % of |Δ_m|               (D2)
    B_count = hours at the minimum committed count;
    B_set   = B_count grouped by committed set                         (A1)

No network, power-flow or short-circuit calculation is performed in this stage.

Usage
-----
Use the default repository paths:

    python -m experiments.glover37.analyze_strength_metrics_v1

Specify paths explicitly:

    python -m experiments.glover37.analyze_strength_metrics_v1 \
        --input-dir results/glover37/strength/raw \
        --output-dir results/glover37/strength/analysis_v1

Inputs (written by run_strength_metrics.py)
------
results/glover37/strength/raw/metrics_timeseries.csv
results/glover37/strength/raw/metric_variables.csv
results/glover37/strength/raw/run_summary.csv
results/glover37/strength/raw/dispatch_schedule.csv

What answers what
-----------------
RQ1  Do the metrics reach the same screening conclusions?
     D1  Critical conditions (core): when is each site weakest?
         C1  Does each metric's weakest hour t*_m fall in the other's 17 weakest hours K_n?
         C2  How much do the 17 weakest hours overlap, J(K_m, K_n)?
         C3  Do they rank the whole week alike (Spearman ρ, Kendall τ)?
             -> d1_pairs, d1_t_star
     D4  Contingency impact (core): how does the line 39-47 outage change strength?
         C4a Do the metrics move in the same direction, hour by hour (sign of r)?
         C4b Does the outage move the weakest hour, and do the metrics agree on that?
         C4c Do they rank the sites alike by outage impact (median r)?
             -> d4_c4_metrics, d4_c4_pairs, d4_c4c_site_ranking
     D5  Planning snapshots (core): would the peak and light-load hours find the minimum?
         D5a Is the lower snapshot among the 17 weakest hours?
         D5b How far is it above the weekly minimum (g)?
             -> d5_snapshots
     D2  Significant changes (secondary): do sharp changes happen at the same hours?
             -> d2_events, d2_metrics, d2_pairs
     D3  Spatial ordering (supporting): do the metrics name the same weakest site?
             -> d3_weakest_site, d3_metrics, d3_pairs
RQ2  Does any metric say more than which generators are committed?
     A1  Do the minima fall in the minimum-commitment hours (B_count), and do the
         metrics order the hours alike within one committed set (B_set)?
             -> a1_b_set, a1_metrics, a1_within_b_set
RQ3  Where conclusions differ, is it the conventions or the formulas?
     A2  Does a disagreement survive on numerators only, or with interaction removed?
             -> a2_pairs, a2_interaction, a2_c4_metrics, a2_c4_pairs
     A3  Are the conclusions unchanged under rated, output and MVA denominators?
             -> a3_denominator_invariance, a3_reconstruction_check
     A4  Is |V| or Z_ii behind SCR site 21's outage minimum (look 2)?
             -> a4_hours, a4_summary, a4_pairs
Context: annotations (flagged states kept in the analysis).

Outputs (results/glover37/strength/analysis_v1/)
-------
summary.csv   one line per result, with its RQ, dimension, criterion and question
index.csv     one line per table, with the question it answers
<table>.csv   the tables named above

Operational details v1.0 leaves open
------------------------------------
- Set sizes are ceil(fraction * n): |K| = 17 of 168 at 10 %, 9 at 5 %, 34 at 20 %;
  |E| = 9 of the 167 changes at 5 %. Values tied (1e-9 relative) with the k-th
  value are included, so a set can be larger than k.
- Tied minima make t*_m a set; "t*_m ∈ K_n" holds if any tied minimum is in K_n.
- WSCR is one group series, compared against each site; per-metric tables list it
  once with site "group". SCR-WSCR is reported with informative = False.
- The secondary C1 check removes annotated states from every metric at once and
  recomputes K on the remaining hours.
- C3 labels use the signed coefficient.
- D2 commitment change: the committed set changes from t-1 to t in that metric's
  network ("base" 37-bus for SCR/WSCR, "ibr" 46-bus for ESCR/SDSCR).
- A3: columns whose name contains "value" are scale dependent; every other column
  must be unchanged under the rated, output and MVA denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from experiments.glover37.run_temporal import CONTINGENCY_CASE, ROOT, S_BASE_MVA
from experiments.glover37.strength_config import SITES

STRENGTH_ROOT = ROOT / "results" / "glover37" / "strength"
DEFAULT_INPUT = STRENGTH_ROOT / "raw"
DEFAULT_OUTPUT = STRENGTH_ROOT / "analysis_v1"
INPUT_FILES = ("metrics_timeseries.csv", "metric_variables.csv",
               "run_summary.csv", "dispatch_schedule.csv")

CASES = ("intact", CONTINGENCY_CASE)
METRICS = ("SCR", "WSCR", "ESCR", "SDSCR")
SITE_METRICS = ("SCR", "ESCR", "SDSCR")
PAIRS = tuple(combinations(METRICS, 2))
BY_CONSTRUCTION = {("SCR", "WSCR")}  # shared 37-bus SCMVA: reported, not counted
SITE_IDS = tuple(str(s.host_bus) for s in SITES)
GROUP = "group"
UNITS = tuple((m, s) for m in METRICS for s in ((GROUP,) if m == "WSCR" else SITE_IDS))

# evaluation bus -> site label (original 69 kV host bus)
SITE_OF = {
    "SCR": {s.poi_bus: str(s.host_bus) for s in SITES},
    "WSCR": {s.poi_bus: str(s.host_bus) for s in SITES},
    "ESCR": {s.escr_bus: str(s.host_bus) for s in SITES},
    "SDSCR": {s.sdscr_bus: str(s.host_bus) for s in SITES},
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
K_FRACTION = 0.10
K_SENSITIVITY = (0.05, 0.20)
E_FRACTION = 0.05
IF_ZERO = 3e-5  # IF ≈ 0 state: every IF < 3e-5 (Gate 3 record, file 02 §5)
ANNOTATIONS = ("IF_zero", "probe_PV_PQ_switch", "voltage_flag", "any_annotation")

RQ1 = "RQ1 consistency"
RQ2 = "RQ2 beyond commitment"
RQ3 = "RQ3 attribution"
CONTEXT = "context"
D1 = "D1 critical conditions (core)"
D4 = "D4 contingency impact (core)"
D5 = "D5 planning snapshots (core)"
D2 = "D2 significant changes (secondary)"
D3 = "D3 spatial ordering (supporting)"

# table -> (research question, dimension, criteria, what it answers)
TABLES = {
    "summary": (CONTEXT, "", "", "One line per result, traced to its RQ, dimension and criterion"),
    "index": (CONTEXT, "", "", "This list: what each table answers"),
    "annotations": (CONTEXT, "", "", "Which states are flagged (IF ≈ 0, probe switch, voltage); kept in the analysis"),
    "d1_pairs": (RQ1, D1, "C1 C2 C3", "Per metric pair and site: same weakest hour? overlap of the 17 weakest hours? same weekly ranking?"),
    "d1_t_star": (RQ1, D1, "C1", "Per metric and site: the weakest hour t*, its 17 weakest hours K, and load, commitment and flags at t*"),
    "d4_c4_metrics": (RQ1, D4, "C4a C4b", "Per metric and site: outage effect r over the week; does the outage move the weakest hour?"),
    "d4_c4_pairs": (RQ1, D4, "C4a C4b", "Per metric pair and site: same direction of outage effect each hour? same answer on the moved weakest hour?"),
    "d4_c4c_site_ranking": (RQ1, D4, "C4c", "Per pair of site-level metrics: same ranking of sites by median outage effect?"),
    "d5_snapshots": (RQ1, D5, "D5a D5b", "Per metric and site: do the peak and light-load hours find the weekly minimum, and how far off are they?"),
    "d2_events": (RQ1, D2, "", "Each sharp hour-to-hour change E_m, and whether commitment changed in that hour"),
    "d2_metrics": (RQ1, D2, "", "Per metric and site: the sharp-change hours, and the share at a commitment change"),
    "d2_pairs": (RQ1, D2, "", "Per metric pair and site: overlap of sharp-change hours and their sign agreement"),
    "d3_weakest_site": (RQ1, D3, "", "Per hour and metric: the weakest site"),
    "d3_metrics": (RQ1, D3, "", "Per metric: how often the weakest site changes and which site it is"),
    "d3_pairs": (RQ1, D3, "", "Per metric pair: share of hours naming the same weakest site"),
    "a1_b_set": (RQ2, "A1 commitment baseline", "A1", "The minimum-commitment hours B_count and their committed sets B_set"),
    "a1_metrics": (RQ2, "A1 commitment baseline", "A1", "Per metric and site: does the weakest hour fall in B_count?"),
    "a1_within_b_set": (RQ2, "A1 commitment baseline", "A1", "Within one committed set: do metric pairs pick the same weakest hour and order hours alike?"),
    "a2_pairs": (RQ3, "A2 conventions vs formulas", "A2", "Per metric pair: does C1 still agree or disagree on numerators only, or with interaction removed?"),
    "a2_interaction": (RQ3, "A2 conventions vs formulas", "A2", "ESCR and SDSCR: does removing the interaction term (IF = 0, w = 0) move the weakest hour?"),
    "a2_c4_metrics": (RQ3, "A2 conventions vs formulas", "A2 C4a", "Per metric and site: share of outage-weakened hours on numerators only and with interaction removed; do rising hours coincide with IF ≈ 0 states?"),
    "a2_c4_pairs": (RQ3, "A2 conventions vs formulas", "A2 C4a", "Per metric pair: same direction of outage effect each hour, original vs numerators only vs interaction removed"),
    "a3_denominator_invariance": (RQ3, "A3 denominators", "A3", "Are the D1-D5 and A1 results unchanged under rated, output and MVA denominators?"),
    "a3_reconstruction_check": (RQ3, "A3 denominators", "A3", "Do the metrics recompute from their recorded terms?"),
    "a4_hours": (RQ3, "A4 look-2 hypothesis", "A4", "|V|, Z_ii, voltage flags, Q limits and taps at the hours that matter for SCR site 21"),
    "a4_summary": (RQ3, "A4 look-2 hypothesis", "A4", "Does |V| move SCR site 21's outage weakest hour, and by how much?"),
    "a4_pairs": (RQ3, "A4 look-2 hypothesis", "A4", "Does SCR site 21 agree with the others at the same site once |V| is held at 1 pu?"),
}


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


def load_inputs(input_dir):
    """The run's files, restricted to intact + line 39-47: the results of record v2
    (N-1 set) contain the v1.0 scenarios unchanged, alongside the other outages."""
    d = Path(input_dir)
    if missing := [f for f in INPUT_FILES if not (d / f).is_file()]:
        raise FileNotFoundError(f"Missing inputs in {d}: {missing}")

    def v1_cases(frame):
        return frame[frame["case"].isin(CASES)].reset_index(drop=True)

    metrics = v1_cases(pd.read_csv(d / "metrics_timeseries.csv", dtype={"bus": str}))
    metrics["bus"] = metrics["bus"].map(_bus_id)
    variables = v1_cases(pd.read_csv(d / "metric_variables.csv",
                                     dtype={"bus": str, "related_bus": str}))
    variables["bus"] = variables["bus"].map(_bus_id)
    variables["related_bus"] = variables["related_bus"].map(_bus_id)
    return {
        "metrics": metrics,
        "variables": variables,
        "summary": v1_cases(pd.read_csv(d / "run_summary.csv")),
        "dispatch": v1_cases(pd.read_csv(d / "dispatch_schedule.csv")),
        "sha256": {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in INPUT_FILES},
    }


def check_inputs(data):
    """Fail on anything incomplete; never treat a missing state as valid."""
    metrics, summary = data["metrics"], data["summary"]
    if set(summary["case"]) != set(CASES):
        raise ValueError(f"Expected cases {CASES}, found {sorted(set(summary['case']))}")
    hours = sorted(summary["hour_index"].unique())
    for case in CASES:
        s = summary[summary["case"] == case]
        if sorted(s["hour_index"]) != hours:
            raise ValueError(f"{case}: hours differ from the other case or repeat")
        if not (s["base_pf_ok"].astype(bool).all() and s["ibr_pf_ok"].astype(bool).all()):
            raise ValueError(f"{case}: unconverged power flow in run_summary.csv")
        if s["error"].fillna("").astype(str).str.strip().ne("").any():
            raise ValueError(f"{case}: errors recorded in run_summary.csv")
    if set(metrics["metric"]) != set(METRICS):
        raise ValueError(f"Unexpected metrics {sorted(set(metrics['metric']))}")
    expected = len(hours) * len(CASES) * len(UNITS)
    if len(metrics) != expected:
        raise ValueError(f"metrics_timeseries.csv has {len(metrics)} rows, expected {expected}")
    v = metrics["value"].to_numpy(dtype=float)
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Non-finite or non-positive metric values")
    return np.asarray(hours, dtype=int)


def site_values(metrics):
    """Long table: case, hour_index, metric, site, value."""
    sites = []
    for metric, bus in zip(metrics["metric"], metrics["bus"]):
        if metric == "WSCR":
            sites.append(GROUP)
        elif bus in SITE_OF[metric]:
            sites.append(SITE_OF[metric][bus])
        else:
            raise ValueError(f"{metric}: unexpected evaluation bus {bus}")
    out = metrics[["case", "hour_index", "metric"]].copy()
    out["site"] = sites
    out["value"] = metrics["value"].astype(float)
    return out


def to_tables(values):
    """{case: DataFrame(index hour_index, columns (metric, site))}."""
    tables = {}
    for case in CASES:
        wide = values[values["case"] == case].pivot(
            index="hour_index", columns=["metric", "site"], values="value").sort_index()
        if wide.isna().any().any():
            raise ValueError(f"{case}: incomplete metric series")
        tables[case] = wide
    return tables


def series(tables, case, metric, site):
    return tables[case][(metric, GROUP if metric == "WSCR" else site)]


def _unit(metric, site):
    return metric, GROUP if metric == "WSCR" else site


# =============================================================================
# 2. State context: load scale λ(t), commitment, annotations
# =============================================================================

def _wide(variables, metric, variable, part="real", pair=False):
    rows = variables[(variables["metric"] == metric) & (variables["variable"] == variable)]
    if rows.empty:
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


def build_context(data, hours):
    summary, dispatch, variables = data["summary"], data["dispatch"], data["variables"]
    s = summary.set_index(["case", "hour_index"]).sort_index()
    load_scale = s.loc["intact", "load_scale"]
    if not np.allclose(load_scale, s.loc[CONTINGENCY_CASE, "load_scale"]):
        raise ValueError("load_scale differs between cases")

    n_committed, committed_set = {}, {}
    committed = dispatch[dispatch["committed"].astype(int) == 1]
    for case in CASES:
        for network, column in N_COMMITTED_COLUMN.items():
            n_committed[case, network] = s.loc[case, column].astype(int)
            rows = committed[(committed["case"] == case) & (committed["network"] == network)]
            sets = rows.groupby("hour_index")["generator"].agg(frozenset)
            committed_set[case, network] = pd.Series(
                [sets.get(h, frozenset()) for h in hours], index=hours)
            sizes = committed_set[case, network].map(len).to_numpy()
            if not (sizes == n_committed[case, network].loc[hours].to_numpy()).all():
                raise ValueError(f"{case}/{network}: dispatch_schedule and run_summary "
                                 "commitment counts differ")

    if_max = _wide(variables, "ESCR", "if_magnitude", pair=True).max(axis=1)
    pv_changes = _wide(variables, "ESCR", "probe_pv_changes").max(axis=1)
    ann = pd.DataFrame(index=s.index)
    ann["IF_zero"] = if_max.reindex(s.index) < IF_ZERO
    ann["probe_PV_PQ_switch"] = pv_changes.reindex(s.index) > 0
    ann["voltage_flag"] = s["base_voltage_flag"].astype(bool) | s["ibr_voltage_flag"].astype(bool)
    for col in ("base_n_under_v", "base_n_over_v", "ibr_n_under_v", "ibr_n_over_v"):
        ann[col.replace("_n_", "_")] = s[col] > 0
    ann["any_annotation"] = ann[["IF_zero", "probe_PV_PQ_switch", "voltage_flag"]].any(axis=1)
    return {"hours": hours, "load_scale": load_scale, "n_committed": n_committed,
            "committed_set": committed_set, "annotations": ann, "summary": s}


# =============================================================================
# 3. Set and rank helpers
# =============================================================================

def set_size(fraction, n):
    return math.ceil(round(fraction * n, 9))


def lowest(s, k):
    """Index labels of the k lowest values; values tied with the k-th are included."""
    threshold = s.sort_values(kind="mergesort").iloc[k - 1]
    return set(s.index[s <= threshold + REL_TIE * abs(threshold)])


def t_star(s):
    return sorted(lowest(s, 1))


def critical_set(s, fraction=K_FRACTION):
    return lowest(s, set_size(fraction, len(s)))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def rank_label(x):
    if not np.isfinite(x):
        return ""
    return "strong" if x >= 0.9 else "moderate" if x >= 0.7 else "weak"


def sign0(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= REL_TIE, 0.0, np.sign(x))


def c1_c3(m, n, fraction=K_FRACTION):
    """C1 (t* agreement), C2 (J of K) and C3 (Spearman ρ, Kendall τ) for one pair."""
    tm, tn = t_star(m), t_star(n)
    km, kn = critical_set(m, fraction), critical_set(n, fraction)
    m_in_kn, n_in_km = bool(set(tm) & kn), bool(set(tn) & km)
    rho = spearmanr(m.to_numpy(), n.to_numpy()).statistic
    tau = kendalltau(m.to_numpy(), n.to_numpy()).statistic
    return {
        "n_hours": len(m), "k": set_size(fraction, len(m)),
        "t_star_m": _join(tm), "t_star_n": _join(tn),
        "size_K_m": len(km), "size_K_n": len(kn),
        "t_star_m_in_K_n": m_in_kn, "t_star_n_in_K_m": n_in_km,
        "C1_agree": m_in_kn and n_in_km,
        "abs_t_star_diff_h": min(abs(x - y) for x in tm for y in tn),
        "C2_J_K": jaccard(km, kn),
        "C3_spearman_rho": rho, "C3_rho_label": rank_label(rho),
        "C3_kendall_tau": tau, "C3_tau_label": rank_label(tau),
    }


# =============================================================================
# 4. D1-D5 and A1 (core tables, recomputed for A3)
# =============================================================================

def d1_pairs(tables, ctx):
    analyses = [("primary", K_FRACTION, None)]
    analyses += [(f"K_{round(f * 100)}pct", f, None) for f in K_SENSITIVITY]
    analyses += [(f"without_{a}", K_FRACTION, a) for a in ANNOTATIONS]
    rows = []
    for case in CASES:
        ann = ctx["annotations"].loc[case]
        for analysis, fraction, exclusion in analyses:
            keep = ctx["hours"] if exclusion is None else ann.index[~ann[exclusion]]
            for site in SITE_IDS:
                for m, n in PAIRS:
                    rows.append({
                        "case": case, "analysis": analysis, "K_fraction": fraction,
                        "site": site, "metric_m": m, "metric_n": n,
                        "informative": (m, n) not in BY_CONSTRUCTION,
                        **c1_c3(series(tables, case, m, site).loc[keep],
                                series(tables, case, n, site).loc[keep], fraction),
                    })
    return pd.DataFrame(rows)


def d1_t_star(tables, ctx):
    """Per metric and site: t*_m, K_m, drivers (λ, commitment) and annotations at t*_m."""
    lam = ctx["load_scale"]
    ann_cols = ("IF_zero", "probe_PV_PQ_switch", "voltage_flag")
    rows = []
    for case in CASES:
        ann = ctx["annotations"].loc[case]
        for metric, site in UNITS:
            s = series(tables, case, metric, site)
            ts, k = t_star(s), sorted(critical_set(s))
            t0 = ts[0]
            network = NETWORK[metric]
            n = ctx["n_committed"][case, network]
            rows.append({
                "case": case, "metric": metric, "site": site,
                "t_star": _join(ts), "n_tied_t_star": len(ts), "min_value": float(s.min()),
                "K": _join(k), "size_K": len(k),
                "load_scale_at_t_star": float(lam.loc[t0]),
                "load_scale_percentile_at_t_star": float((lam < lam.loc[t0]).mean()),
                "network": network, "n_committed_at_t_star": int(n.loc[t0]),
                "min_n_committed": int(n.min()),
                **{f"t_star_{a}": bool(ann.loc[t0, a]) for a in ann_cols},
                **{f"share_K_{a}": float(ann.loc[k, a].mean()) for a in ann_cols},
            })
    return pd.DataFrame(rows)


def d4_c4(tables):
    """C4 on r_m(t) = m_out(t) / m_int(t) - 1."""
    r, metric_rows = {}, []
    for metric, site in UNITS:
        s_int = series(tables, "intact", metric, site)
        s_out = series(tables, CONTINGENCY_CASE, metric, site)
        r[metric, site] = s_out / s_int - 1
        t_int, t_out = t_star(s_int), t_star(s_out)
        metric_rows.append({
            "metric": metric, "site": site,
            "C4a_median_r": float(r[metric, site].median()),
            "min_r": float(r[metric, site].min()), "max_r": float(r[metric, site].max()),
            "share_r_negative": float((sign0(r[metric, site]) < 0).mean()),
            "share_r_zero": float((sign0(r[metric, site]) == 0).mean()),
            "t_star_int": _join(t_int), "t_star_out": _join(t_out),
            "C4b_t_star_out_not_in_K_int": not (set(t_out) & critical_set(s_int)),
            "abs_t_star_diff_h": min(abs(x - y) for x in t_int for y in t_out),
        })
    metric_rows = pd.DataFrame(metric_rows)
    moves = metric_rows.set_index(["metric", "site"])["C4b_t_star_out_not_in_K_int"]

    pair_rows = []
    for site in SITE_IDS:
        for m, n in PAIRS:
            um, un = _unit(m, site), _unit(n, site)
            pair_rows.append({
                "site": site, "metric_m": m, "metric_n": n,
                "informative": (m, n) not in BY_CONSTRUCTION,
                "C4a_share_same_sign_r": float((sign0(r[um]) == sign0(r[un])).mean()),
                "C4b_moves_m": bool(moves[um]), "C4b_moves_n": bool(moves[un]),
                "C4b_agree": bool(moves[um] == moves[un]),
            })

    median = metric_rows.set_index(["metric", "site"])["C4a_median_r"]
    order = {m: sorted(SITE_IDS, key=lambda s: (median[m, s], s)) for m in SITE_METRICS}
    rank_rows = []
    for m, n in combinations(SITE_METRICS, 2):
        tau = kendalltau([median[m, s] for s in SITE_IDS],
                         [median[n, s] for s in SITE_IDS]).statistic
        rank_rows.append({"metric_m": m, "metric_n": n,
                          "site_order_m": _join(order[m]), "site_order_n": _join(order[n]),
                          "C4c_agree": order[m] == order[n],
                          "kendall_tau_median_r": tau})
    return metric_rows, pd.DataFrame(pair_rows), pd.DataFrame(rank_rows)


def d5_snapshots(tables, ctx):
    """Planning snapshots: hours of maximum (peak) and minimum (light) λ(t)."""
    lam = ctx["load_scale"]
    peak, light = int(lam.idxmax()), int(lam.idxmin())
    rows = []
    for case in CASES:
        for metric, site in UNITS:
            s = series(tables, case, metric, site)
            weekly_min = float(s.min())
            t_snap = peak if s.loc[peak] <= s.loc[light] else light
            snapshot_min = float(s.loc[t_snap])
            rows.append({
                "case": case, "metric": metric, "site": site,
                "t_peak": peak, "t_light": light,
                "n_tied_peak": int((lam == lam.max()).sum()),
                "n_tied_light": int((lam == lam.min()).sum()),
                "value_peak": float(s.loc[peak]), "value_light": float(s.loc[light]),
                "t_snapshot_min": t_snap, "snapshot_min_value": snapshot_min,
                "weekly_min_value": weekly_min, "t_star": _join(t_star(s)),
                "D5a_snapshot_min_in_K": t_snap in critical_set(s),
                "D5b_g": (snapshot_min - weekly_min) / weekly_min,
            })
    return pd.DataFrame(rows)


def d2_changes(tables, ctx):
    """Δ_m(t), event sets E_m and their coincidence with commitment changes."""
    hours = ctx["hours"]
    change = {}
    for (case, network), sets in ctx["committed_set"].items():
        prev = sets.shift(1)
        n = ctx["n_committed"][case, network].loc[hours]
        change[case, network] = pd.DataFrame({
            "set_change": [h != hours[0] and sets.loc[h] != prev.loc[h] for h in hours],
            "count_change": (n != n.shift(1)).to_numpy() & (hours != hours[0]),
        }, index=hours)

    events, metric_rows, pair_rows = [], [], []
    for case in CASES:
        e_sets, deltas = {}, {}
        for metric, site in UNITS:
            s = series(tables, case, metric, site)
            delta = (s / s.shift(1) - 1).iloc[1:]
            k = set_size(E_FRACTION, len(delta))
            e = sorted(lowest(-delta.abs(), k))
            e_sets[metric, site], deltas[metric, site] = set(e), delta
            ch = change[case, NETWORK[metric]]
            for h in e:
                events.append({"case": case, "metric": metric, "site": site, "hour_index": h,
                               "Delta": float(delta.loc[h]),
                               "commitment_set_change": bool(ch.loc[h, "set_change"]),
                               "commitment_count_change": bool(ch.loc[h, "count_change"])})
            metric_rows.append({"case": case, "metric": metric, "site": site,
                                "n_Delta": len(delta), "k": k, "size_E": len(e), "E": _join(e),
                                "share_E_commitment_set_change":
                                    float(ch.loc[e, "set_change"].mean()),
                                "share_E_commitment_count_change":
                                    float(ch.loc[e, "count_change"].mean())})
        for site in SITE_IDS:
            for m, n in PAIRS:
                um, un = _unit(m, site), _unit(n, site)
                shared = sorted(e_sets[um] & e_sets[un])
                same = [np.sign(deltas[um].loc[h]) == np.sign(deltas[un].loc[h]) for h in shared]
                pair_rows.append({"case": case, "site": site, "metric_m": m, "metric_n": n,
                                  "informative": (m, n) not in BY_CONSTRUCTION,
                                  "J_E": jaccard(e_sets[um], e_sets[un]),
                                  "n_shared_E": len(shared), "shared_E": _join(shared),
                                  "share_shared_E_same_sign":
                                      float(np.mean(same)) if same else float("nan")})
    return pd.DataFrame(events), pd.DataFrame(metric_rows), pd.DataFrame(pair_rows)


def d3_spatial(tables):
    """Weakest site per hour by SCR, ESCR and SDSCR (WSCR excluded)."""
    hourly, metric_rows, pair_rows = [], [], []
    for case in CASES:
        weakest = {}
        for metric in SITE_METRICS:
            frame = tables[case][metric][list(SITE_IDS)]
            low = frame.min(axis=1)
            tied = frame.le(low + REL_TIE * low.abs(), axis=0).sum(axis=1) > 1
            w = frame.idxmin(axis=1)
            weakest[metric] = w
            metric_rows.append({
                "case": case, "metric": metric,
                "n_weakest_site_changes": int((w != w.shift(1)).iloc[1:].sum()),
                "n_hours_tied": int(tied.sum()),
                **{f"share_weakest_{s}": float((w == s).mean()) for s in SITE_IDS},
            })
            for h in frame.index:
                hourly.append({"case": case, "metric": metric, "hour_index": h,
                               "weakest_site": w.loc[h], "tied": bool(tied.loc[h])})
        for m, n in combinations(SITE_METRICS, 2):
            pair_rows.append({"case": case, "metric_m": m, "metric_n": n,
                              "share_same_weakest_site": float((weakest[m] == weakest[n]).mean())})
    return pd.DataFrame(hourly), pd.DataFrame(metric_rows), pd.DataFrame(pair_rows)


def a1_commitment(tables, ctx):
    """A1: t*_m against B_count and B_set in each metric's own network."""
    set_rows, metric_rows, within_rows = [], [], []
    for case in CASES:
        b_sets = {}
        for network in N_COMMITTED_COLUMN:
            n = ctx["n_committed"][case, network]
            b_count = sorted(n.index[n == n.min()])
            sets = ctx["committed_set"][case, network].loc[b_count]
            labels = {}
            for h in b_count:  # B_set labels in order of first appearance
                labels.setdefault(sets.loc[h], f"B_set_{len(labels) + 1}")
            b_sets[network] = (b_count, {h: labels[sets.loc[h]] for h in b_count})
            for members, label in labels.items():
                hrs = [h for h in b_count if sets.loc[h] == members]
                set_rows.append({"case": case, "network": network,
                                 "min_n_committed": int(n.min()), "B_set": label,
                                 "n_hours": len(hrs), "hours": _join(hrs),
                                 "committed_generators": _join(sorted(members))})

        for metric, site in UNITS:
            s = series(tables, case, metric, site)
            b_count, label_of = b_sets[NETWORK[metric]]
            ts = t_star(s)
            metric_rows.append({
                "case": case, "metric": metric, "site": site, "network": NETWORK[metric],
                "size_B_count": len(b_count), "t_star": _join(ts),
                "t_star_in_B_count": bool(set(ts) & set(b_count)),
                "t_star_B_set": _join(sorted({label_of[t] for t in ts if t in label_of}))
                                or "outside_B_count",
                "J_K_B_count": jaccard(critical_set(s), b_count),
            })

        for network, (b_count, label_of) in b_sets.items():
            for label in sorted(set(label_of.values())):
                hrs = [h for h in b_count if label_of[h] == label]
                if len(hrs) < 2:
                    continue
                for site in SITE_IDS:
                    for m, n in PAIRS:
                        sm = series(tables, case, m, site).loc[hrs]
                        sn = series(tables, case, n, site).loc[hrs]
                        tm, tn = t_star(sm), t_star(sn)
                        within_rows.append({
                            "case": case, "network": network, "B_set": label,
                            "n_hours": len(hrs), "site": site, "metric_m": m, "metric_n": n,
                            "informative": (m, n) not in BY_CONSTRUCTION,
                            "t_star_m_in_B_set": _join(tm), "t_star_n_in_B_set": _join(tn),
                            "t_star_agree": bool(set(tm) & set(tn)),
                            "kendall_tau": kendalltau(sm.to_numpy(), sn.to_numpy()).statistic,
                        })
    return pd.DataFrame(set_rows), pd.DataFrame(metric_rows), pd.DataFrame(within_rows)


def core_tables(tables, ctx):
    d4_metrics, d4_pairs, d4_ranking = d4_c4(tables)
    d2_events, d2_metrics, d2_pairs = d2_changes(tables, ctx)
    d3_hourly, d3_metrics, d3_pairs = d3_spatial(tables)
    a1_b_set, a1_metrics, a1_within = a1_commitment(tables, ctx)
    return {
        "d1_pairs": d1_pairs(tables, ctx),
        "d1_t_star": d1_t_star(tables, ctx),
        "d4_c4_metrics": d4_metrics,
        "d4_c4_pairs": d4_pairs,
        "d4_c4c_site_ranking": d4_ranking,
        "d5_snapshots": d5_snapshots(tables, ctx),
        "d2_events": d2_events,
        "d2_metrics": d2_metrics,
        "d2_pairs": d2_pairs,
        "d3_weakest_site": d3_hourly,
        "d3_metrics": d3_metrics,
        "d3_pairs": d3_pairs,
        "a1_b_set": a1_b_set,
        "a1_metrics": a1_metrics,
        "a1_within_b_set": a1_within,
    }


# =============================================================================
# 5. Reconstruction from metric_variables.csv (A2, A3, RQ3)
# =============================================================================

def _long(frame, metric):
    """Wide (case, hour_index) x site frame -> long values table."""
    out = frame.stack().rename("value").reset_index()
    out.columns = ["case", "hour_index", "site", "value"]
    out.insert(2, "metric", metric)
    return out


def _by_site(frame, metric):
    return frame.rename(columns=lambda b: SITE_OF[metric][b])[list(SITE_IDS)]


def metric_terms(variables):
    """SCMVA, IF_ji, V, Z_ii and w_ij per metric, indexed (case, hour_index)."""
    z_rr = _wide(variables, "SDSCR", "z_rr", part="complex", pair=True)
    return {
        "SCMVA_SCR": _by_site(_wide(variables, "SCR", "scmva"), "SCR"),
        "SCMVA_WSCR": _by_site(_wide(variables, "WSCR", "scmva"), "WSCR"),
        "SCMVA_ESCR": _by_site(_wide(variables, "ESCR", "scmva"), "ESCR"),
        # columns (j, i): IF_ji, observe j when disturbing i
        "IF": _wide(variables, "ESCR", "if_magnitude", pair=True),
        "V_SDSCR": _by_site(_wide(variables, "SDSCR", "v_pf", part="complex"), "SDSCR"),
        "Z_ii_SDSCR": pd.DataFrame({SITE_OF["SDSCR"][b]: z_rr[(b, b)]
                                    for b in SITE_OF["SDSCR"]})[list(SITE_IDS)],
        # columns (i, j): w_ij
        "w": _wide(variables, "SDSCR", "w_complex", part="complex", pair=True),
    }


def reconstruct(terms, p_mw):
    """Metric values from the recorded terms, with p_mw[metric] = {site: MW}."""
    out = []
    p = p_mw["SCR"]
    out.append(_long(terms["SCMVA_SCR"].apply(lambda c: c / p[c.name]), "SCR"))

    p = p_mw["WSCR"]
    numerator = sum(terms["SCMVA_WSCR"][s] * p[s] for s in SITE_IDS)
    out.append(_long((numerator / sum(p.values()) ** 2).to_frame(GROUP), "WSCR"))

    p, escr = p_mw["ESCR"], {}
    for i_bus, i in SITE_OF["ESCR"].items():
        denominator = p[i] + sum(terms["IF"][(j_bus, i_bus)] * p[j]
                                 for j_bus, j in SITE_OF["ESCR"].items() if j_bus != i_bus)
        escr[i] = terms["SCMVA_ESCR"][i] / denominator
    out.append(_long(pd.DataFrame(escr)[list(SITE_IDS)], "ESCR"))

    p, sdscr = p_mw["SDSCR"], {}
    for i_bus, i in SITE_OF["SDSCR"].items():
        effective_p = p[i] / S_BASE_MVA + sum(
            terms["w"][(i_bus, j_bus)] * p[j] / S_BASE_MVA
            for j_bus, j in SITE_OF["SDSCR"].items() if j_bus != i_bus)
        sdscr[i] = (np.abs(terms["V_SDSCR"][i]) ** 2
                    / (np.abs(terms["Z_ii_SDSCR"][i]) * np.abs(effective_p)))
    out.append(_long(pd.DataFrame(sdscr)[list(SITE_IDS)], "SDSCR"))
    return pd.concat(out, ignore_index=True)


def denominators(kind):
    if kind == "as_run":
        return {m: P_MW[AS_RUN[m]] for m in METRICS}
    return {m: P_MW[kind] for m in METRICS}


def attribution_variants(terms):
    """RQ3: numerator only (SCMVA^37 at POI, SCMVA^46 at MV, |V|²/|Z_ii| at LV),
    and interaction removed (IF = 0, w = 0)."""
    rated, output = P_MW["rated"], P_MW["output"]
    lv = S_BASE_MVA * np.abs(terms["V_SDSCR"]) ** 2 / np.abs(terms["Z_ii_SDSCR"])
    wscr_numerator = sum(terms["SCMVA_WSCR"][s] * output[s] for s in SITE_IDS)
    numerator_only = pd.concat([
        _long(terms["SCMVA_SCR"], "SCR"),
        _long(wscr_numerator.to_frame(GROUP), "WSCR"),
        _long(terms["SCMVA_ESCR"], "ESCR"),
        _long(lv, "SDSCR"),
    ], ignore_index=True)
    as_run = reconstruct(terms, denominators("as_run"))
    interaction_removed = pd.concat([
        as_run[as_run["metric"].isin(["SCR", "WSCR"])],
        _long(terms["SCMVA_ESCR"].apply(lambda c: c / rated[c.name]), "ESCR"),
        _long(lv.apply(lambda c: c / output[c.name]), "SDSCR"),
    ], ignore_index=True)
    return {"numerator_only": to_tables(numerator_only),
            "interaction_removed": to_tables(interaction_removed)}


def a2_attribution(primary, variants):
    pair_rows = []
    for case in CASES:
        for site in SITE_IDS:
            for m, n in PAIRS:
                row = {"case": case, "site": site, "metric_m": m, "metric_n": n,
                       "informative": (m, n) not in BY_CONSTRUCTION}
                for name, tables in [("original", primary), *variants.items()]:
                    c = c1_c3(series(tables, case, m, site), series(tables, case, n, site))
                    row[f"{name}_C1_agree"] = c["C1_agree"]
                    row[f"{name}_t_star_m"] = c["t_star_m"]
                    row[f"{name}_t_star_n"] = c["t_star_n"]
                    row[f"{name}_abs_t_star_diff_h"] = c["abs_t_star_diff_h"]
                pair_rows.append(row)

    interaction_rows = []
    for case in CASES:
        for site in SITE_IDS:
            for metric in ("ESCR", "SDSCR"):
                c = c1_c3(series(primary, case, metric, site),
                          series(variants["interaction_removed"], case, metric, site))
                interaction_rows.append({
                    "case": case, "site": site, "metric": metric,
                    "t_star": c["t_star_m"], "t_star_interaction_removed": c["t_star_n"],
                    "C1_agree": c["C1_agree"], "abs_t_star_diff_h": c["abs_t_star_diff_h"],
                    "C2_J_K": c["C2_J_K"], "C3_spearman_rho": c["C3_spearman_rho"],
                })
    return pd.DataFrame(pair_rows), pd.DataFrame(interaction_rows)


def a2_contingency(primary, variants, ctx):
    """C4a attribution: does the sign of r_m survive on numerators only, or with
    interaction removed, and do the hours where r_m > 0 coincide with IF ≈ 0 states?"""
    ann = ctx["annotations"]
    if_int = ann.loc["intact", "IF_zero"]
    if_out = ann.loc[CONTINGENCY_CASE, "IF_zero"]
    r = {name: {u: series(t, CONTINGENCY_CASE, *u) / series(t, "intact", *u) - 1 for u in UNITS}
         for name, t in [("original", primary), *variants.items()]}

    metric_rows = []
    for unit in UNITS:
        r0 = r["original"][unit]
        rising = sign0(r0) > 0
        either = if_int.loc[r0.index] | if_out.loc[r0.index]
        metric_rows.append({
            "metric": unit[0], "site": unit[1],
            **{f"{name}_share_r_negative": float((sign0(r[name][unit]) < 0).mean()) for name in r},
            "n_r_positive": int(rising.sum()),
            "n_r_positive_IF_zero_outage": int((rising & if_out.loc[r0.index]).sum()),
            "n_r_positive_IF_zero_intact": int((rising & if_int.loc[r0.index]).sum()),
            "n_r_positive_no_IF_zero": int((rising & ~either).sum()),
            "share_r_negative_no_IF_zero": float((sign0(r0[~either]) < 0).mean()),
        })

    pair_rows = []
    for site in SITE_IDS:
        for m, n in PAIRS:
            um, un = _unit(m, site), _unit(n, site)
            pair_rows.append({
                "site": site, "metric_m": m, "metric_n": n,
                "informative": (m, n) not in BY_CONSTRUCTION,
                **{f"{name}_C4a_share_same_sign_r":
                       float((sign0(r[name][um]) == sign0(r[name][un])).mean()) for name in r},
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(pair_rows)


def a3_reconstruction_check(values, terms):
    recon = reconstruct(terms, denominators("as_run"))
    keys = ["case", "hour_index", "metric", "site"]
    merged = values.merge(recon, on=keys, suffixes=("_recorded", "_reconstructed"),
                          how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Reconstructed and recorded metric rows do not match one-to-one")
    rel = (merged["value_reconstructed"] / merged["value_recorded"] - 1).abs()
    return (merged.assign(rel=rel).groupby("metric")["rel"].max()
            .rename("max_rel_diff").reset_index())


def compare_tables(reference, other, rtol=1e-9):
    """A3: every column without 'value' in its name must be unchanged."""
    rows = []
    for name, ref in reference.items():
        alt = other[name]
        cols = [c for c in ref.columns if "value" not in c]
        differing, max_diff = [], 0.0
        if len(ref) != len(alt):
            differing = ["<row count>"]
        else:
            for c in cols:
                x, y = ref[c].reset_index(drop=True), alt[c].reset_index(drop=True)
                if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
                    xa, ya = x.to_numpy(float), y.to_numpy(float)
                    same = np.allclose(xa, ya, rtol=rtol, atol=1e-12, equal_nan=True)
                    finite = np.isfinite(xa) & np.isfinite(ya)
                    if finite.any():
                        max_diff = max(max_diff, float(np.abs(xa - ya)[finite].max()))
                else:
                    same = x.astype(str).equals(y.astype(str))
                if not same:
                    differing.append(c)
        rows.append({"table": name, "n_rows": len(ref), "n_columns_compared": len(cols),
                     "unchanged": not differing, "differing_columns": _join(differing),
                     "max_abs_diff_float": max_diff})
    return pd.DataFrame(rows)


# =============================================================================
# 6. A4: |V| vs Z_ii at SCR site 21's outage minimum
# =============================================================================

def a4_decomposition(variables, primary, ctx, site="21"):
    """SCMVA = 100 |V| / |Z_ii|: does |V| move SCR's outage t* at this site?"""
    case = CONTINGENCY_CASE
    poi = next(b for b, s in SITE_OF["SCR"].items() if s == site)
    rows = variables[(variables["metric"] == "SCR") & (variables["bus"] == poi)
                     & (variables["case"] == case)]
    v = rows[rows["variable"] == "v_pf"].set_index("hour_index")["magnitude"].sort_index()
    z = rows[rows["variable"] == "z_ii"].set_index("hour_index")["magnitude"].sort_index()
    scmva = S_BASE_MVA * v / z
    scmva_v1 = S_BASE_MVA / z  # |V| = 1 pu: Z_ii only
    scr = series(primary, case, "SCR", site)
    ts, ts_v1 = t_star(scr), t_star(scmva_v1)
    others = {(m, s): t_star(series(primary, case, m, s))
              for m, s in UNITS if (m, s) != ("SCR", site)}

    roles = {}
    for h in ts:
        roles.setdefault(h, []).append(f"t_star_SCR_{site}")
    for h in ts_v1:
        roles.setdefault(h, []).append(f"t_star_SCR_{site}_V_1pu")
    for (m, s), hrs in others.items():
        for h in hrs:
            roles.setdefault(h, []).append(f"t_star_{m}_{s}")
    state = ctx["summary"].loc[case]
    hour_rows = [{
        "hour_index": h, "roles": _join(roles[h]),
        "V_abs_pu": float(v.loc[h]), "Z_ii_abs_pu": float(z.loc[h]),
        "SCMVA_value": float(scmva.loc[h]),
        "ln_V_abs": float(np.log(v.loc[h])), "ln_Z_ii_abs": float(np.log(z.loc[h])),
        "load_scale": float(ctx["load_scale"].loc[h]),
        **{col: state.loc[h, col] for col in (
            "base_n_under_v", "base_v_min_pu", "base_v_min_bus", "base_gens_at_qmax",
            "base_gens_at_qmin", "base_tap_steps_total", "base_tap_steps")},
    } for h in sorted(roles)]

    t0, t1 = ts[0], ts_v1[0]
    dln_scmva = float(np.log(scmva.loc[t0] / scmva.loc[t1]))
    dln_v = float(np.log(v.loc[t0] / v.loc[t1]))
    dln_z = float(np.log(z.loc[t0] / z.loc[t1]))
    summary_rows = [{
        "case": case, "site": site, "t_star": _join(ts), "t_star_V_1pu": _join(ts_v1),
        "V_moves_t_star": not (set(ts) & set(ts_v1)),
        "t_star_in_K_V_1pu": bool(set(ts) & critical_set(scmva_v1)),
        "dln_SCMVA": dln_scmva, "dln_V_abs": dln_v, "minus_dln_Z_ii_abs": -dln_z,
        "share_dln_SCMVA_from_V": dln_v / dln_scmva if dln_scmva else float("nan"),
        "t_star_base_under_v": bool(ctx["annotations"].loc[(case, t0), "base_under_v"]),
    }]

    pair_rows = []
    for (m, s) in others:
        if s not in (site, GROUP):
            continue
        for variant, ser in [("original", scr), ("V_1pu", scmva_v1.loc[scr.index])]:
            c = c1_c3(ser, series(primary, case, m, s))
            pair_rows.append({"variant": variant, "metric_n": m, "site_n": s,
                              "t_star_SCR": c["t_star_m"], "t_star_n": c["t_star_n"],
                              "C1_agree": c["C1_agree"],
                              "abs_t_star_diff_h": c["abs_t_star_diff_h"]})
    return pd.DataFrame(hour_rows), pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


# =============================================================================
# 7. Summary and driver
# =============================================================================

def _count(frame, column):
    return f"{int(frame[column].sum())}/{len(frame)}"


A4_QUESTIONS = {
    "t_star": "SCR site 21, outage: weakest hour t*",
    "t_star_V_1pu": "Weakest hour with |V| held at 1 pu (Z_ii only)",
    "V_moves_t_star": "Does |V| move the weakest hour?",
    "t_star_in_K_V_1pu": "Is t* still among the 17 weakest hours with |V| = 1 pu?",
    "dln_SCMVA": "Δln SCMVA between the two hours",
    "dln_V_abs": "  of which from |V| (Δln|V|)",
    "minus_dln_Z_ii_abs": "  of which from Z_ii (−Δln|Z_ii|)",
    "share_dln_SCMVA_from_V": "Share of Δln SCMVA from |V|",
    "t_star_base_under_v": "Is t* an under-voltage state (the prime suspect)?",
}


def build_summary(data, ctx, out):
    rows = []

    def add(rq, dimension, criterion, question, result, table=""):
        rows.append({"rq": rq, "dimension": dimension, "criterion": criterion,
                     "question": question, "result": result, "table": table})

    # Context: what was read, and which states are flagged
    add(CONTEXT, "inputs", "", "Hours per case", len(ctx["hours"]))
    for f, h in data["sha256"].items():
        add(CONTEXT, "inputs", "", f"SHA-256 of {f}", h)
    ann = ctx["annotations"]
    for col in ann.columns:
        for case in CASES:
            add(CONTEXT, "annotations", "", f"States flagged {col} [{case}]",
                int(ann.loc[case, col].sum()), "annotations")

    # RQ1 / D1: when is each site weakest?
    d1 = out["d1_pairs"]
    primary = d1[d1["analysis"] == "primary"]
    informative = primary[primary["informative"]]
    add(RQ1, D1, "C1", "H0: does every informative pair agree on the weakest hour?",
        bool(informative["C1_agree"].all()), "d1_pairs")
    for case in CASES:
        add(RQ1, D1, "C1", f"Informative pairs agreeing on the weakest hour [{case}]",
            _count(informative[informative["case"] == case], "C1_agree"), "d1_pairs")
    add(RQ1, D1, "C1", "Which informative pairs disagree?", _join(
        f"{r.case}/{r.site}/{r.metric_m}-{r.metric_n}"
        for r in informative[~informative["C1_agree"]].itertuples()) or "none", "d1_pairs")
    for analysis in d1["analysis"].unique():
        if analysis == "primary":
            continue
        part = d1[(d1["analysis"] == analysis) & d1["informative"]]
        add(RQ1, D1, "C1 check", f"Informative pairs agreeing [{analysis}]",
            _count(part, "C1_agree"), "d1_pairs")
    add(RQ1, D1, "C1", "SCR-WSCR pairs agreeing (shared SCMVA: reported, not counted)",
        _count(primary[~primary["informative"]], "C1_agree"), "d1_pairs")
    add(RQ1, D1, "C2", "Median overlap J of the 17 weakest hours (informative pairs)",
        float(informative["C2_J_K"].median()), "d1_pairs")
    for label in ("strong", "moderate", "weak"):
        add(RQ1, D1, "C3", f"Informative pairs ranking the week alike: {label} (Spearman ρ)",
            int((informative["C3_rho_label"] == label).sum()), "d1_pairs")

    # RQ1 / D4: how does the outage change strength?
    d4m, d4p, d4c = out["d4_c4_metrics"], out["d4_c4_pairs"], out["d4_c4c_site_ranking"]
    d4i = d4p[d4p["informative"]]
    add(RQ1, D4, "C4a", "Informative pairs moving in the same direction in every hour",
        f"{int((d4i['C4a_share_same_sign_r'] == 1).sum())}/{len(d4i)}", "d4_c4_pairs")
    add(RQ1, D4, "C4a", "Lowest share of hours moving in the same direction (informative)",
        float(d4i["C4a_share_same_sign_r"].min()), "d4_c4_pairs")
    add(RQ1, D4, "C4b", "Metrics whose weakest hour leaves the intact 17 weakest hours", _join(
        f"{r.metric}/{r.site}" for r in d4m[d4m["C4b_t_star_out_not_in_K_int"]].itertuples())
        or "none", "d4_c4_metrics")
    add(RQ1, D4, "C4b", "Informative pairs agreeing on whether the weakest hour moves",
        _count(d4i, "C4b_agree"), "d4_c4_pairs")
    add(RQ1, D4, "C4c", "Site-level pairs ranking the sites alike by outage impact",
        _count(d4c, "C4c_agree"), "d4_c4c_site_ranking")

    # RQ1 / D5: would the planning snapshots find the minimum?
    d5 = out["d5_snapshots"]
    add(RQ1, D5, "D5a", "Cases where the lower snapshot is among the 17 weakest hours",
        _count(d5, "D5a_snapshot_min_in_K"), "d5_snapshots")
    add(RQ1, D5, "D5b", "Cases where a snapshot is exactly the weekly minimum",
        f"{int((d5['D5b_g'] <= REL_TIE).sum())}/{len(d5)}", "d5_snapshots")
    worst = d5.loc[d5["D5b_g"].idxmax()]
    add(RQ1, D5, "D5b", "Largest gap g above the weekly minimum", float(worst["D5b_g"]),
        "d5_snapshots")
    add(RQ1, D5, "D5b", "Where the largest gap occurs",
        f"{worst['case']}/{worst['metric']}/{worst['site']}", "d5_snapshots")

    # RQ1 / D2: do sharp changes happen at the same hours?
    d2m, d2p = out["d2_metrics"], out["d2_pairs"]
    d2i = d2p[d2p["informative"]]
    add(RQ1, D2, "", "Median overlap J of the sharp-change hours (informative pairs)",
        float(d2i["J_E"].median()), "d2_pairs")
    add(RQ1, D2, "", "Lowest / highest overlap J of the sharp-change hours (informative pairs)",
        f"{d2i['J_E'].min():.6g} / {d2i['J_E'].max():.6g}", "d2_pairs")
    add(RQ1, D2, "", "Median share of sharp changes at a commitment change",
        float(d2m["share_E_commitment_set_change"].median()), "d2_metrics")

    # RQ1 / D3: do the metrics name the same weakest site?
    for r in out["d3_pairs"].itertuples():
        add(RQ1, D3, "", f"Share of hours naming the same weakest site [{r.case}, "
            f"{r.metric_m}-{r.metric_n}]", float(r.share_same_weakest_site), "d3_pairs")
    for r in out["d3_metrics"].itertuples():
        add(RQ1, D3, "", f"Times the weakest site changes [{r.case}, {r.metric}]",
            int(r.n_weakest_site_changes), "d3_metrics")

    # RQ2 / A1: more than commitment?
    a1m, a1w = out["a1_metrics"], out["a1_within_b_set"]
    dim = "A1 commitment baseline"
    add(RQ2, dim, "A1", "Weakest hours falling in the minimum-commitment hours (B_count)",
        _count(a1m, "t_star_in_B_count"), "a1_metrics")
    add(RQ2, dim, "A1", "Metrics adding information beyond commitment (t* outside B_count)",
        _join(f"{r.case}/{r.metric}/{r.site}"
              for r in a1m[~a1m["t_star_in_B_count"]].itertuples()) or "none", "a1_metrics")
    if not a1w.empty:
        add(RQ2, dim, "A1", "Informative pairs picking the same weakest hour within one "
            "committed set (B_set)", _count(a1w[a1w["informative"]], "t_star_agree"),
            "a1_within_b_set")

    # RQ3 / A2-A4: why do conclusions differ?
    a2, a2i = out["a2_pairs"], out["a2_interaction"]
    dis = a2[a2["informative"] & ~a2["original_C1_agree"]]
    dim = "A2 conventions vs formulas"
    add(RQ3, dim, "A2", "Disagreeing pairs that agree once compared on numerators only",
        f"{int(dis['numerator_only_C1_agree'].sum())}/{len(dis)}", "a2_pairs")
    add(RQ3, dim, "A2", "Disagreeing pairs that agree once interaction is removed",
        f"{int(dis['interaction_removed_C1_agree'].sum())}/{len(dis)}", "a2_pairs")
    add(RQ3, dim, "A2", "ESCR/SDSCR weakest hours moved by removing interaction",
        f"{int((~a2i['C1_agree']).sum())}/{len(a2i)}", "a2_interaction")
    c4m, c4p = out["a2_c4_metrics"], out["a2_c4_pairs"]
    c4i = c4p[c4p["informative"]]
    for name in ("original", "numerator_only", "interaction_removed"):
        add(RQ3, dim, "A2 C4a", f"Informative pairs moving in the same direction in every hour [{name}]",
            f"{int((c4i[f'{name}_C4a_share_same_sign_r'] == 1).sum())}/{len(c4i)}", "a2_c4_pairs")
    escr = c4m[c4m["metric"] == "ESCR"]
    add(RQ3, dim, "A2 C4a", "ESCR hours rising under the outage (all sites)",
        int(escr["n_r_positive"].sum()), "a2_c4_metrics")
    add(RQ3, dim, "A2 C4a", "  of which IF ≈ 0 in the outage state",
        int(escr["n_r_positive_IF_zero_outage"].sum()), "a2_c4_metrics")
    add(RQ3, dim, "A2 C4a", "  of which IF ≈ 0 in neither state",
        int(escr["n_r_positive_no_IF_zero"].sum()), "a2_c4_metrics")

    a3 = out["a3_denominator_invariance"]
    dim = "A3 denominators"
    add(RQ3, dim, "A3", "Are D1-D5 and A1 unchanged under rated, output and MVA denominators?",
        bool(a3["unchanged"].all()), "a3_denominator_invariance")
    add(RQ3, dim, "A3", "Largest relative difference recomputing metrics from their terms",
        float(out["a3_reconstruction_check"]["max_rel_diff"].max()), "a3_reconstruction_check")

    a4 = out["a4_summary"].iloc[0]
    for col, question in A4_QUESTIONS.items():
        add(RQ3, "A4 look-2 hypothesis", "A4", question, a4[col], "a4_summary")
    return pd.DataFrame(rows)


def table_index(out):
    if missing := set(out) - set(TABLES):
        raise ValueError(f"Tables without an index entry: {sorted(missing)}")
    return pd.DataFrame([{"table": name, "rq": rq, "dimension": dim, "criteria": crit,
                          "answers": answers}
                         for name, (rq, dim, crit, answers) in TABLES.items()])


def run(input_dir=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("Output directory must differ from the input directory")
    data = load_inputs(input_dir)
    hours = check_inputs(data)
    ctx = build_context(data, hours)
    values = site_values(data["metrics"])
    primary = to_tables(values)
    core = core_tables(primary, ctx)
    out = dict(core)

    terms = metric_terms(data["variables"])
    variants = attribution_variants(terms)
    out["a2_pairs"], out["a2_interaction"] = a2_attribution(primary, variants)
    out["a2_c4_metrics"], out["a2_c4_pairs"] = a2_contingency(primary, variants, ctx)
    out["a3_denominator_invariance"] = pd.concat([
        compare_tables(core, core_tables(to_tables(reconstruct(terms, denominators(kind))), ctx))
        .assign(denominator=kind)
        for kind in ("as_run", "rated", "output", "mva")
    ], ignore_index=True)
    out["a3_reconstruction_check"] = a3_reconstruction_check(values, terms)
    out["a4_hours"], out["a4_summary"], out["a4_pairs"] = a4_decomposition(
        data["variables"], primary, ctx)
    out["annotations"] = ctx["annotations"].reset_index()
    out["summary"] = build_summary(data, ctx, out)
    out["index"] = table_index(out)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = run(args.input_dir, args.output_dir)
    summary = out["summary"]
    summary = summary[~summary["question"].str.startswith(("SHA-256", "States flagged"))]
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 90,
                           "display.width", 200):
        for rq, part in summary.groupby("rq", sort=False):
            print(f"\n{rq}")
            print(part[["dimension", "criterion", "question", "result"]].to_string(index=False))
    print(f"\nSaved {len(out)} tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
