"""
Temporal-spatial analysis of the 37-bus short-circuit-level experiment.

Scientific questions
--------------------
The analysis preserves the bus dimension and evaluates:

1. whether local SCL trajectories evolve differently across buses;
2. whether different buses attain minimum SCL under different system states;
3. whether the spatial ordering of SCL changes over time;
4. how the fixed line contingency modifies temporal-spatial SCL evolution.

For bus-wise temporal visualization, each bus may be normalized relative to
its own temporal mean:

    delta_i(t) = 100 * [SCL_i(t) - mean_t(SCL_i)] / mean_t(SCL_i)

This preserves local temporal variation without privileging an arbitrary
initial state.

The analysis also produces targeted illustrations for selected observed
behaviors, including spatial-order reversals, differences in temporal
sensitivity between buses with similar SCL magnitude, and intact-versus-
contingency trajectory differences.

The contingency/load relationship is computed only as a descriptive
diagnostic and is not interpreted as causal.

Expected inputs
---------------
scl_timeseries.csv
dispatch_schedule.csv
selected_week.csv
contingency_effect.csv

Example
-------
python -m experiments.glover37.analyze_temporal \
    --input-dir results/glover37/temporal \
    --output-dir results/glover37/temporal/analysis
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED = {
    "scl": "scl_timeseries.csv",
    "dispatch": "dispatch_schedule.csv",
    "week": "selected_week.csv",
    "cont": "contingency_effect.csv",
}


def parse_args():
    p = argparse.ArgumentParser(description="Analyze temporal-spatial SCL variation in the 37-bus experiment.")
    p.add_argument("--input-dir", type=Path, default=Path("."))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/glover37/temporal/analysis"),
    )
    return p.parse_args()


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_inputs(input_dir):
    paths = {k: input_dir / v for k, v in REQUIRED.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    scl = pd.read_csv(paths["scl"])
    dispatch = pd.read_csv(paths["dispatch"])
    week = pd.read_csv(paths["week"])
    cont = pd.read_csv(paths["cont"])

    require_columns(scl, ["hour_index", "timestamp", "case", "bus", "SCL_MVA"], "scl_timeseries.csv")
    require_columns(dispatch, ["hour_index", "timestamp", "generator", "bus", "committed", "dispatch_MW"], "dispatch_schedule.csv")
    require_columns(week, ["hour_index", "timestamp", "se3_mw", "load_scale"], "selected_week.csv")
    require_columns(cont, ["hour_index", "timestamp", "bus", "SCL_intact_MVA", "SCL_contingency_MVA", "delta_SCL_pct"], "contingency_effect.csv")

    for df in (scl, dispatch, week, cont):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    for df in (scl, dispatch, cont):
        df["bus"] = pd.to_numeric(df["bus"], errors="raise").astype(int)

    dispatch["committed"] = dispatch["committed"].astype(bool)
    return scl, dispatch, week, cont


def commitment_context(dispatch):
    rows = []
    for h, g in dispatch.groupby("hour_index", sort=True):
        committed = g[g["committed"]].copy()
        offline = g[~g["committed"]].copy()
        rows.append({
            "hour_index": int(h),
            "n_committed": int(len(committed)),
            "committed_buses": ";".join(map(str, committed["bus"].astype(int).tolist())),
            "offline_buses": ";".join(map(str, offline["bus"].astype(int).tolist())),
            "offline_generators": ";".join(offline["generator"].astype(str).tolist()),
            "total_dispatch_MW": float(g["dispatch_MW"].sum()),
        })
    ctx = pd.DataFrame(rows)
    ctx["commitment_changed"] = (
        ctx["offline_generators"].fillna("") != ctx["offline_generators"].shift().fillna("")
    )
    ctx.loc[ctx.index[0], "commitment_changed"] = False
    return ctx


def build_intact(scl):
    intact = scl[scl["case"].str.lower() == "intact"].copy()
    if intact.empty:
        raise ValueError("No intact case found in scl_timeseries.csv")
    intact = intact.sort_values(["bus", "hour_index"]).reset_index(drop=True)

    means = intact.groupby("bus")["SCL_MVA"].mean().rename("SCL_temporal_mean_MVA")
    intact = intact.merge(means, on="bus", how="left")
    intact["SCL_deviation_from_mean_pct"] = 100.0 * (
        intact["SCL_MVA"] - intact["SCL_temporal_mean_MVA"]
    ) / intact["SCL_temporal_mean_MVA"]

    return intact


def intact_variation(intact):
    out = (
        intact.groupby("bus")
        .agg(
            SCL_min_MVA=("SCL_MVA", "min"),
            SCL_max_MVA=("SCL_MVA", "max"),
            SCL_mean_MVA=("SCL_MVA", "mean"),
            mean_normalized_min_pct=("SCL_deviation_from_mean_pct", "min"),
            mean_normalized_max_pct=("SCL_deviation_from_mean_pct", "max"),
            mean_normalized_std_pct=("SCL_deviation_from_mean_pct", "std"),
        )
        .reset_index()
    )
    out["SCL_range_MVA"] = out["SCL_max_MVA"] - out["SCL_min_MVA"]
    out["relative_temporal_range_pct"] = 100.0 * out["SCL_range_MVA"] / out["SCL_mean_MVA"]
    out["mean_normalized_range_pp"] = out["mean_normalized_max_pct"] - out["mean_normalized_min_pct"]
    return out.sort_values("relative_temporal_range_pct", ascending=False)


def critical_states(intact, week, ctx):
    idx = intact.groupby("bus")["SCL_MVA"].idxmin()
    crit = intact.loc[idx, ["bus", "hour_index", "timestamp", "SCL_MVA", "SCL_deviation_from_mean_pct"]].copy()
    crit = crit.rename(columns={
        "hour_index": "critical_hour_intact",
        "timestamp": "critical_timestamp_intact",
        "SCL_MVA": "min_SCL_intact_MVA",
        "SCL_deviation_from_mean_pct": "min_deviation_from_mean_pct",
    })
    context = week[["hour_index", "se3_mw", "load_scale"]].merge(ctx, on="hour_index", how="left")
    crit = crit.merge(context, left_on="critical_hour_intact", right_on="hour_index", how="left").drop(columns="hour_index")
    crit = crit.sort_values(["critical_hour_intact", "bus"]).reset_index(drop=True)

    counts = (
        crit.groupby(["critical_hour_intact", "critical_timestamp_intact"])
        .size().reset_index(name="n_buses_with_minimum")
        .sort_values("critical_hour_intact")
    )
    return crit, counts


def rank_analysis(intact):
    ranks = intact[["hour_index", "timestamp", "bus", "SCL_MVA"]].copy()
    ranks["SCL_rank"] = ranks.groupby("hour_index")["SCL_MVA"].rank(method="min", ascending=False).astype(int)

    summary = (
        ranks.groupby("bus")
        .agg(best_rank=("SCL_rank", "min"), worst_rank=("SCL_rank", "max"), n_distinct_ranks=("SCL_rank", "nunique"))
        .reset_index()
    )
    summary["rank_span"] = summary["worst_rank"] - summary["best_rank"]
    summary["rank_changed"] = summary["rank_span"] > 0

    wide = intact.pivot(index="hour_index", columns="bus", values="SCL_MVA").sort_index()
    reversals = []
    for a, b in combinations(wide.columns, 2):
        d = (wide[a] - wide[b]).to_numpy(float)
        s = np.sign(d)
        nz = s[s != 0]
        n = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
        if n:
            reversals.append({"bus_a": int(a), "bus_b": int(b), "n_order_reversals": n})
    reversals = pd.DataFrame(reversals, columns=["bus_a", "bus_b", "n_order_reversals"])
    if not reversals.empty:
        reversals = reversals.sort_values(["n_order_reversals", "bus_a", "bus_b"], ascending=[False, True, True])
    return ranks, summary, reversals


def contingency_analysis(cont):
    c = cont.sort_values(["bus", "hour_index"]).copy()

    ii = c.groupby("bus")["SCL_intact_MVA"].idxmin()
    ic = c.groupby("bus")["SCL_contingency_MVA"].idxmin()
    ci = c.loc[ii, ["bus", "hour_index", "timestamp", "SCL_intact_MVA"]].rename(columns={
        "hour_index": "critical_hour_intact", "timestamp": "critical_timestamp_intact", "SCL_intact_MVA": "min_SCL_intact_MVA"
    })
    cc = c.loc[ic, ["bus", "hour_index", "timestamp", "SCL_contingency_MVA"]].rename(columns={
        "hour_index": "critical_hour_contingency", "timestamp": "critical_timestamp_contingency", "SCL_contingency_MVA": "min_SCL_contingency_MVA"
    })
    crit = ci.merge(cc, on="bus")
    crit["critical_hour_shift"] = crit["critical_hour_contingency"] - crit["critical_hour_intact"]
    crit["critical_state_changed"] = crit["critical_hour_shift"] != 0

    ranks = c[["hour_index", "timestamp", "bus", "SCL_intact_MVA", "SCL_contingency_MVA"]].copy()
    ranks["rank_intact"] = ranks.groupby("hour_index")["SCL_intact_MVA"].rank(method="min", ascending=False).astype(int)
    ranks["rank_contingency"] = ranks.groupby("hour_index")["SCL_contingency_MVA"].rank(method="min", ascending=False).astype(int)
    ranks["rank_shift"] = ranks["rank_contingency"] - ranks["rank_intact"]
    ranks["rank_changed"] = ranks["rank_shift"] != 0

    rank_summary = (
        ranks.groupby("bus")
        .agg(
            hours_with_rank_change=("rank_changed", "sum"),
            max_abs_rank_shift=("rank_shift", lambda x: int(np.abs(x).max())),
            min_rank_shift=("rank_shift", "min"),
            max_rank_shift=("rank_shift", "max"),
        )
        .reset_index()
    )
    rank_summary["fraction_hours_rank_changed"] = rank_summary["hours_with_rank_change"] / ranks["hour_index"].nunique()

    effect = (
        c.groupby("bus")
        .agg(
            min_delta_SCL_pct=("delta_SCL_pct", "min"),
            max_delta_SCL_pct=("delta_SCL_pct", "max"),
            median_delta_SCL_pct=("delta_SCL_pct", "median"),
        )
        .reset_index()
    )
    effect["effect_range_pp"] = effect["max_delta_SCL_pct"] - effect["min_delta_SCL_pct"]
    effect = effect.sort_values("effect_range_pp", ascending=False)
    return c, crit, ranks, rank_summary, effect


def contiguous_commitment_segments(ctx):
    x = ctx[["hour_index", "offline_generators"]].copy()
    x["segment"] = (x["offline_generators"] != x["offline_generators"].shift()).cumsum()
    return (
        x.groupby("segment")
        .agg(start_hour=("hour_index", "min"), end_hour=("hour_index", "max"), n_hours=("hour_index", "size"), offline_generators=("offline_generators", "first"))
        .reset_index()
    )


def contingency_load_diagnostic(cont, week, ctx):
    """Quantify load-SCL relation inside the longest fixed-commitment segment.

    This avoids mixing commitment transitions into the correlation diagnostic.
    Correlation is descriptive only; it is not a causal estimate.
    """
    seg = contiguous_commitment_segments(ctx)
    longest = seg.sort_values(["n_hours", "start_hour"], ascending=[False, True]).iloc[0]
    h0, h1 = int(longest.start_hour), int(longest.end_hour)

    x = cont.merge(week[["hour_index", "se3_mw", "load_scale"]], on="hour_index", how="left")
    x = x[x["hour_index"].between(h0, h1)].copy()

    rows = []
    for bus, g in x.groupby("bus"):
        corr_i = g["SCL_intact_MVA"].corr(g["se3_mw"])
        corr_c = g["SCL_contingency_MVA"].corr(g["se3_mw"])
        corr_d = g["delta_SCL_pct"].corr(g["se3_mw"])
        rows.append({
            "bus": int(bus),
            "segment_start_hour": h0,
            "segment_end_hour": h1,
            "segment_n_hours": int(longest.n_hours),
            "offline_generators": longest.offline_generators,
            "corr_intact_SCL_vs_load": corr_i,
            "corr_contingency_SCL_vs_load": corr_c,
            "corr_contingency_effect_vs_load": corr_d,
            "correlation_shift_cont_minus_intact": corr_c - corr_i,
        })
    out = pd.DataFrame(rows)
    out["direction_changed"] = np.sign(out["corr_intact_SCL_vs_load"]) != np.sign(out["corr_contingency_SCL_vs_load"])
    return out.sort_values("correlation_shift_cont_minus_intact"), longest


def add_commitment_markers(ax, ctx):
    hours = ctx.loc[ctx["commitment_changed"], "hour_index"].tolist()
    for h in hours:
        ax.axvline(h, linewidth=0.55, alpha=0.35)


def fig_mean_normalized_heatmap(intact, ctx, outdir):
    wide = intact.pivot(index="bus", columns="hour_index", values="SCL_deviation_from_mean_pct").sort_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    vmax = np.nanmax(np.abs(wide.to_numpy()))
    im = ax.imshow(wide.to_numpy(), aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title("Intact temporal SCL deviation from each bus's temporal mean")
    ax.set_xlabel("Hour index")
    ax.set_ylabel("Bus")
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels(wide.index.astype(str))
    for h in ctx.loc[ctx["commitment_changed"], "hour_index"]:
        ax.axvline(h - 0.5, color="k", linewidth=0.45, alpha=0.35)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Deviation from own temporal mean [%]")
    fig.tight_layout()
    fig.savefig(outdir / "figure_intact_mean_normalized_heatmap.png", dpi=200)
    plt.close(fig)


def fig_bus13_14(intact, ctx, outdir):
    sub = intact[intact["bus"].isin([13, 14])].copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for bus, g in sub.groupby("bus"):
        ax.plot(g["hour_index"], g["SCL_MVA"], label=f"Bus {bus}")
    add_commitment_markers(ax, ctx)
    ax.set_title("Buses 13 and 14: temporal SCL ordering")
    ax.set_xlabel("Hour index")
    ax.set_ylabel("SCL [MVA]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "figure_case_bus13_bus14_rank_reversal.png", dpi=200)
    plt.close(fig)


def fig_bus5_34(intact, ctx, outdir):
    sub = intact[intact["bus"].isin([5, 34])].copy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for bus, g in sub.groupby("bus"):
        axes[0].plot(g["hour_index"], g["SCL_MVA"], label=f"Bus {bus}")
        axes[1].plot(g["hour_index"], g["SCL_deviation_from_mean_pct"], label=f"Bus {bus}")
    for ax in axes:
        add_commitment_markers(ax, ctx)
        ax.legend()
    axes[0].set_title("Buses 5 and 34: similar SCL magnitude, different temporal sensitivity")
    axes[0].set_ylabel("SCL [MVA]")
    axes[1].set_ylabel("Deviation from own temporal mean [%]")
    axes[1].set_xlabel("Hour index")
    fig.tight_layout()
    fig.savefig(outdir / "figure_case_bus5_bus34_sensitivity.png", dpi=200)
    plt.close(fig)


def fig_bus44_contingency(cont, week, ctx, outdir):
    g = cont[cont["bus"] == 44].sort_values("hour_index").merge(
        week[["hour_index", "se3_mw"]], on="hour_index", how="left"
    )
    if g.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(g["hour_index"], g["SCL_intact_MVA"], label="Intact")
    axes[0].plot(g["hour_index"], g["SCL_contingency_MVA"], label="Contingency")
    axes[0].set_ylabel("SCL [MVA]")
    axes[0].set_title("Bus 44: contingency modifies the temporal SCL trajectory")
    axes[0].legend()

    axes[1].plot(g["hour_index"], g["delta_SCL_pct"], label="Contingency effect")
    axes[1].set_ylabel("Contingency effect [%]")
    axes[1].set_xlabel("Hour index")
    axes[1].legend()
    for ax in axes:
        add_commitment_markers(ax, ctx)
    fig.tight_layout()
    fig.savefig(outdir / "figure_case_bus44_contingency_interaction.png", dpi=200)
    plt.close(fig)


def fig_critical_hours(critical, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(critical["critical_hour_intact"], critical["bus"])
    ax.set_title("Hour of minimum intact SCL by bus")
    ax.set_xlabel("Critical hour index")
    ax.set_ylabel("Bus")
    ax.set_yticks(sorted(critical["bus"].unique()))
    fig.tight_layout()
    fig.savefig(outdir / "figure_intact_critical_hours.png", dpi=200)
    plt.close(fig)


def fig_rank_heatmap(ranks, outdir):
    wide = ranks.pivot(index="bus", columns="hour_index", values="SCL_rank").sort_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(wide.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_title("Intact spatial SCL rank evolution")
    ax.set_xlabel("Hour index")
    ax.set_ylabel("Bus")
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels(wide.index.astype(str))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("SCL rank (1 = highest)")
    fig.tight_layout()
    fig.savefig(outdir / "figure_intact_rank_heatmap.png", dpi=200)
    plt.close(fig)


def fig_contingency_effect_heatmap(cont, outdir):
    wide = cont.pivot(index="bus", columns="hour_index", values="delta_SCL_pct").sort_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(wide.to_numpy(), aspect="auto", interpolation="nearest", cmap="RdBu_r")
    ax.set_title("Time-varying local effect of the fixed contingency")
    ax.set_xlabel("Hour index")
    ax.set_ylabel("Bus")
    ax.set_yticks(np.arange(len(wide.index)))
    ax.set_yticklabels(wide.index.astype(str))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Contingency effect on SCL [%]")
    fig.tight_layout()
    fig.savefig(outdir / "figure_contingency_effect_heatmap.png", dpi=200)
    plt.close(fig)


def fig_contingency_load_diagnostic(cont, week, diagnostic, longest, outdir):
    # Plot a small set where the contingency most changes the load-SCL relationship.
    buses = diagnostic.head(6)["bus"].astype(int).tolist()
    h0, h1 = int(longest.start_hour), int(longest.end_hour)
    x = cont[cont["hour_index"].between(h0, h1) & cont["bus"].isin(buses)].merge(
        week[["hour_index", "se3_mw"]], on="hour_index", how="left"
    )
    load = week[week["hour_index"].between(h0, h1)][["hour_index", "se3_mw"]].copy()
    load["load_z"] = (load["se3_mw"] - load["se3_mw"].mean()) / load["se3_mw"].std()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(load["hour_index"], load["load_z"], linewidth=2.0, label="Load (standardized)")
    for bus, g in x.groupby("bus"):
        y = g["SCL_contingency_MVA"]
        z = (y - y.mean()) / y.std()
        ax.plot(g["hour_index"], z, label=f"Bus {bus} contingency SCL")
    ax.set_title(f"Diagnostic: load and contingency SCL during fixed commitment (hours {h0}-{h1})")
    ax.set_xlabel("Hour index")
    ax.set_ylabel("Standardized value")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "figure_diagnostic_contingency_load_relation.png", dpi=200)
    plt.close(fig)


def write_readme(outdir, intact, variation, critical, rank_summary, reversals, contcrit, diagnostic, longest):
    high = variation.head(10)[["bus", "relative_temporal_range_pct"]]
    lines = [
        "Temporal-spatial SCL analysis", "=" * 29, "",
        "Core design", "-----------",
        "- Bus-wise spatial information is preserved.",
        "- No system-wide mean change is used as a primary temporal metric.",
        "- Heatmap normalization is relative to each bus's temporal mean, not hour 0.",
        "- Contingency analysis is kept as a dedicated block.", "",
        "Mean-normalized definition", "--------------------------",
        "delta_i(t) = 100 * [SCL_i(t) - mean_t(SCL_i)] / mean_t(SCL_i)", "",
        "Quick diagnostics", "-----------------",
        f"Buses with intact rank changes: {int(rank_summary['rank_changed'].sum())}/{intact['bus'].nunique()}",
        f"Bus pairs with at least one intact ordering reversal: {len(reversals)}",
        f"Buses whose critical hour changes under contingency: {int(contcrit['critical_state_changed'].sum())}/{intact['bus'].nunique()}",
        f"Longest fixed-commitment interval used for load diagnostic: hours {int(longest.start_hour)}-{int(longest.end_hour)} ({int(longest.n_hours)} h)", "",
        "Highest relative temporal ranges (intact)", "----------------------------------------",
    ]
    for _, r in high.iterrows():
        lines.append(f"Bus {int(r.bus)}: {r.relative_temporal_range_pct:.3f}%")
    lines += ["", "Targeted figures", "----------------",
        "figure_intact_mean_normalized_heatmap.png",
        "figure_case_bus13_bus14_rank_reversal.png",
        "figure_case_bus5_bus34_sensitivity.png",
        "figure_case_bus44_contingency_interaction.png",
        "figure_intact_critical_hours.png",
        "figure_intact_rank_heatmap.png",
        "figure_contingency_effect_heatmap.png",
        "figure_diagnostic_contingency_load_relation.png (diagnostic; not automatically a paper figure)", "",
        "Important interpretation boundary", "---------------------------------",
        "The contingency/load correlation output is descriptive. It is calculated within the longest interval with unchanged generator commitment to avoid mixing commitment transitions into the relation. Correlation does not establish that load itself causes the SCL response; load scaling also changes dispatch, voltage, losses and other state variables.",
    ]
    (outdir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scl, dispatch, week, cont = load_inputs(args.input_dir)
    ctx = commitment_context(dispatch)
    intact = build_intact(scl)
    variation = intact_variation(intact)
    critical, critical_counts = critical_states(intact, week, ctx)
    ranks, rank_summary, reversals = rank_analysis(intact)
    cont_ts, contcrit, contranks, contrank_summary, conteffect = contingency_analysis(cont)
    load_diag, longest = contingency_load_diagnostic(cont_ts, week, ctx)

    # Core tables
    ctx.to_csv(args.output_dir / "commitment_state_by_hour.csv", index=False)
    intact.to_csv(args.output_dir / "intact_trajectories.csv", index=False)
    variation.to_csv(args.output_dir / "intact_variation_by_bus.csv", index=False)
    critical.to_csv(args.output_dir / "intact_critical_states.csv", index=False)
    critical_counts.to_csv(args.output_dir / "intact_critical_hour_counts.csv", index=False)
    ranks.to_csv(args.output_dir / "intact_rank_timeseries.csv", index=False)
    rank_summary.to_csv(args.output_dir / "intact_rank_summary_by_bus.csv", index=False)
    reversals.to_csv(args.output_dir / "intact_pairwise_rank_reversals.csv", index=False)

    # Dedicated contingency tables
    cont_ts.to_csv(args.output_dir / "contingency_effect_timeseries.csv", index=False)
    contcrit.to_csv(args.output_dir / "contingency_critical_state_comparison.csv", index=False)
    contranks.to_csv(args.output_dir / "contingency_rank_timeseries.csv", index=False)
    contrank_summary.to_csv(args.output_dir / "contingency_rank_summary_by_bus.csv", index=False)
    conteffect.to_csv(args.output_dir / "contingency_effect_summary_by_bus.csv", index=False)
    load_diag.to_csv(args.output_dir / "contingency_load_relation_diagnostic.csv", index=False)

    # Figures
    fig_mean_normalized_heatmap(intact, ctx, args.output_dir)
    fig_bus13_14(intact, ctx, args.output_dir)
    fig_bus5_34(intact, ctx, args.output_dir)
    fig_bus44_contingency(cont_ts, week, ctx, args.output_dir)
    fig_critical_hours(critical, args.output_dir)
    fig_rank_heatmap(ranks, args.output_dir)
    fig_contingency_effect_heatmap(cont_ts, args.output_dir)
    fig_contingency_load_diagnostic(cont_ts, week, load_diag, longest, args.output_dir)

    write_readme(args.output_dir, intact, variation, critical, rank_summary, reversals, contcrit, load_diag, longest)

    print("Temporal-spatial SCL analysis complete.")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Buses with intact rank changes: {int(rank_summary['rank_changed'].sum())}/{intact['bus'].nunique()}")
    print(f"Bus pairs with ordering reversals: {len(reversals)}")
    print(f"Buses whose critical hour changes under contingency: {int(contcrit['critical_state_changed'].sum())}/{intact['bus'].nunique()}")
    print(f"Longest fixed-commitment segment: hours {int(longest.start_hour)}-{int(longest.end_hour)} ({int(longest.n_hours)} h)")
    print("Largest contingency-induced shifts in load-SCL correlation:")
    print(load_diag.head(10)[['bus','corr_intact_SCL_vs_load','corr_contingency_SCL_vs_load','correlation_shift_cont_minus_intact','direction_changed']].to_string(index=False))


if __name__ == "__main__":
    main()
