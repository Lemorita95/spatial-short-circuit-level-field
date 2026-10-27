"""N-1 line contingency set for the system-strength experiment (evidence plan v2).

Fixes the set before any metric is computed: a line outage is included only if it
islands no bus and both pre-fault power flows (37-bus base, 46-bus with IBRs)
converge in every hour of the week, with commitment and setpoints held at the
intact schedule. No strength metric is evaluated here.

Run from repository root:
    python -m experiments.glover37.screen_contingencies
    python -m experiments.glover37.screen_contingencies --hours 2   # smoke test

Output: results/glover37/strength/raw/contingency_set.csv, one row per line.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from models.PowerFlow import PowerFlow
from experiments.glover37.run_strength_metrics import outage_network
from experiments.glover37.strength_config import SITES, augment_network
from experiments.glover37.run_temporal import (
    GEN_LIMITS_FILE, ROOT, baseline_load_mw, prepare_network, read_gen_limits,
    read_se3, select_week, sequential_dispatch,
)

DEFAULT_OUTPUT = ROOT / 'results' / 'glover37' / 'strength' / 'raw' / 'contingency_set.csv'


def case_labels(lines):
    """{line key: case label}; the circuit number is kept only for parallel lines."""
    pairs = Counter(tuple(sorted((ln.bus_1, ln.bus_2))) for ln in lines.values())
    labels = {}
    for key, ln in lines.items():
        a, b = sorted((ln.bus_1, ln.bus_2))
        ckt = key.rsplit('ckt', 1)[-1].strip()
        labels[key] = f'line_{a}_{b}_ckt{ckt}_out' if pairs[(a, b)] > 1 else f'line_{a}_{b}_out'
    return labels


def islands(net, outaged):
    """True if removing `outaged` leaves any bus disconnected from the rest."""
    adjacency = {b: set() for b in net.buses}
    for key, branch in (net.lines | net.transformers).items():
        if key == outaged or not branch.status:
            continue
        adjacency[branch.bus_1].add(branch.bus_2)
        adjacency[branch.bus_2].add(branch.bus_1)
    start = next(iter(adjacency))
    seen, stack = {start}, [start]
    while stack:
        for other in adjacency[stack.pop()] - seen:
            seen.add(other)
            stack.append(other)
    return len(seen) != len(adjacency)


def converges(net):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        return PowerFlow().solve_power_flow(net) is not None


def screen(week):
    limits = read_gen_limits(GEN_LIMITS_FILE)
    base_load_mw = baseline_load_mw()
    reference = prepare_network(1.0, sequential_dispatch(base_load_mw, limits), limits, False)
    labels = case_labels(reference.lines)
    ibr_mw = sum(site.operating_mw for site in SITES)

    rows = []
    for k, (line, ln) in enumerate(reference.lines.items(), 1):
        row = {'line': line, 'case': labels[line], 'from_bus': ln.bus_1, 'to_bus': ln.bus_2,
               'islands': islands(reference, line), 'n_hours': len(week),
               'n_converged_base': 0, 'n_converged_ibr': 0, 'failed_hours': ''}
        if not row['islands']:
            failed = []
            for r in week.itertuples(index=False):
                demand = base_load_mw * float(r.load_scale)
                ok_base = converges(outage_network(
                    r.load_scale, sequential_dispatch(demand, limits), limits, line))
                aug_net = outage_network(r.load_scale, sequential_dispatch(demand - ibr_mw, limits),
                                         limits, line)
                augment_network(aug_net)  # rebuilds the Y-bus
                ok_ibr = converges(aug_net)
                row['n_converged_base'] += ok_base
                row['n_converged_ibr'] += ok_ibr
                if not (ok_base and ok_ibr):
                    failed.append(int(r.hour_index))
            row['failed_hours'] = ';'.join(map(str, failed))
        row['included'] = (not row['islands'] and row['n_converged_base'] == len(week)
                           and row['n_converged_ibr'] == len(week))
        row['reason'] = ('islands a bus' if row['islands'] else
                         'power flow fails in some hours' if not row['included'] else '')
        print(f"[{k:02d}/{len(reference.lines)}] {row['case']}: "
              f"{'included' if row['included'] else 'excluded (' + row['reason'] + ')'}", flush=True)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--se3-file', type=Path, default=ROOT / 'data/se3_2025.csv')
    parser.add_argument('--hours', type=int, default=168)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    week = select_week(read_se3(args.se3_file, 'timestamp', 'SE3')).iloc[:args.hours]
    table = screen(week)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"\n{int(table['included'].sum())}/{len(table)} lines included; saved {args.output}")


if __name__ == '__main__':
    main()
