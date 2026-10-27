"""Simulation stage: N-1 screening and the temporal SCR/WSCR/ESCR/SDSCR run.

`screen` fixes the N-1 set before any metric is computed: a line outage is
included only if it islands no bus and both pre-fault power flows (37-bus base,
46-bus with IBRs) converge in every hour of the week, with commitment and
setpoints held at the intact schedule. No strength metric is evaluated there.

`run` evaluates the four metrics for the intact network and every included
line outage, hour by hour.

Run from repository root:
    python -m experiments.glover37.simulate screen
    python -m experiments.glover37.simulate screen --hours 2        # smoke test
    python -m experiments.glover37.simulate run --hours 2
    python -m experiments.glover37.simulate run \
        --contingencies results/glover37/raw/contingency_set.csv
    # intact + line 39-47 only (the v1.0 results, c263896):
    python -m experiments.glover37.simulate run \
        --contingencies results/glover37/raw/contingency_set.csv --case line_39_47_out

Outputs under results/glover37/raw/ by default:
- `contingency_set.csv` (screen): one row per line;
- `metrics_timeseries.csv`: one row per metric / evaluation bus / time / scenario;
- `metric_variables.csv`: the underlying voltage, SCMVA, impedance, IF and power terms;
- `run_summary.csv`, `dispatch_schedule.csv`: per-state audit and dispatch;
- `z_full_intact.npz`: the full short-circuit impedance matrix Z = Y_SC^-1 of
  every intact state, for both networks (needed to explain outage effects).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from models.PowerFlow import PowerFlow
from models.SystemStrength import SCR, WSCR, ESCR, SDSCR, short_circuit_admittance
from models.VoltageLimits import PRESETS, VoltageLimits
from experiments.glover37.config import (
    GEN_LIMITS_FILE, IBR_MW, IF_STEP, INTACT, RAW_DIR, SE3_FILE, SITES, VOLTAGE_PRESET,
    WINDOW_HOURS,
)
from experiments.glover37.network import (
    augment_network, baseline_load_mw, operating_state_audit, prepare_network,
    read_gen_limits, read_se3, select_week, sequential_dispatch,
)

CONTINGENCY_FILE = RAW_DIR / 'contingency_set.csv'
RUN_FILES = ('metrics_timeseries.csv', 'metric_variables.csv', 'run_summary.csv',
             'dispatch_schedule.csv', 'z_full_intact.npz')


def load_week(se3_file=SE3_FILE, hours=WINDOW_HOURS):
    return select_week(read_se3(se3_file)).iloc[:hours]


# =============================================================================
# 1. N-1 screening
# =============================================================================

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
    reference = prepare_network(1.0, sequential_dispatch(base_load_mw, limits), limits)
    labels = case_labels(reference.lines)

    rows = []
    for k, (line, ln) in enumerate(reference.lines.items(), 1):
        row = {'line': line, 'case': labels[line], 'from_bus': ln.bus_1, 'to_bus': ln.bus_2,
               'islands': islands(reference, line), 'n_hours': len(week),
               'n_converged_base': 0, 'n_converged_ibr': 0, 'failed_hours': ''}
        if not row['islands']:
            failed = []
            for r in week.itertuples(index=False):
                demand = base_load_mw * float(r.load_scale)
                ok_base = converges(prepare_network(
                    r.load_scale, sequential_dispatch(demand, limits), limits, line))
                aug_net = prepare_network(r.load_scale, sequential_dispatch(demand - IBR_MW, limits),
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


def read_contingencies(path, cases=None):
    """[(case label, line key)] for the included rows of contingency_set.csv."""
    table = pd.read_csv(path)
    table = table[table['included'].astype(bool)]
    if cases:
        if unknown := set(cases) - set(table['case']):
            raise ValueError(f'Cases not in the included contingency set: {sorted(unknown)}')
        table = table[table['case'].isin(cases)]
    return list(zip(table['case'], table['line']))


# =============================================================================
# 2. Temporal metrics run
# =============================================================================

def solve_pf(network):
    '''complex bus voltages and the PowerFlow instance (generator states, control records).'''
    pf = PowerFlow()
    solved = pf.solve_power_flow(network)
    if solved is None:
        raise RuntimeError('Power flow failed')
    theta, magnitude = solved
    return magnitude * np.exp(1j * theta), pf


def scalar_components(value):
    """Serialize a complex number as numeric real/imag/magnitude fields."""
    z = complex(value)
    return {'real': float(z.real), 'imag': float(z.imag), 'magnitude': float(abs(z))}


def append_record(rows, *, hour, timestamp, case, metric, bus, variable,
                  value, related_bus=None, unit=''):
    numeric = scalar_components(value)
    rows.append({
        'hour_index': int(hour), 'timestamp': str(timestamp), 'case': case,
        'metric': metric, 'bus': bus, 'related_bus': related_bus,
        'variable': variable, 'unit': unit, **numeric,
    })


def emit_metric(rows, details, metric_name, record, hour, timestamp, case):
    values = record['value']
    items = values.items() if isinstance(values, dict) else [(None, values)]
    for bus, value in items:
        rows.append({
            'hour_index': int(hour), 'timestamp': str(timestamp), 'case': case,
            'metric': metric_name, 'bus': bus if bus is not None else 'group',
            'value': float(value),
        })
    # Keep the inputs and intermediate terms needed to explain temporal changes.
    for key, unit in [('scmva', 'MVA'), ('rated_mw', 'MW'),
                      ('operating_mw', 'MW'), ('probe_reactive_power_mvar', 'MVAr'),
                      ('probe_pv_changes', 'count'),
                      ('target_voltage_pu', 'pu'), ('actual_voltage_pu', 'pu'),
                      ('z_ii', 'pu'), ('v_pf', 'pu')]:
        for bus, val in record.get(key, {}).items():
            append_record(details, hour=hour, timestamp=timestamp, case=case,
                          metric=metric_name, bus=bus, variable=key,
                          value=val, unit=unit)
    for key in ('if_magnitude', 'w_complex'):
        for (bus, other), val in record.get(key, {}).items():
            append_record(details, hour=hour, timestamp=timestamp, case=case,
                          metric=metric_name, bus=bus, related_bus=other,
                          variable=key, value=val, unit='ratio')
    for disturbed, responses in record.get('delta_v_pu', {}).items():
        for measured, dv in responses.items():
            append_record(details, hour=hour, timestamp=timestamp, case=case,
                          metric=metric_name, bus=disturbed,
                          related_bus=measured, variable='delta_v_pu',
                          value=dv, unit='pu')
    if 'z_rr' in record:
        buses = tuple(record['operating_mw'])
        for i, bus in enumerate(buses):
            for j, other in enumerate(buses):
                append_record(details, hour=hour, timestamp=timestamp,
                              case=case, metric=metric_name, bus=bus,
                              related_bus=other, variable='z_rr',
                              value=record['z_rr'][i, j], unit='pu')


def full_z(net, v_pf):
    '''Z = Y_SC^-1 over every bus, rows/columns in net.idx order.'''
    return np.linalg.inv(short_circuit_admittance(net, v_pf).toarray())


def evaluate_temporal(week, *, contingencies=(), out_dir=RAW_DIR,
                      voltage_limits=None, if_step=IF_STEP):
    # voltage bands only flag states in run_summary.csv; nothing is corrected
    voltage_limits = voltage_limits or VoltageLimits.preset()
    limits = read_gen_limits(GEN_LIMITS_FILE)
    base_load_mw = baseline_load_mw()
    scenarios = [(INTACT, None)] + [(case, line.lower()) for case, line in contingencies]
    z_full = {'hour_index': [], 'base': [], 'ibr': [], 'base_buses': None, 'ibr_buses': None}

    # One persistent instance per metric: all time steps go into .history.
    metrics = {
        'SCR': SCR({site.poi_bus: site.pmax_mw for site in SITES}),
        'WSCR': WSCR({site.poi_bus: site.operating_mw for site in SITES}),
        'ESCR': ESCR({site.escr_bus: site.pmax_mw for site in SITES},
                     relative_voltage_step=if_step),
        'SDSCR': SDSCR({site.sdscr_bus: site.operating_mw for site in SITES}),
    }
    results, variables, statuses, dispatch_rows = [], [], [], []
    for row in week.itertuples(index=False):
        h, timestamp = int(row.hour_index), row.timestamp
        scale = float(row.load_scale)
        demand = base_load_mw * scale
        # Crucial: independent schedules, both using existing commitment rule.
        base_dispatch = sequential_dispatch(demand, limits)
        ibr_dispatch = sequential_dispatch(demand - IBR_MW, limits)

        for case, line in scenarios:
            print(f'[{h + 1:03d}/{len(week):03d}] {timestamp} case={case}', flush=True)
            for network_name, dispatch in [('base', base_dispatch), ('ibr', ibr_dispatch)]:
                for gen in limits:
                    dispatch_rows.append({
                        'hour_index': h, 'timestamp': str(timestamp), 'case': case,
                        'network': network_name, 'generator': gen.key,
                        'bus': gen.bus, 'committed': int(dispatch[gen.key] > 0),
                        'dispatch_MW': dispatch[gen.key],
                        'system_demand_MW': demand,
                        'ibr_injection_MW': IBR_MW if network_name == 'ibr' else 0,
                    })
            base_net = prepare_network(scale, base_dispatch, limits, line)
            aug_net = prepare_network(scale, ibr_dispatch, limits, line)
            augment_network(aug_net)
            step = f'{case}:{h}'
            state = {
                'hour_index': h, 'timestamp': str(timestamp), 'case': case,
                'load_scale': scale, 'system_demand_MW': demand,
                'ibr_injection_MW': IBR_MW,
                'base_non_slack_MW': sum(base_dispatch.values()),
                'ibr_case_non_slack_MW': sum(ibr_dispatch.values()),
                'base_n_committed': sum(x > 0 for x in base_dispatch.values()),
                'ibr_case_n_committed': sum(x > 0 for x in ibr_dispatch.values()),
                'base_pf_ok': False, 'ibr_pf_ok': False,
                'error': '',
            }
            try:
                base_v, base_pf = solve_pf(base_net)
                state['base_pf_ok'] = True
                state.update(operating_state_audit(base_net, base_pf, np.abs(base_v),
                                                   voltage_limits, prefix='base_'))
                aug_v, aug_pf = solve_pf(aug_net)
                state['ibr_pf_ok'] = True
                state.update(operating_state_audit(aug_net, aug_pf, np.abs(aug_v),
                                                   voltage_limits, prefix='ibr_'))
                for name, net, voltage in [
                    ('SCR', base_net, base_v), ('WSCR', base_net, base_v),
                    ('ESCR', aug_net, aug_v), ('SDSCR', aug_net, aug_v),
                ]:
                    try:
                        record = metrics[name].evaluate(step, net, voltage)
                        emit_metric(results, variables, name, record, h, timestamp, case)
                    except Exception as exc:
                        raise RuntimeError(f'{name}: {exc}') from exc
                if line is None:
                    z_full['hour_index'].append(h)
                    for key, net, voltage in [('base', base_net, base_v), ('ibr', aug_net, aug_v)]:
                        z_full[key].append(full_z(net, voltage))
                        z_full[f'{key}_buses'] = sorted(net.idx, key=net.idx.get)
            except Exception as exc:
                state['error'] = str(exc)
                statuses.append(state)
                # Fail visibly; never silently treat missing hours as valid.
                write_outputs(out_dir, results, variables, statuses, dispatch_rows, z_full)
                raise RuntimeError(f'Failed at hour={h}, case={case}: {exc}') from exc
            statuses.append(state)

    write_outputs(out_dir, results, variables, statuses, dispatch_rows, z_full)
    return metrics


def write_outputs(out_dir, results, variables, statuses, dispatch_rows, z_full=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=['hour_index','timestamp','case','metric','bus','value']).to_csv(
        out_dir / 'metrics_timeseries.csv', index=False)
    pd.DataFrame(variables, columns=['hour_index','timestamp','case','metric','bus',
                                     'related_bus','variable','unit','real','imag','magnitude']).to_csv(
        out_dir / 'metric_variables.csv', index=False)
    pd.DataFrame(statuses).to_csv(out_dir / 'run_summary.csv', index=False)
    pd.DataFrame(dispatch_rows).to_csv(out_dir / 'dispatch_schedule.csv', index=False)
    if z_full and z_full['hour_index']:
        np.savez_compressed(
            out_dir / 'z_full_intact.npz', hour_index=np.asarray(z_full['hour_index']),
            base_buses=np.asarray(z_full['base_buses']), ibr_buses=np.asarray(z_full['ibr_buses']),
            z_base=np.stack(z_full['base']), z_ibr=np.stack(z_full['ibr']))


def run(week, output, *, contingencies=(), voltage_limits=None, if_step=IF_STEP):
    """evaluate_temporal() into a temporary directory, then publish to `output`.

    The published result set is replaced only after every requested state has
    completed successfully. Failed runs leave the previous full result set intact.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(tempfile.mkdtemp(prefix='.strength-run-', dir=output.parent))
    try:
        metrics = evaluate_temporal(week, contingencies=contingencies, out_dir=temp_output,
                                    voltage_limits=voltage_limits, if_step=if_step)
        output.mkdir(parents=True, exist_ok=True)
        for filename in RUN_FILES:
            source = temp_output / filename
            if source.exists():
                source.replace(output / filename)
        print('\nSaved results to', output)
        print('History records:', {k: len(m.history) for k, m in metrics.items()})
        return metrics
    finally:
        shutil.rmtree(temp_output, ignore_errors=True)


# =============================================================================
# 3. Command line
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('screen', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--se3-file', type=Path, default=SE3_FILE)
        p.add_argument('--hours', type=int, default=WINDOW_HOURS,
                       help='Run first N hours of the established 168-hour window (smoke test: 2).')
    screen_p, run_p = sub.choices['screen'], sub.choices['run']
    screen_p.add_argument('--output', type=Path, default=CONTINGENCY_FILE)
    run_p.add_argument('--contingencies', type=Path,
                       help='contingency_set.csv: also evaluate every included line outage.')
    run_p.add_argument('--case', action='append', dest='cases',
                       help='With --contingencies: evaluate only this case label (repeatable).')
    run_p.add_argument('--output', type=Path, default=RAW_DIR)
    run_p.add_argument('--voltage-preset', default=VOLTAGE_PRESET, choices=sorted(PRESETS),
                       help='VoltageLimits preset used to flag out-of-band states.')
    run_p.add_argument('--if-step', type=float, default=IF_STEP,
                       help='ESCR interaction-factor probe: relative MV voltage step '
                            f'(default {IF_STEP}).')
    args = parser.parse_args()
    if not 1 <= args.hours <= WINDOW_HOURS:
        parser.error(f'--hours must be between 1 and {WINDOW_HOURS}')
    week = load_week(args.se3_file, args.hours)

    if args.stage == 'screen':
        table = screen(week)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\n{int(table['included'].sum())}/{len(table)} lines included; saved {args.output}")
        return

    if args.cases and not args.contingencies:
        parser.error('--case needs --contingencies')
    contingencies = read_contingencies(args.contingencies, args.cases) if args.contingencies else ()
    run(week, args.output, contingencies=contingencies,
        voltage_limits=VoltageLimits.preset(args.voltage_preset), if_step=args.if_step)


if __name__ == '__main__':
    main()
