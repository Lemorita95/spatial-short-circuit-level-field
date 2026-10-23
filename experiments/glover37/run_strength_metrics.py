"""Temporal SCR/WSCR/ESCR/SDSCR experiment; keeps the validated SCL run intact.

Run from repository root:
    python -m experiments.glover37.run_strength_metrics --hours 2
    python -m experiments.glover37.run_strength_metrics --with-outage

Outputs under results/glover37/strength/ by default. `metrics_timeseries.csv`
contains one row per metric / evaluation bus / time / scenario. `metric_variables.csv`
records the underlying voltage, SCMVA, impedance, IF, and power terms.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from models.PowerFlow import PowerFlow
from models.SystemStrength import SCR, WSCR, ESCR, SDSCR
from models.VoltageLimits import PRESETS, VoltageLimits
from experiments.glover37.state_audit import operating_state_audit
from experiments.glover37.strength_config import SITES, augment_network
from experiments.glover37.run_temporal import (
    GEN_LIMITS_FILE, CONTINGENCY_CASE, ROOT, S_BASE_MVA,
    baseline_load_mw, prepare_network, read_gen_limits,
    read_se3, select_week, sequential_dispatch,
)

DEFAULT_OUTPUT = ROOT / 'results' / 'glover37' / 'strength'


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


def evaluate_temporal(week, *, with_outage=False, out_dir=DEFAULT_OUTPUT, voltage_limits=None):
    # voltage bands only flag states in run_summary.csv; nothing is corrected
    voltage_limits = voltage_limits or VoltageLimits.preset()
    limits = read_gen_limits(GEN_LIMITS_FILE)
    base_load_mw = baseline_load_mw()
    ibr_mw = sum(site.operating_mw for site in SITES)
    scenarios = [('intact', False)]
    if with_outage:
        scenarios.append((CONTINGENCY_CASE, True))

    # One persistent instance per metric: all time steps go into .history.
    metrics = {
        'SCR': SCR({site.poi_bus: site.pmax_mw for site in SITES}),
        'WSCR': WSCR({site.poi_bus: site.operating_mw for site in SITES}),
        'ESCR': ESCR({site.escr_bus: site.pmax_mw for site in SITES}),
        'SDSCR': SDSCR({site.sdscr_bus: site.operating_mw for site in SITES}),
    }
    results, variables, statuses, dispatch_rows = [], [], [], []
    for row in week.itertuples(index=False):
        h, timestamp = int(row.hour_index), row.timestamp
        scale = float(row.load_scale)
        demand = base_load_mw * scale
        # Crucial: independent schedules, both using existing commitment rule.
        base_dispatch = sequential_dispatch(demand, limits)
        ibr_dispatch = sequential_dispatch(demand - ibr_mw, limits)

        for case, outage in scenarios:
            print(f'[{h + 1:03d}/{len(week):03d}] {timestamp} case={case}', flush=True)
            for network_name, dispatch in [('base', base_dispatch), ('ibr', ibr_dispatch)]:
                for gen in limits:
                    dispatch_rows.append({
                        'hour_index': h, 'timestamp': str(timestamp), 'case': case,
                        'network': network_name, 'generator': gen.key,
                        'bus': gen.bus, 'committed': int(dispatch[gen.key] > 0),
                        'dispatch_MW': dispatch[gen.key],
                        'system_demand_MW': demand,
                        'ibr_injection_MW': ibr_mw if network_name == 'ibr' else 0,
                    })
            base_net = prepare_network(scale, base_dispatch, limits, outage)
            aug_net = prepare_network(scale, ibr_dispatch, limits, outage)
            augment_network(aug_net)
            step = f'{case}:{h}'
            state = {
                'hour_index': h, 'timestamp': str(timestamp), 'case': case,
                'load_scale': scale, 'system_demand_MW': demand,
                'ibr_injection_MW': ibr_mw,
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
            except Exception as exc:
                state['error'] = str(exc)
                statuses.append(state)
                # Fail visibly; never silently treat missing hours as valid.
                write_outputs(out_dir, results, variables, statuses, dispatch_rows)
                raise RuntimeError(f'Failed at hour={h}, case={case}: {exc}') from exc
            statuses.append(state)

    write_outputs(out_dir, results, variables, statuses, dispatch_rows)
    return metrics


def write_outputs(out_dir, results, variables, statuses, dispatch_rows):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=['hour_index','timestamp','case','metric','bus','value']).to_csv(
        out_dir / 'metrics_timeseries.csv', index=False)
    pd.DataFrame(variables, columns=['hour_index','timestamp','case','metric','bus',
                                     'related_bus','variable','unit','real','imag','magnitude']).to_csv(
        out_dir / 'metric_variables.csv', index=False)
    pd.DataFrame(statuses).to_csv(out_dir / 'run_summary.csv', index=False)
    pd.DataFrame(dispatch_rows).to_csv(out_dir / 'dispatch_schedule.csv', index=False)
    # Offline report: embedded data means it works when opened via file://.
    template = Path(__file__).with_name('strength_timeseries_explorer.html')
    if template.is_file():
        import json
        page = template.read_text(encoding='utf-8')
        page = page.replace('/*EMBED_METRICS*/[]',
                            json.dumps(results, default=str).replace('</', '<\\/'))
        page = page.replace('/*EMBED_VARIABLES*/[]',
                            json.dumps(variables, default=str).replace('</', '<\\/'))
        (out_dir / 'strength_timeseries_explorer.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--se3-file', type=Path, default=ROOT / 'data/se3_2025.csv')
    parser.add_argument('--hours', type=int, default=168,
                        help='Run first N hours of the established 168-hour window (smoke test: 2).')
    parser.add_argument('--with-outage', action='store_true',
                        help='Also evaluate the established line 39-47 contingency.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--voltage-preset', default='default', choices=sorted(PRESETS),
                        help='VoltageLimits preset used to flag out-of-band states.')
    args = parser.parse_args()
    if not 1 <= args.hours <= 168:
        parser.error('--hours must be between 1 and 168')
    week = select_week(read_se3(args.se3_file, 'timestamp', 'SE3')).iloc[:args.hours]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(tempfile.mkdtemp(prefix='.strength-run-', dir=output.parent))
    try:
        metrics = evaluate_temporal(week, with_outage=args.with_outage, out_dir=temp_output,
                                    voltage_limits=VoltageLimits.preset(args.voltage_preset))
        # Replace the published result set only after every requested state has
        # completed successfully. Failed runs leave the previous full result set intact.
        output.mkdir(parents=True, exist_ok=True)
        for filename in (
            'metrics_timeseries.csv', 'metric_variables.csv', 'run_summary.csv',
            'dispatch_schedule.csv', 'strength_timeseries_explorer.html',
        ):
            source = temp_output / filename
            if source.exists():
                source.replace(output / filename)
        print('\nSaved results to', output)
        print('History records:', {k: len(m.history) for k, m in metrics.items()})
    finally:
        shutil.rmtree(temp_output, ignore_errors=True)


if __name__ == '__main__':
    main()
