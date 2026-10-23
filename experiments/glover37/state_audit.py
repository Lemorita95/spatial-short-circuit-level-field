"""Per-state audit columns for run summaries: voltage-band flags and control actions.

Voltage violations are reported, never corrected (no curtailment in this study).
Columns are flat so they append directly to a run_summary.csv row; `prefix`
separates several networks solved for the same state (e.g. "base_", "ibr_").
"""
import numpy as np


def _join(items):
    return ';'.join(str(x) for x in items)


def operating_state_audit(net, pf, v_pu, limits, prefix=''):
    '''
        net     solved CustomNetwork
        pf      the PowerFlow instance that solved it (pv_state, control records)
        v_pu    |V| per bus, indexed like net.idx
        limits  VoltageLimits
    '''
    v_pu = np.asarray(v_pu, dtype=float)
    bus_of = {i: b for b, i in net.idx.items()}
    violations = limits.violations(net, v_pu)
    under = [x for x in violations if x.deviation_pu < 0]
    over = [x for x in violations if x.deviation_pu > 0]
    worst = violations[0] if violations else None
    pv_state = getattr(pf, 'pv_state', {}) or {}
    steps = getattr(pf, 'control_steps', {}) or {}

    audit = {
        'voltage_flag': bool(violations),
        'n_under_v': len(under),
        'n_over_v': len(over),
        'under_v_buses': _join(x.bus for x in under),
        'over_v_buses': _join(x.bus for x in over),
        'worst_violation_bus': worst.bus if worst else '',
        'worst_violation_pu': worst.deviation_pu if worst else 0.0,
        'v_min_pu': float(v_pu.min()),
        'v_min_bus': bus_of[int(v_pu.argmin())],
        'v_max_pu': float(v_pu.max()),
        'v_max_bus': bus_of[int(v_pu.argmax())],
        'unclassified_buses': _join(limits.unclassified(net.buses)),
        'tap_steps_total': int(sum(abs(s) for s in steps.values())),
        'tap_steps': _join(f'{name.split(".", 1)[-1]}:{s:+d}' for name, s in steps.items() if s),
        'control_passes': int(getattr(pf, 'control_passes', 0) or 0),
        'frozen_controllers': _join(getattr(pf, 'frozen_controllers', []) or []),
        'gens_at_qmax': _join(b for b, s in pv_state.items() if s == 1),
        'gens_at_qmin': _join(b for b, s in pv_state.items() if s == -1),
    }
    return {f'{prefix}{k}': v for k, v in audit.items()}
