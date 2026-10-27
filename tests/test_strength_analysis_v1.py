"""Synthetic checks of analyze_strength_metrics_v1 (stdlib unittest).

The synthetic results are written by the runner's own writers (emit_metric,
write_outputs, operating_state_audit), so they follow the production CSV schema.
Every metric value is computed from its recorded terms with the file 02 formulas,
and the planted features below give known answers:

    f(h)   common strength shape: 3 + sin(2*pi*h/24) + 0.002 h, with a dip f(100) = 1
    outage factor per site 0.95 / 0.97 / 0.99 (sites 21 / 48 / 16), on every metric
    outage, site 21, 37-bus POI: |V| = 0.9, 0.6, 0.25, 0.9 at hours 95-98 (look-2 shape),
        so SCR at site 21 has t* = 97 while every other series has t* = 100
    intact: IF = 0 at hours 10-19, probe PV/PQ switches at hours 30-34,
        IBR-bus over-voltage at hours 50-56; outage: 37-bus under-voltage at 95-98
    weakest site: 21 by SCR and ESCR, 16 by SDSCR

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.glover37 import analyze_strength_metrics_v1 as asm
from experiments.glover37.run_strength_metrics import emit_metric, write_outputs
from experiments.glover37.run_temporal import CONTINGENCY_CASE
from experiments.glover37.state_audit import operating_state_audit
from experiments.glover37.strength_config import SITES

N_HOURS = 168
DIP_HOUR, V_DIP = 100, {95: 0.9, 96: 0.6, 97: 0.25, 98: 0.9}
OUT_FACTOR = {21: 0.95, 48: 0.97, 16: 0.99}
SCR_C = {21: 1600.0, 48: 2500.0, 16: 1900.0}
ESCR_C = {21: 450.0, 48: 400.0, 16: 420.0}
SDSCR_Z0 = {21: 0.08, 48: 0.09, 16: 0.20}
SDSCR_ANGLE = {21: 0.10, 48: 0.05, 16: 0.02}
GENS = ['generator.g10-1', 'generator.g12-1', 'generator.g44-1',
        'generator.g48-1', 'generator.g53-1']
IF_ZERO_HOURS, PROBE_HOURS, OVER_V_HOURS = range(10, 20), range(30, 35), range(50, 57)


def f(h):
    return 1.0 if h == DIP_HOUR else 3 + np.sin(2 * np.pi * h / 24) + 0.002 * h


def load_scale(h):
    return 1 + 0.25 * np.sin(2 * np.pi * (h - 6) / 24) + 0.0005 * h


def committed(h, network):
    lam = load_scale(h)
    n = 3 if lam < 0.9 else 4 if lam < 1.1 else 5
    n -= network == 'ibr'
    gens = GENS[:n]
    if n == (2 if network == 'ibr' else 3) and h % 2:  # same count, another set
        gens = gens[:-1] + [GENS[-1]]
    return set(gens)


class _Violation:
    def __init__(self, bus, deviation):
        self.bus, self.deviation_pu = bus, deviation


class _Limits:
    def __init__(self, violations):
        self._violations = violations

    def violations(self, net, v_pu):
        return self._violations

    def unclassified(self, buses):
        return []


def audit(prefix, violations):
    net = SimpleNamespace(idx={1: 0, 2: 1}, buses={1: None, 2: None})
    pf = SimpleNamespace(pv_state={}, control_steps={}, control_passes=1, frozen_controllers=[])
    return operating_state_audit(net, pf, np.array([1.0, 0.98]), _Limits(violations), prefix)


def records(h, case):
    out = case != 'intact'
    k = {s.host_bus: (OUT_FACTOR[s.host_bus] if out else 1.0) for s in SITES}

    # SCR / WSCR: 37-bus POI, SCMVA = 100 |V| / |Z_ii|
    v_poi, z_poi, scmva = {}, {}, {}
    for s in SITES:
        v = V_DIP.get(h, 1.0) if (out and s.host_bus == 21) else 1.0
        z = 100.0 / (SCR_C[s.host_bus] * f(h) * k[s.host_bus]) * 1j
        v_poi[s.poi_bus], z_poi[s.poi_bus] = complex(v), z
        scmva[s.poi_bus] = 100.0 * abs(v) / abs(z)
    rated = {s.poi_bus: s.pmax_mw for s in SITES}
    op = {s.poi_bus: s.operating_mw for s in SITES}
    scr = {'value': {b: scmva[b] / rated[b] for b in rated}, 'scmva': scmva,
           'v_pf': v_poi, 'z_ii': z_poi, 'rated_mw': rated}
    wscr = {'value': sum(scmva[b] * op[b] for b in op) / sum(op.values()) ** 2,
            'scmva': scmva, 'operating_mw': op, 'z_ii': z_poi}

    # ESCR: MV padmount-HV buses, IF_ji = observe j when disturbing i
    mv = [s.escr_bus for s in SITES]
    rated_mv = {s.escr_bus: s.pmax_mw for s in SITES}
    scmva_mv = {s.escr_bus: ESCR_C[s.host_bus] * f(h) * k[s.host_bus] for s in SITES}
    if_value = 1e-6 if (not out and h in IF_ZERO_HOURS) else 0.1
    interaction = {(j, i): if_value for i in mv for j in mv if j != i}
    escr = {
        'value': {i: scmva_mv[i] / (rated_mv[i] + sum(interaction[(j, i)] * rated_mv[j]
                                                       for j in mv if j != i)) for i in mv},
        'scmva': scmva_mv, 'if_magnitude': interaction,
        'probe_reactive_power_mvar': {i: 5.5 for i in mv},
        'probe_pv_changes': {i: (2 if (not out and h in PROBE_HOURS and i == mv[0]) else 0)
                             for i in mv},
        'target_voltage_pu': {i: 1.01 for i in mv}, 'actual_voltage_pu': {i: 1.01 for i in mv},
        'delta_v_pu': {i: {j: 0.01 if j == i else 0.01 * interaction[(j, i)] for j in mv}
                       for i in mv},
        'rated_mw': rated_mv, 'v_pf': {i: 1.0 + 0j for i in mv},
        'z_ii': {i: 100.0 / scmva_mv[i] * 1j for i in mv},
    }

    # SDSCR: LV converter terminals, Wu (2018) with complex weights
    lv = [s.sdscr_bus for s in SITES]
    p = {s.sdscr_bus: s.operating_mw for s in SITES}
    v_lv = {s.sdscr_bus: np.exp(1j * SDSCR_ANGLE[s.host_bus]) for s in SITES}
    z0 = np.full((3, 3), 0.005 + 0.01j)
    np.fill_diagonal(z0, [SDSCR_Z0[s.host_bus] * 1j for s in SITES])
    z_rr = z0 / (f(h) * np.array([k[s.host_bus] for s in SITES]))[:, None]
    z_rr = (z_rr + z_rr.T) / 2
    w, sdscr_value = {}, {}
    for a, i in enumerate(lv):
        eff = p[i] / 100
        for b, j in enumerate(lv):
            if i != j:
                w[(i, j)] = z_rr[a, b] / z_rr[a, a] * np.conjugate(v_lv[i] / v_lv[j])
                eff += w[(i, j)] * p[j] / 100
        sdscr_value[i] = abs(v_lv[i]) ** 2 / (abs(z_rr[a, a]) * abs(eff))
    sdscr = {'value': sdscr_value, 'w_complex': w, 'v_pf': v_lv, 'z_rr': z_rr,
             'operating_mw': p}
    return {'SCR': scr, 'WSCR': wscr, 'ESCR': escr, 'SDSCR': sdscr}


def write_synthetic(out_dir):
    results, variables, statuses, dispatch_rows = [], [], [], []
    for h in range(N_HOURS):
        timestamp = pd.Timestamp('2025-01-06') + pd.Timedelta(hours=h)
        for case in ('intact', CONTINGENCY_CASE):
            out = case != 'intact'
            sets = {net: committed(h, net) for net in ('base', 'ibr')}
            for net, members in sets.items():
                for g in GENS:
                    dispatch_rows.append({
                        'hour_index': h, 'timestamp': str(timestamp), 'case': case,
                        'network': net, 'generator': g, 'bus': int(g[11:13]),
                        'committed': int(g in members),
                        'dispatch_MW': 100.0 * (g in members),
                        'system_demand_MW': 1000 * load_scale(h),
                        'ibr_injection_MW': 120 if net == 'ibr' else 0})
            base_v = [_Violation(21, -0.013)] if out and h in V_DIP else []
            ibr_v = [_Violation(2103, 0.009)] if not out and h in OVER_V_HOURS else []
            statuses.append({
                'hour_index': h, 'timestamp': str(timestamp), 'case': case,
                'load_scale': load_scale(h), 'system_demand_MW': 1000 * load_scale(h),
                'ibr_injection_MW': 120, 'base_non_slack_MW': 0, 'ibr_case_non_slack_MW': 0,
                'base_n_committed': len(sets['base']), 'ibr_case_n_committed': len(sets['ibr']),
                'base_pf_ok': True, 'ibr_pf_ok': True, 'error': '',
                **audit('base_', base_v), **audit('ibr_', ibr_v)})
            for name, record in records(h, case).items():
                emit_metric(results, variables, name, record, h, timestamp, case)
    write_outputs(out_dir, results, variables, statuses, dispatch_rows)


def analyze(raw, analysis):
    with contextlib.redirect_stdout(io.StringIO()):
        return asm.run(raw, analysis)


class TestHelpers(unittest.TestCase):

    def test_set_sizes(self):
        self.assertEqual(asm.set_size(0.10, 168), 17)
        self.assertEqual(asm.set_size(0.05, 168), 9)
        self.assertEqual(asm.set_size(0.20, 168), 34)
        self.assertEqual(asm.set_size(0.05, 167), 9)

    def test_ties_enlarge_sets(self):
        s = pd.Series([1.0, 1.0 + 1e-12, 2.0, 2.0, 3.0])
        self.assertEqual(asm.t_star(s), [0, 1])
        self.assertEqual(asm.lowest(s, 3), {0, 1, 2, 3})


class TestSyntheticAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls._tmp.name) / 'raw'
        write_synthetic(cls.raw)
        cls.out = analyze(cls.raw, Path(cls._tmp.name) / 'analysis')

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def summary(self, criterion, question):
        s = self.out['summary']
        return s[(s['criterion'] == criterion) & (s['question'] == question)]['result'].item()

    def test_tables_written(self):
        written = {p.stem for p in (Path(self._tmp.name) / 'analysis').glob('*.csv')}
        self.assertEqual(written, set(self.out))

    def test_every_table_and_result_is_traced(self):
        index = self.out['index']
        self.assertEqual(set(index['table']), set(self.out))
        s = self.out['summary']
        self.assertFalse(s['rq'].eq('').any() or s['question'].eq('').any())
        self.assertTrue(set(s['table']) - {''} <= set(self.out))

    def test_annotations(self):
        ann = self.out['annotations'].set_index('case')
        self.assertEqual(int(ann.loc['intact', 'IF_zero'].sum()), len(IF_ZERO_HOURS))
        self.assertEqual(int(ann.loc[CONTINGENCY_CASE, 'IF_zero'].sum()), 0)
        self.assertEqual(int(ann.loc['intact', 'probe_PV_PQ_switch'].sum()), len(PROBE_HOURS))
        self.assertEqual(int(ann.loc[CONTINGENCY_CASE, 'base_under_v'].sum()), len(V_DIP))
        self.assertEqual(int(ann.loc['intact', 'ibr_over_v'].sum()), len(OVER_V_HOURS))
        self.assertEqual(int(ann['voltage_flag'].sum()), len(V_DIP) + len(OVER_V_HOURS))

    def test_t_star(self):
        t = self.out['d1_t_star'].set_index(['case', 'metric', 'site'])['t_star']
        self.assertEqual(t[CONTINGENCY_CASE, 'SCR', '21'], '97')
        self.assertTrue((t.drop((CONTINGENCY_CASE, 'SCR', '21')) == str(DIP_HOUR)).all())

    def test_c1_detects_planted_disagreement(self):
        d1 = self.out['d1_pairs']
        primary = d1[(d1['analysis'] == 'primary') & d1['informative']]
        intact = primary[primary['case'] == 'intact']
        self.assertTrue(intact['C1_agree'].all())
        failing = primary[~primary['C1_agree']]
        self.assertEqual(
            set(map(tuple, failing[['case', 'site', 'metric_m', 'metric_n']].to_numpy())),
            {(CONTINGENCY_CASE, '21', 'SCR', 'ESCR'), (CONTINGENCY_CASE, '21', 'SCR', 'SDSCR')})
        self.assertEqual(set(failing['abs_t_star_diff_h']), {3})
        self.assertEqual(
            self.summary('C1', 'H0: does every informative pair agree on the weakest hour?'), False)
        self.assertEqual(self.summary(
            'C1', f'Informative pairs agreeing on the weakest hour [{CONTINGENCY_CASE}]'), '13/15')
        self.assertTrue((primary['size_K_m'] == 17).all())

    def test_c4(self):
        m = self.out['d4_c4_metrics'].set_index(['metric', 'site'])
        # SCR and ESCR scale exactly with the site factor. Synthetic SDSCR does not:
        # symmetrizing z_rr mixes the site factors into Z_ij, so only the sign is exact.
        for metric in ('SCR', 'ESCR'):
            for site in asm.SITE_IDS:
                self.assertAlmostEqual(m.loc[(metric, site), 'C4a_median_r'],
                                       OUT_FACTOR[int(site)] - 1, places=12)
        self.assertTrue((m.loc['SDSCR', 'share_r_negative'] == 1).all())
        moves = m['C4b_t_star_out_not_in_K_int']
        self.assertEqual(list(moves[moves].index), [('SCR', '21')])
        self.assertTrue(self.out['d4_c4c_site_ranking']['C4c_agree'].all())
        self.assertTrue((self.out['d4_c4c_site_ranking']['site_order_m'] == '21;48;16').all())

    def test_d5_gap(self):
        d5 = self.out['d5_snapshots'].set_index(['case', 'metric', 'site'])
        row = d5.loc[('intact', 'SCR', '21')]
        lam = [load_scale(h) for h in range(N_HOURS)]
        peak, light = int(np.argmax(lam)), int(np.argmin(lam))
        self.assertEqual((row['t_peak'], row['t_light']), (peak, light))
        expected = min(f(peak), f(light)) / f(DIP_HOUR) - 1
        self.assertAlmostEqual(row['D5b_g'], expected, places=12)
        self.assertFalse(row['D5a_snapshot_min_in_K'])

    def test_d2_events(self):
        d2 = self.out['d2_metrics']
        self.assertTrue((d2['size_E'] == 9).all())
        events = d2.set_index(['case', 'metric', 'site']).loc[('intact', 'SCR', '21'), 'E']
        self.assertTrue({DIP_HOUR, DIP_HOUR + 1} <= {int(x) for x in events.split(';')})

    def test_d3_weakest_site(self):
        pairs = self.out['d3_pairs'].set_index(['case', 'metric_m', 'metric_n'])
        share = pairs['share_same_weakest_site']
        self.assertEqual(share['intact', 'SCR', 'ESCR'], 1.0)
        self.assertEqual(share['intact', 'SCR', 'SDSCR'], 0.0)

    def test_a1_b_set(self):
        b = self.out['a1_b_set']
        self.assertEqual(len(b[(b['case'] == 'intact') & (b['network'] == 'base')]), 2)
        self.assertFalse(self.out['a1_within_b_set'].empty)

    def test_a2_numerator_keeps_disagreement(self):
        a2 = self.out['a2_pairs']
        dis = a2[a2['informative'] & ~a2['original_C1_agree']]
        self.assertEqual(len(dis), 2)
        self.assertFalse(dis['numerator_only_C1_agree'].any())

    def test_a2_c4_attribution(self):
        # every planted outage factor is < 1 and the intact IF ≈ 0 hours only raise
        # intact ESCR, so no metric rises under the outage in any variant
        m = self.out['a2_c4_metrics']
        for name in ('original', 'numerator_only', 'interaction_removed'):
            self.assertTrue((m[f'{name}_share_r_negative'] == 1).all())
        self.assertEqual(int(m['n_r_positive'].sum()), 0)
        p = self.out['a2_c4_pairs']
        self.assertTrue((p['original_C4a_share_same_sign_r'] == 1).all())

    def test_a3_invariance_and_reconstruction(self):
        self.assertTrue(self.out['a3_denominator_invariance']['unchanged'].all())
        self.assertLess(self.out['a3_reconstruction_check']['max_rel_diff'].max(), 1e-12)

    def test_a4_voltage_explains_shift(self):
        a4 = self.out['a4_summary'].iloc[0]
        self.assertEqual((a4['t_star'], a4['t_star_V_1pu']), ('97', str(DIP_HOUR)))
        self.assertTrue(a4['V_moves_t_star'])
        self.assertTrue(a4['t_star_base_under_v'])
        self.assertAlmostEqual(a4['dln_V_abs'], np.log(V_DIP[97]), places=12)
        self.assertAlmostEqual(a4['dln_SCMVA'], a4['dln_V_abs'] + a4['minus_dln_Z_ii_abs'],
                               places=12)
        pairs = self.out['a4_pairs'].set_index(['variant', 'metric_n'])
        self.assertFalse(pairs.loc[('original', 'ESCR'), 'C1_agree'])
        self.assertTrue(pairs.loc[('V_1pu', 'ESCR'), 'C1_agree'])


class TestInputGuards(unittest.TestCase):

    def test_output_must_differ_from_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                asm.run(tmp, tmp)

    def test_incomplete_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw'
            write_synthetic(raw)
            m = pd.read_csv(raw / 'metrics_timeseries.csv')
            m.iloc[:-1].to_csv(raw / 'metrics_timeseries.csv', index=False)
            with self.assertRaises(ValueError):
                analyze(raw, Path(tmp) / 'analysis')


if __name__ == '__main__':
    unittest.main()
