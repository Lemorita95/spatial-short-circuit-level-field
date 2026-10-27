"""Checks of the analysis stage (experiments.glover37.analyze), evidence plan v2 (stdlib unittest).

- planted answers for the D1 attribution and the exact log decomposition;
- the Woodbury branch update against direct inversion;
- an integration run on a fresh 3-hour simulation (intact, line 39-47 and both
  18-37 circuits) written to a temporary folder, never the results of record.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glover37 import analyze as asm
from experiments.glover37 import simulate as rsm
from experiments.glover37.config import GEN_LIMITS_FILE, SE3_FILE
from experiments.glover37.network import (
    baseline_load_mw, prepare_network, read_gen_limits, read_se3, select_week,
    sequential_dispatch,
)
from models.SystemStrength import short_circuit_admittance

LINES = {  # case -> (line key, from, to)
    'line_39_47_out': ('line.line 39-47 ckt 1', 39, 47),
    'line_18_37_ckt1_out': ('line.line 18-37 ckt 1', 18, 37),
    'line_18_37_ckt2_out': ('line.line 18-37 ckt 2', 18, 37),
}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def synthetic_values(n_hours, overrides):
    """values[variant][metric] with every series equal to `base`, except overrides
    {(variant, metric, site): array}."""
    base = np.linspace(2.0, 3.0, n_hours)
    index = pd.MultiIndex.from_product([['intact'], range(n_hours)], names=['case', 'hour_index'])
    values = {}
    for v in asm.VARIANTS:
        values[v] = {}
        for metric in asm.METRICS:
            cols = [asm.GROUP] if metric == 'WSCR' else list(asm.SITE_IDS)
            frame = pd.DataFrame({c: base for c in cols}, index=index)
            for (ov, om, os_), arr in overrides.items():
                if ov == v and om == metric:
                    frame[os_] = arr
            values[v][metric] = frame
    return values


class TestPlantedLogic(unittest.TestCase):

    def setUp(self):
        hours = np.arange(6)
        self.ctx = {'load_scale': pd.Series(np.linspace(0.7, 1.2, 6), index=hours),
                    'b_count': {'base': {0, 1}, 'ibr': {0, 1}}}

    def d1_row(self, overrides, metric='ESCR', site='21'):
        d1 = asm.d1_weakest_hour(synthetic_values(6, overrides), self.ctx, ['intact'])
        return d1[(d1['metric'] == metric) & (d1['site'] == site)].iloc[0]

    def test_commitment(self):
        row = self.d1_row({})
        self.assertEqual(row['t_star'], '0')
        self.assertEqual(row['explanation'], 'commitment')

    def test_voltage_exception(self):
        dip = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 1.5])  # |V| dip at hour 5
        row = self.d1_row({('m', 'ESCR', '21'): dip, ('noC', 'ESCR', '21'): dip})
        self.assertEqual(row['t_star'], '5')
        self.assertEqual(row['explanation'], '|V|')

    def test_coupling_exception(self):
        dip = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 1.5])  # coupling dip at hour 5
        row = self.d1_row({('m', 'ESCR', '21'): dip, ('noV', 'ESCR', '21'): dip})
        self.assertEqual(row['explanation'], 'coupling')

    def test_commitment_not_confirmed(self):
        late = np.array([3.0, 2.9, 2.8, 2.7, 2.6, 2.5])
        row = self.d1_row({(v, 'SCR', '21'): late for v in asm.VARIANTS}, metric='SCR')
        self.assertTrue(row['explanation'].startswith('commitment not confirmed'))

    def test_split_is_exact(self):
        rng = np.random.default_rng(1)
        overrides = {(v, 'SDSCR', '48'): rng.uniform(1, 2, 6) for v in asm.VARIANTS}
        values = synthetic_values(6, overrides)
        parts = asm.split(values, 'SDSCR', '48', ('intact', 1), ('intact', 4))
        m = values['m']['SDSCR']['48']
        self.assertAlmostEqual(parts['dln_total'], np.log(m.loc[('intact', 4)] / m.loc[('intact', 1)]))
        self.assertAlmostEqual(parts['dln_total'],
                               parts['dln_V'] + parts['dln_coupling'] + parts['dln_Z_ii'], places=12)


class TestWoodbury(unittest.TestCase):

    def test_branch_update_matches_inversion(self):
        rng = np.random.default_rng(0)
        n = 8
        y = np.zeros((n, n), dtype=complex)
        for a in range(n):  # a ring plus chords, with a shunt at every bus
            for b in ((a + 1) % n, (a + 3) % n):
                yl = 1 / complex(rng.uniform(0.01, 0.05), rng.uniform(0.05, 0.2))
                y[a, a] += yl; y[b, b] += yl; y[a, b] -= yl; y[b, a] -= yl
            y[a, a] += complex(0.5, -0.2)
        a, b = 2, 5
        yl, ys = 1 / complex(0.02, 0.1), 0.03j
        stamp = np.array([[yl + ys / 2, -yl], [-yl, yl + ys / 2]])
        y_with = y.copy()
        y_with[np.ix_([a, b], [a, b])] += stamp
        z = np.linalg.inv(y_with)[None]
        buses = list(range(10, 10 + n))
        npz = {'base_buses': np.array(buses), 'hour_index': np.array([0]), 'z_base': z}
        z_out, z_int = asm.topology_z_ii(npz, 'base', [10, 13, 17], buses[a], buses[b], -stamp)
        direct = np.abs(np.diag(np.linalg.inv(y)))[[0, 3, 7]]
        np.testing.assert_allclose(z_out[0], direct, rtol=1e-12)
        np.testing.assert_allclose(z_int[0], np.abs(np.diag(z[0]))[[0, 3, 7]], rtol=1e-12)
        # series branch alone: the charging shunts stay in service
        y_series_out = y_with.copy()
        y_series_out[np.ix_([a, b], [a, b])] -= np.array([[yl, -yl], [-yl, yl]])
        z_series, _ = asm.topology_z_ii(npz, 'base', [10, 13, 17], buses[a], buses[b],
                                        asm.series_only(-stamp))
        np.testing.assert_allclose(z_series[0], np.abs(np.diag(np.linalg.inv(y_series_out)))[[0, 3, 7]],
                                   rtol=1e-12)

    def test_capacitive_shunt_raises_inductive_impedance(self):
        # the electrical basis of the shunt part: jX in parallel with jB is jX / (1 - XB)
        x, b = 0.2, 0.05
        self.assertGreater(abs(1 / (1 / (1j * x) + 1j * b)), x)


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls._tmp.name) / 'raw'
        cls.raw.mkdir()
        pd.DataFrame([{'line': key, 'case': case, 'from_bus': a, 'to_bus': b, 'islands': False,
                       'n_hours': 3, 'n_converged_base': 3, 'n_converged_ibr': 3,
                       'failed_hours': '', 'included': True, 'reason': ''}
                      for case, (key, a, b) in LINES.items()]
                     ).to_csv(cls.raw / 'contingency_set.csv', index=False)
        cls.week = select_week(read_se3(SE3_FILE)).iloc[:3]
        contingencies = rsm.read_contingencies(cls.raw / 'contingency_set.csv')
        quiet(rsm.evaluate_temporal, cls.week, contingencies=contingencies, out_dir=cls.raw)
        cls.raw_step = Path(cls._tmp.name) / 'raw_if_step_0.02'
        quiet(rsm.evaluate_temporal, cls.week, contingencies=contingencies, out_dir=cls.raw_step,
              if_step=0.02)
        cls.out = quiet(asm.run, cls.raw, Path(cls._tmp.name) / 'analysis', [cls.raw_step])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_scenarios_count_corridors(self):
        d1 = self.out['d1_weakest_hour']
        self.assertEqual(sorted(d1['case'].unique()),
                         ['intact', 'line_18_37_ckt1_out', 'line_39_47_out'])

    def test_reconstruction(self):
        self.assertLess(self.out['reconstruction_check']['max_rel_diff'].max(), 1e-12)

    def test_topology_equals_reinversion_at_intact_voltage(self):
        # Z' from the Woodbury update must equal inverting the outage Y_SC built at the
        # intact pre-fault voltages (same loads, sources and taps; only the line removed).
        limits = read_gen_limits(GEN_LIMITS_FILE)
        row = self.week.iloc[0]
        dispatch = sequential_dispatch(baseline_load_mw() * float(row.load_scale), limits)
        net = quiet(prepare_network, row.load_scale, dispatch, limits)
        v, _ = quiet(rsm.solve_pf, net)
        key, a, b = LINES['line_39_47_out']
        net.change_element_status(key, False)
        net.build_ybus()
        z_direct = np.linalg.inv(short_circuit_admittance(net, v).toarray())
        z = dict(np.load(self.raw / 'z_full_intact.npz'))
        update = asm.branch_update(pd.read_csv(self.raw / 'contingency_set.csv'), ['line_39_47_out'])
        _, _, c = update['line_39_47_out']
        pois = [asm.EVAL_BUS['SCR'][s] for s in asm.SITE_IDS]
        z_out, _ = asm.topology_z_ii(z, 'base', pois, a, b, c)
        idx = [net.idx[p] for p in pois]
        np.testing.assert_allclose(z_out[0], np.abs(np.diag(z_direct))[idx], rtol=1e-10)

    def test_parallel_circuits_identical(self):
        par = self.out['parallel_circuits']
        self.assertEqual(set(par['case']), {'line_18_37_ckt2_out'})
        self.assertTrue(par['same_t_star'].all())
        self.assertLess(par['max_abs_dln'].max(), 1e-9)

    def test_a3_unchanged(self):
        self.assertTrue(self.out['a3_denominators']['unchanged'].all())

    def test_decompositions_add_up(self):
        for name in ('d1_weakest_hour', 'd5_snapshots'):
            t = self.out[name]
            np.testing.assert_allclose(t['dln_total'], t['dln_V'] + t['dln_coupling'] + t['dln_Z_ii'],
                                       atol=1e-12)
        d4 = self.out['d4_outage']
        scr = d4[d4['metric'].isin(['SCR', 'WSCR'])]
        self.assertTrue((scr['median_dln_coupling'].abs() < 1e-12).all())

    def test_if_step_robustness(self):
        rob = self.out['if_step_robustness']
        self.assertEqual(sorted(rob['if_step'].unique()), [0.01, 0.02])
        s = self.out['summary'].set_index('item')['result']
        self.assertLess(float(s['SCR, WSCR, SDSCR unchanged by the step (max relative difference)']), 1e-12)

    def test_coupling_load_definition(self):
        v = pd.read_csv(self.raw / 'metric_variables.csv')
        intact0 = v[(v.case == 'intact') & (v.hour_index == 0) & (v.metric == 'ESCR')]
        ifm = intact0[intact0.variable == 'if_magnitude']
        site = asm.SITE_IDS[0]
        i_bus = asm.EVAL_BUS['ESCR'][site]
        rated = asm.P_MW['rated']
        expected = sum(float(r.real) * rated[next(s for s, b in asm.EVAL_BUS['ESCR'].items() if b == r.bus)]
                       for r in ifm[ifm.related_bus == i_bus].itertuples()) / rated[site]
        values = asm.variant_values(asm.load_inputs(self.raw)['variables'],
                                    {m: asm.P_MW[asm.AS_RUN[m]] for m in asm.METRICS})
        self.assertAlmostEqual(asm.coupling_load(values, 'ESCR', site, 'intact').loc[0], expected, places=12)

    def test_outputs_written(self):
        written = {p.stem for p in (Path(self._tmp.name) / 'analysis').glob('*.csv')}
        self.assertEqual(written, set(self.out))

    def test_rejects_input_as_output(self):
        with self.assertRaises(ValueError):
            asm.run(self.raw, self.raw)


if __name__ == '__main__':
    unittest.main()
