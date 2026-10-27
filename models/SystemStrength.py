"""System-strength screening metrics, with independent per-state histories.

Study conventions:
  SCR/WSCR: pre-installation 37-bus network (no proposed converter equipment).
  ESCR: augmented network, 34.5 kV padmount-HV buses, prescribed +1% voltage step.
  SDSCR: augmented network, 0.69 kV converter terminals, Wu (2018).

All voltages and impedances are on the 100 MVA common base. The calculation
state (including synchronous commitment) is supplied by the experiment runner.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
import warnings
from typing import Mapping

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from models.Generators import Generator
from models.PowerFlow import PowerFlow

S_BASE_MVA = 100.0
EPS = 1e-12


def short_circuit_admittance(net, v_pf):
    """Consistent pre-fault linear SCC admittance.

    Retain the passive YBus, connected conventional-source impedances, and
    constant-power loads as pre-fault *equivalent admittances*. Converters are
    PQ injections in the pre-fault power flow but have no fault admittance.
    This matches the equivalent-load convention of the validated SCC solver.
    """
    if not hasattr(net, "YBus") or not hasattr(net, "idx"):
        raise ValueError("Build net.YBus first")
    v_pf = np.asarray(v_pf, dtype=complex)
    if v_pf.shape != (len(net.idx),):
        raise ValueError("Expected complete complex pre-fault voltage vector")
    y = csc_matrix(net.YBus, dtype=complex).copy()
    diagonal = np.zeros(y.shape[0], dtype=complex)
    for group in (net.generators, net.external_grids):
        for element in group.values():
            if not element.status:
                continue
            rows, cols, vals = element.get_Y()
            for r, c, val in zip(rows, cols, vals):
                if r != c:
                    raise ValueError("Unexpected non-diagonal source admittance")
                diagonal[net.idx[r]] += val
    for load in net.loads.values():
        if not load.status:
            continue
        idx = net.idx[load.bus_1]
        if abs(v_pf[idx]) ** 2 < EPS:
            raise ValueError(f"Cannot form load equivalent at bus {load.bus_1}: zero voltage")
        rows, cols, vals = load.get_Y(v_pf[idx], complex(load.P, load.Q))
        for r, c, val in zip(rows, cols, vals):
            if r != c:
                raise ValueError("Unexpected non-diagonal load admittance")
            diagonal[net.idx[r]] += val
    y.setdiag(y.diagonal() + diagonal)
    return y


def z_block(net, bus_ids, *, v_pf, y=None):
    """(Y_SC_full^-1)[R,R], calculated using sparse solves."""
    bus_ids = tuple(int(b) for b in bus_ids)
    if not bus_ids or len(bus_ids) != len(set(bus_ids)):
        raise ValueError("Evaluation buses must be distinct")
    indices = np.asarray([net.idx[b] for b in bus_ids], dtype=int)
    if y is None:
        y = short_circuit_admittance(net, v_pf)
    rhs = np.zeros((y.shape[0], len(indices)), dtype=complex)
    rhs[indices, np.arange(len(indices))] = 1
    return splu(csc_matrix(y)).solve(rhs)[indices, :]


def _voltage(net, v, bus):
    v = np.asarray(v, dtype=complex)
    if v.shape != (len(net.idx),):
        raise ValueError('Expected complete complex pre-fault voltage vector')
    return v[net.idx[bus]]


def _ratio(num, den):
    if abs(den) < EPS or not (np.isfinite(num) and np.isfinite(den)):
        return float('nan')
    return float(num / den)


@dataclass
class _MetricHistory:
    """Each recorded state contains inputs, intermediate values, and result."""
    history: list[dict] = field(default_factory=list, init=False)

    def _record(self, step, **values):
        if any(r['step'] == step for r in self.history):
            raise ValueError(f'Duplicate metric state: {step!r}')
        record = {'step': step, **values}
        self.history.append(record)
        return record


@dataclass
class SCR(_MetricHistory):
    """IEEE 2800 POI SCR, pre-installation 37-bus reference network."""
    rated_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA

    def evaluate(self, step, net, v_pf):
        if net.full_converters:
            raise ValueError('SCR requires no-IBR reference network')
        if any(b not in net.idx for b in self.rated_mw):
            raise ValueError('POI missing from reference network')
        buses = tuple(self.rated_mw)
        z = z_block(net, buses, v_pf=v_pf)
        scmva = {}
        values = {}
        voltages = {}
        for k, bus in enumerate(buses):
            v = _voltage(net, v_pf, bus)
            voltages[bus] = v
            scmva[bus] = _ratio(self.s_base_mva * abs(v), abs(z[k,k]))
            values[bus] = _ratio(scmva[bus], self.rated_mw[bus])
        return self._record(step, value=values, scmva=scmva,
                            v_pf=voltages, z_ii=dict(zip(buses, np.diag(z))),
                            rated_mw=dict(self.rated_mw))


@dataclass
class WSCR(_MetricHistory):
    """IEEE 2800 grouped WSCR, same no-IBR SCMVA convention as SCR."""
    operating_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA

    def evaluate(self, step, net, v_pf):
        if net.full_converters:
            raise ValueError('WSCR requires no-IBR reference network')
        buses = tuple(self.operating_mw)
        z = z_block(net, buses, v_pf=v_pf)
        scmva = {b: _ratio(self.s_base_mva * abs(_voltage(net, v_pf, b)), abs(z[k,k]))
                 for k, b in enumerate(buses)}
        p_total = sum(self.operating_mw.values())
        numerator = sum(scmva[b] * self.operating_mw[b] for b in buses)
        return self._record(step, value=_ratio(numerator, p_total*p_total),
                            scmva=scmva, operating_mw=dict(self.operating_mw),
                            z_ii=dict(zip(buses, np.diag(z))))


@dataclass
class ESCR(_MetricHistory):
    """Fingrid-inspired MV ESCR using an explicitly prescribed +1% voltage step.

    For each evaluated MV bus, a temporary ideal numerical PV probe holds the
    bus at 1% above its pre-fault voltage magnitude. The probe has infinite
    series impedance (zero Y-bus contribution), zero active power, and wide
    reactive limits. It is added to a deep copy of the solved operating state,
    so neither the production network nor other perturbations are mutated.

    IF[j,i] = | |V_j(new)| - |V_j(base)| | / | |V_i(new)| - |V_i(base)| |.
    This avoids a Q-injection root search and its possible PV/PQ solution-branch
    discontinuities. It is an explicitly prescribed voltage-step method, not a
    claim of exact reproduction of any particular grid-code implementation.
    """
    rated_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA
    relative_voltage_step: float = 0.01
    voltage_tolerance_pu: float = 1e-5
    probe_q_limit_pu: float = 1e6

    def _solve_voltage_step(self, net, v0, bus, base_state=None):
        """Solve one prescribed voltage step on a disposable network copy.

        The solve starts from the base voltages and generator states (base_state).
        Returns also the generators whose state changed during the step,
        {generator bus: (base state, probe state)}.
        """
        working = deepcopy(net)
        name = f'generator.escr_voltage_probe_{bus}'
        if name in working.generators:
            raise ValueError(f'Temporary generator name collision: {name}')

        base_mag = abs(_voltage(net, v0, bus))
        target_mag = base_mag * (1.0 + self.relative_voltage_step)
        # Infinite Z contributes zero source admittance; the active generator
        # object is used only to classify this bus as PV and prescribe |V|.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with redirect_stdout(io.StringIO()):
                probe = Generator(
                    name=name,
                    bus_1=bus,
                    Z_series=float('inf'),
                    V=float(base_mag),
                    P=0.0,
                    Q=0.0,
                    Q_max=self.probe_q_limit_pu,
                    Q_min=-self.probe_q_limit_pu,
                    status=True,
                )
        working.generators[name] = probe
        probe.V = float(target_mag)

        # Taps (and any other discrete controllers) stay frozen during the near-instant
        # voltage step; the solve starts from the solved base state.
        pf = PowerFlow()
        v0 = np.asarray(v0, dtype=complex)
        solved = pf.solve_power_flow(working, warm_start=(np.angle(v0), np.abs(v0)),
                                     pv_state=base_state, controls=False)
        if solved is None:
            raise RuntimeError(f'Power flow failed for prescribed voltage step at bus {bus}')
        angles, magnitudes = solved
        v_new = np.asarray(magnitudes, dtype=float) * np.exp(1j * np.asarray(angles, dtype=float))
        actual_mag = abs(_voltage(working, v_new, bus))
        if not np.isfinite(actual_mag) or abs(actual_mag - target_mag) > self.voltage_tolerance_pu:
            raise RuntimeError(
                f'Voltage-step target not met at bus {bus}: '
                f'target={target_mag:.8f} pu, actual={actual_mag:.8f} pu'
            )

        # Record the probe's reactive output as an audit variable. The evaluated
        # bus has no other generator in this study, so this is its probe Q.
        # bus_flow() returns (PQ_net, PQ_gen, PQ_converter, PQ_load, PQ_shunt);
        # each PQ array has shape (2, n_bus), with P in row 0 and Q in row 1.
        pq_gen = pf.bus_flow(working, np.r_[angles, magnitudes])[1]
        probe_q_pu = pq_gen[1, working.idx[bus]]
        probe_q_mvar = float(probe_q_pu) * self.s_base_mva

        # generators (other than the probe) that changed PV/PQ state during the step
        base_state = base_state or {}
        pv_changes = {g: (base_state[g], s) for g, s in pf.pv_state.items()
                      if g in base_state and s != base_state[g]}
        return probe_q_mvar, target_mag, actual_mag, v_new, pv_changes

    def evaluate(self, step, net, v_pf, pv_state):
        """pv_state: generator states (0 PV, +1 at Qmax, -1 at Qmin) of the pre-fault solve
        that produced v_pf (PowerFlow.pv_state); every probe starts from them."""
        buses = tuple(self.rated_mw)
        z = z_block(net, buses, v_pf=v_pf)
        scmva = {
            b: _ratio(self.s_base_mva * abs(_voltage(net, v_pf, b)), abs(z[k, k]))
            for k, b in enumerate(buses)
        }
        interaction = {}
        probe_q = {}
        target_v = {}
        actual_v = {}
        delta_v = {}
        values = {}
        pv_changes = {}
        base_state = dict(pv_state)

        for i in buses:
            q_mvar, target_mag, actual_mag, v_new, changes = self._solve_voltage_step(
                net, v_pf, i, base_state)
            pv_changes[i] = len(changes)
            if changes:
                warnings.warn(f'{step}: ESCR probe at bus {i} changed generator states '
                              f'{changes} (bus: (base, probe); 0 PV, +1 Qmax, -1 Qmin)')
            probe_q[i] = q_mvar
            target_v[i] = target_mag
            actual_v[i] = actual_mag
            delta_v[i] = {
                j: abs(_voltage(net, v_new, j)) - abs(_voltage(net, v_pf, j))
                for j in buses
            }
            dv_i = delta_v[i][i]
            if not np.isfinite(dv_i) or abs(dv_i) < EPS:
                raise RuntimeError(f'Invalid local voltage response at ESCR bus {i}: {dv_i}')
            for j in buses:
                if j != i:
                    interaction[(j, i)] = _ratio(abs(delta_v[i][j]), abs(dv_i))

        for i in buses:
            denominator = self.rated_mw[i] + sum(
                interaction[(j, i)] * self.rated_mw[j] for j in buses if j != i
            )
            values[i] = _ratio(scmva[i], denominator)

        return self._record(
            step,
            value=values,
            scmva=scmva,
            if_magnitude=interaction,
            probe_reactive_power_mvar=probe_q,
            probe_pv_changes=pv_changes,
            target_voltage_pu=target_v,
            actual_voltage_pu=actual_v,
            delta_v_pu=delta_v,
            rated_mw=dict(self.rated_mw),
            v_pf={b: _voltage(net, v_pf, b) for b in buses},
            z_ii=dict(zip(buses, np.diag(z))),
        )


@dataclass
class SDSCR(_MetricHistory):
    """Wu (2018) terminal-based SDSCR using complex coupling weights."""
    operating_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA

    def evaluate(self, step, net, v_pf):
        buses = tuple(self.operating_mw)
        z = z_block(net, buses, v_pf=v_pf)
        voltages = {b: _voltage(net, v_pf, b) for b in buses}
        weights = {}
        values = {}
        for k, i in enumerate(buses):
            zii, vi = z[k,k], voltages[i]
            if abs(zii) < EPS:
                values[i] = float('nan')
                continue
            effective_p = complex(self.operating_mw[i] / self.s_base_mva)
            for h, j in enumerate(buses):
                if i == j:
                    continue
                vj = voltages[j]
                w = (z[k,h] / zii) * np.conjugate(vi/vj) if abs(vj) > EPS else complex('nan')
                weights[(i,j)] = w
                effective_p += w * self.operating_mw[j] / self.s_base_mva
            values[i] = _ratio(abs(vi)**2, abs(zii)*abs(effective_p))
        return self._record(step, value=values, w_complex=weights,
                            v_pf=voltages, z_rr=z.copy(),
                            operating_mw=dict(self.operating_mw))
