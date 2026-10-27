"""Every parameter of the system-strength study, in one place.

Sites and equipment (Gate 2), the week and commitment rules, the contingency
rule, the ESCR probe steps, the PowerWorld validation states (E0) and the
analysis definitions approved at Gate 4 (evidence plan v2). Nothing here runs
a calculation; the other modules of this package import from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# =============================================================================
# 1. Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = ROOT / "cases" / "glover37.json"
POWERWORLD_ROOT = ROOT / "data" / "powerworld" / "glover37"
GEN_LIMITS_FILE = POWERWORLD_ROOT / "base" / "generators.csv"
AXD_FILE = POWERWORLD_ROOT / "base" / "DesignCase2_2010.axd"
SE3_FILE = ROOT / "data" / "se3_2025.csv"

RESULTS_ROOT = ROOT / "results" / "glover37"
RAW_DIR = RESULTS_ROOT / "raw"
ANALYSIS_DIR = RESULTS_ROOT / "analysis"
VALIDATION_DIR = RESULTS_ROOT / "validation"
FIGURES_DIR = ROOT / "figures" / "glover37"

# =============================================================================
# 2. Network and week
# =============================================================================

S_BASE_MVA = 100.0
SLACK_BUS = 31

# Public SE3 demand is only a load-scaling signal: the 168 h window of YEAR
# with the largest load range, scaled to its own mean.
YEAR = 2025
WINDOW_HOURS = 168
SE3_TIMESTAMP_COLUMN = "timestamp"
SE3_LOAD_COLUMN = "SE3"

# Commitment rule: non-slack units filled in descending MWMax (tie-break bus,
# ID); stop when the residual is below the next unit's MWMin; slack takes the rest.

# =============================================================================
# 3. IBR sites (Gate 2)
# =============================================================================
# All branch impedances are on the 100 MVA system base. Transformer per-unit
# nominal taps are unity because the 69/34.5/0.69 kV voltage bases follow the
# respective bus sides.

HV_KV, MV_KV, LV_KV = 69.0, 34.5, 0.69
TRANSFORMER_Z_EQUIPMENT_BASE = 0.005 + 0.060j
FEEDER_Z_SYSTEM_BASE = 0.011 + 0.027j
FEEDER_B_SYSTEM_BASE = 0.069
IBR_MVA_MULTIPLIER = 1.1
TRANSFORMER_MVA_MULTIPLIER = 1.2
Q_CAPABILITY_MULTIPLIER = 0.436


@dataclass(frozen=True)
class IBRSite:
    host_bus: int
    pmax_mw: float
    operating_mw: float
    main_mv_bus: int
    padmount_hv_bus: int
    converter_lv_bus: int

    @property
    def poi_bus(self) -> int:
        """IEEE 2800 evaluation point (original Glover HV connection bus)."""
        return self.host_bus

    @property
    def escr_bus(self) -> int:
        """Fingrid-inspired ESCR evaluation point (padmount HV, 34.5 kV)."""
        return self.padmount_hv_bus

    @property
    def sdscr_bus(self) -> int:
        """Wu (2018) evaluation point (converter LV terminal)."""
        return self.converter_lv_bus

    @property
    def converter_mva(self) -> float:
        return IBR_MVA_MULTIPLIER * self.pmax_mw

    @property
    def transformer_mva(self) -> float:
        return TRANSFORMER_MVA_MULTIPLIER * self.converter_mva

    @property
    def q_limit_mvar(self) -> float:
        return Q_CAPABILITY_MULTIPLIER * self.converter_mva

    @property
    def transformer_z_pu(self) -> complex:
        return TRANSFORMER_Z_EQUIPMENT_BASE * (S_BASE_MVA / self.transformer_mva)


# Two MV buses + one LV bus at each site. Bus IDs do not overlap Glover IDs.
SITES = (
    IBRSite(21, 60.0, 48.0, 2101, 2102, 2103),
    IBRSite(48, 45.0, 36.0, 4801, 4802, 4803),
    IBRSite(16, 45.0, 36.0, 1601, 1602, 1603),
)
IBR_MW = sum(site.operating_mw for site in SITES)

# =============================================================================
# 4. Metrics, operating-state checks, contingencies
# =============================================================================

METRICS = ("SCR", "WSCR", "ESCR", "SDSCR")
# Network each metric is evaluated on: the 37-bus case or the 46-bus IBR case.
NETWORK = {"SCR": "base", "WSCR": "base", "ESCR": "ibr", "SDSCR": "ibr"}
# Denominator as run: rated MW or operating MW.
AS_RUN = {"SCR": "rated", "WSCR": "output", "ESCR": "rated", "SDSCR": "output"}
EVAL_BUS = {
    "SCR": {str(s.host_bus): s.poi_bus for s in SITES},
    "WSCR": {str(s.host_bus): s.poi_bus for s in SITES},
    "ESCR": {str(s.host_bus): s.escr_bus for s in SITES},
    "SDSCR": {str(s.host_bus): s.sdscr_bus for s in SITES},
}

# ESCR interaction-factor probe: relative MV voltage step. The run of record
# uses IF_STEP; IF_STEP_ROBUSTNESS are the reruns of the IF-step check.
IF_STEP = 0.01
IF_STEP_ROBUSTNESS = (0.005, 0.02)

# VoltageLimits preset (models/VoltageLimits.py); bands only flag states.
VOLTAGE_PRESET = "default"

# N-1 rule: every line whose outage islands no bus and whose base and IBR power
# flows converge in every hour, commitment and setpoints held at the intact schedule.
INTACT = "intact"

# =============================================================================
# 5. Analysis definitions (Gate 4, evidence plan v2)
# =============================================================================

REL_TIE = 1e-9
IF_ZERO = 3e-5  # IF ≈ 0 state: every IF < 3e-5 (Gate 3 record, file 02 §5)
RECONSTRUCTION_TOL = 1e-9

# =============================================================================
# 6. E0: PowerWorld validation states
# =============================================================================


class Action(str, Enum):
    SET_GENERATOR_P = "set_generator_p"
    DISCONNECT = "disconnect"


@dataclass(frozen=True)
class Change:
    action: Action
    element: str
    value: float | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    changes: tuple[Change, ...]


E0_SCENARIOS = {
    "A1": Scenario("A1", "Generator 28 dispatched to 0 MW while remaining "
                         "connected and synchronized.",
                   (Change(Action.SET_GENERATOR_P, "generator.g28-1", 0.0),)),
    "A2": Scenario("A2", "Generator 28 disconnected.",
                   (Change(Action.DISCONNECT, "generator.g28-1"),)),
    "A3": Scenario("A3", "Generator 14 disconnected.",
                   (Change(Action.DISCONNECT, "generator.g14-1"),)),
    "B1": Scenario("B1", "Line 14-34 circuit 1 disconnected.",
                   (Change(Action.DISCONNECT, "line.line 14-34 ckt 1"),)),
    "B2": Scenario("B2", "Line 21-48 circuit 1 disconnected.",
                   (Change(Action.DISCONNECT, "line.line 21-48 ckt 1"),)),
    "B3": Scenario("B3", "Transformer 28-29 circuit 1 disconnected.",
                   (Change(Action.DISCONNECT, "transformer.transformer 28-29 ckt 1"),)),
}
E0_STATES = ("base", *E0_SCENARIOS)
