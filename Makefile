export PYTHONUTF8 := 1

PYTHON ?= python

# VoltageLimits preset used to flag out-of-band states (models/VoltageLimits.py PRESETS)
VOLTAGE_PRESET ?= default

CASE := cases/glover37.json
POWERWORLD_DATA := data/powerworld/glover37
SE3_DATA := data/se3_2025.csv

RESULTS := results/glover37

STATIC_ROOT := $(RESULTS)/static
STATIC_RAW := $(STATIC_ROOT)/raw
STATIC_ANALYSIS := $(STATIC_ROOT)/analysis

VALIDATION_RESULTS := $(RESULTS)/validation

MECHANISM_ROOT := $(RESULTS)/mechanism
MECHANISM_RAW := $(MECHANISM_ROOT)/raw
MECHANISM_ANALYSIS := $(MECHANISM_ROOT)/analysis

TEMPORAL_ROOT := $(RESULTS)/temporal
TEMPORAL_RAW := $(TEMPORAL_ROOT)/raw
TEMPORAL_ANALYSIS := $(TEMPORAL_ROOT)/analysis

STRENGTH_ROOT := $(RESULTS)/strength
STRENGTH_RAW := $(STRENGTH_ROOT)/raw
STRENGTH_ANALYSIS := $(STRENGTH_ROOT)/analysis
# ESCR interaction-factor probe steps for the robustness reruns (results of record: 0.01)
STRENGTH_IF_STEPS := 0.005 0.02

FIGURES := figures/glover37

LOG_DIR := logs
REPRODUCE_LOG := $(LOG_DIR)/reproduce.log


.PHONY: \
	help \
	case \
	static \
	static-run \
	static-analysis \
	static-figures \
	validation \
	mechanism \
	mechanism-run \
	mechanism-analysis \
	mechanism-figures \
	temporal \
	temporal-run \
	temporal-analysis \
	temporal-figures \
	strength \
	strength-contingencies \
	strength-run \
	strength-if-step \
	strength-analysis \
	strength-figures \
	explorer \
	test \
	reproduce \
	clean


help:
	@echo "Conference-paper reproducibility workflow"
	@echo
	@echo "Complete experiment pipelines:"
	@echo "  make static              Run, analyze, and plot the static experiment"
	@echo "  make mechanism           Run, analyze, and plot the mechanism experiment"
	@echo "  make temporal            Run, analyze, and plot the temporal experiment"
	@echo "  make strength            Screen, run, analyze, and plot the 168 h system-strength metrics (intact + N-1)"
	@echo
	@echo "Individual stages:"
	@echo "  make static-run          Generate static raw results"
	@echo "  make static-analysis     Analyze existing static raw results"
	@echo "  make static-figures      Generate static paper figures"
	@echo "  make validation          Reproduce PowerWorld reference agreement"
	@echo "  make mechanism-run       Generate mechanism raw results"
	@echo "  make mechanism-analysis  Analyze existing mechanism raw results"
	@echo "  make mechanism-figures   Generate mechanism figure candidate(s)"
	@echo "  make temporal-run        Generate temporal raw results"
	@echo "  make temporal-analysis   Analyze existing temporal raw results"
	@echo "  make temporal-figures    Generate temporal paper figures"
	@echo "  make strength-contingencies  Screen N-1 line outages (islanding, convergence)"
	@echo "  make strength-run        Generate system-strength raw results (intact + N-1 set)"
	@echo "  make strength-if-step    Rerun with other ESCR probe steps (robustness)"
	@echo "  make strength-analysis   Analyze existing system-strength raw results"
	@echo "  make strength-figures    Generate system-strength figures"
	@echo
	@echo "Other:"
	@echo "  make explorer            Open the strength results explorer (Ctrl-C to stop)"
	@echo "  make test                Run the solver checks in tests/"
	@echo "  make case                Rebuild the 37-bus case from PowerWorld exports"
	@echo "  make reproduce           Reproduce the complete numerical evidence chain"
	@echo "  make clean               Remove generated 37-bus results and figures"
	@echo
	@echo "Variables:"
	@echo "  PYTHON=$(PYTHON)  VOLTAGE_PRESET=$(VOLTAGE_PRESET)"


# -----------------------------------------------------------------------------
# Case construction
# -----------------------------------------------------------------------------

case:
	$(PYTHON) tools/pw2json.py \
		$(POWERWORLD_DATA)/base \
		-o $(CASE) \
		--report $(POWERWORLD_DATA)/conversion_report.md


# -----------------------------------------------------------------------------
# Static experiment
# run -> results/glover37/static/raw/
# analyze -> results/glover37/static/analysis/
# plot -> figures/glover37/
# -----------------------------------------------------------------------------

static:
	$(MAKE) static-run
	$(MAKE) static-analysis
	$(MAKE) static-figures


static-run:
	$(PYTHON) -m experiments.glover37.run_static --all


static-analysis:
	$(PYTHON) -m experiments.glover37.analyze_static --all


static-figures:
	$(PYTHON) -m experiments.glover37.plot_static --single-column
	$(PYTHON) -m experiments.glover37.plot_sld --overview


# -----------------------------------------------------------------------------
# PowerWorld reference agreement
# Consumes static raw results.
# -----------------------------------------------------------------------------

validation:
	$(PYTHON) -m experiments.glover37.validate_powerworld --all


# -----------------------------------------------------------------------------
# Structural-change mechanism experiment
# run -> results/glover37/mechanism/raw/
# analyze -> results/glover37/mechanism/analysis/
# plot -> figures/glover37/
# -----------------------------------------------------------------------------

mechanism:
	$(MAKE) mechanism-run
	$(MAKE) mechanism-analysis
	$(MAKE) mechanism-figures


mechanism-run:
	$(PYTHON) -m experiments.glover37.run_mechanism --all


mechanism-analysis:
	$(PYTHON) -m experiments.glover37.analyze_mechanism --all


mechanism-figures:
	$(PYTHON) -m experiments.glover37.plot_mechanism


# -----------------------------------------------------------------------------
# Temporal experiment
# run -> results/glover37/temporal/raw/
# analyze -> results/glover37/temporal/analysis/
# plot -> figures/glover37/
# -----------------------------------------------------------------------------

temporal:
	$(MAKE) temporal-run
	$(MAKE) temporal-analysis
	$(MAKE) temporal-figures


temporal-run:
	$(PYTHON) -m experiments.glover37.run_temporal \
		--se3-file $(SE3_DATA) \
		--timestamp-column timestamp \
		--load-column SE3 \
		--voltage-preset $(VOLTAGE_PRESET)


temporal-analysis:
	$(PYTHON) -m experiments.glover37.analyze_temporal \
		--input-dir $(TEMPORAL_RAW) \
		--output-dir $(TEMPORAL_ANALYSIS)


temporal-figures:
	$(PYTHON) -m experiments.glover37.plot_temporal


# -----------------------------------------------------------------------------
# System-strength metrics experiment (SCR, WSCR, ESCR, SDSCR)
# screen -> results/glover37/strength/raw/contingency_set.csv
# run -> results/glover37/strength/raw/ (and raw_if_step_<step>/ for the robustness reruns)
# analyze -> results/glover37/strength/analysis/
# plot -> figures/glover37/
# -----------------------------------------------------------------------------

strength:
	$(MAKE) strength-contingencies
	$(MAKE) strength-run
	$(MAKE) strength-if-step
	$(MAKE) strength-analysis
	$(MAKE) strength-figures


strength-contingencies:
	$(PYTHON) -m experiments.glover37.screen_contingencies \
		--output $(STRENGTH_RAW)/contingency_set.csv


strength-run:
	$(PYTHON) -m experiments.glover37.run_strength_metrics \
		--contingencies $(STRENGTH_RAW)/contingency_set.csv \
		--voltage-preset $(VOLTAGE_PRESET) \
		--output $(STRENGTH_RAW)


strength-if-step:
	for step in $(STRENGTH_IF_STEPS); do \
		$(PYTHON) -m experiments.glover37.run_strength_metrics \
			--contingencies $(STRENGTH_RAW)/contingency_set.csv \
			--voltage-preset $(VOLTAGE_PRESET) \
			--if-step $$step \
			--output $(STRENGTH_ROOT)/raw_if_step_$$step || exit 1; \
	done


strength-analysis:
	$(PYTHON) -m experiments.glover37.analyze_strength_metrics \
		--input-dir $(STRENGTH_RAW) \
		--output-dir $(STRENGTH_ANALYSIS) \
		$(foreach step,$(STRENGTH_IF_STEPS),--if-step-run $(STRENGTH_ROOT)/raw_if_step_$(step))


strength-figures:
	$(PYTHON) -m experiments.glover37.plot_strength_metrics


# -----------------------------------------------------------------------------
# Results explorer
# tools/explorer.py serves the repository on localhost and opens tools/explorer.html,
# which reads results/glover37/strength/raw/ at runtime.
# -----------------------------------------------------------------------------

explorer:
	$(PYTHON) tools/explorer.py


# -----------------------------------------------------------------------------
# Solver checks
# -----------------------------------------------------------------------------

test:
	$(PYTHON) -m unittest discover -s tests -v


# -----------------------------------------------------------------------------
# Complete reproduction
# -----------------------------------------------------------------------------

reproduce:
	@mkdir -p $(LOG_DIR)
	@: > $(REPRODUCE_LOG)
	@echo "Starting full reproduction..."

	@echo "[1/7] RUNNING: solver checks"
	@$(MAKE) test >> $(REPRODUCE_LOG) 2>&1
	@echo "[1/7] DONE:    solver checks"

	@echo "[2/7] RUNNING: case reconstruction"
	@$(MAKE) case >> $(REPRODUCE_LOG) 2>&1
	@echo "[2/7] DONE:    case reconstruction"

	@echo "[3/7] RUNNING: static experiment"
	@$(MAKE) static >> $(REPRODUCE_LOG) 2>&1
	@echo "[3/7] DONE:    static experiment"

	@echo "[4/7] RUNNING: PowerWorld validation"
	@$(MAKE) validation >> $(REPRODUCE_LOG) 2>&1
	@echo "[4/7] DONE:    PowerWorld validation"

	@echo "[5/7] RUNNING: mechanism experiment"
	@$(MAKE) mechanism >> $(REPRODUCE_LOG) 2>&1
	@echo "[5/7] DONE:    mechanism experiment"

	@echo "[6/7] RUNNING: temporal experiment"
	@$(MAKE) temporal >> $(REPRODUCE_LOG) 2>&1
	@echo "[6/7] DONE:    temporal experiment"

	@echo "[7/7] RUNNING: system-strength experiment"
	@$(MAKE) strength >> $(REPRODUCE_LOG) 2>&1
	@echo "[7/7] DONE:    system-strength experiment"

	@echo "Reproduction complete."
	@echo "Log file: $(REPRODUCE_LOG)"


clean:
	rm -rf $(RESULTS)
	rm -rf $(FIGURES)
