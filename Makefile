export PYTHONUTF8 := 1

PYTHON ?= python

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
	reproduce \
	clean


help:
	@echo "Conference-paper reproducibility workflow"
	@echo
	@echo "Complete experiment pipelines:"
	@echo "  make static              Run, analyze, and plot the static experiment"
	@echo "  make mechanism           Run, analyze, and plot the mechanism experiment"
	@echo "  make temporal            Run, analyze, and plot the temporal experiment"
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
	@echo
	@echo "Other:"
	@echo "  make case                Rebuild the 37-bus case from PowerWorld exports"
	@echo "  make reproduce           Reproduce the complete numerical evidence chain"
	@echo "  make clean               Remove generated 37-bus results and figures"


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
	$(PYTHON) -m experiments.glover37.plot_static
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
		--load-column SE3


temporal-analysis:
	$(PYTHON) -m experiments.glover37.analyze_temporal \
		--input-dir $(TEMPORAL_RAW) \
		--output-dir $(TEMPORAL_ANALYSIS)


temporal-figures:
	$(PYTHON) -m experiments.glover37.plot_temporal


# -----------------------------------------------------------------------------
# Complete reproduction
# -----------------------------------------------------------------------------

reproduce:
	@mkdir -p $(LOG_DIR)
	@: > $(REPRODUCE_LOG)
	@echo "Starting full reproduction..."

	@echo "[1/5] RUNNING: case reconstruction"
	@$(MAKE) case >> $(REPRODUCE_LOG) 2>&1
	@echo "[1/5] DONE:    case reconstruction"

	@echo "[2/5] RUNNING: static experiment"
	@$(MAKE) static >> $(REPRODUCE_LOG) 2>&1
	@echo "[2/5] DONE:    static experiment"

	@echo "[3/5] RUNNING: PowerWorld validation"
	@$(MAKE) validation >> $(REPRODUCE_LOG) 2>&1
	@echo "[3/5] DONE:    PowerWorld validation"

	@echo "[4/5] RUNNING: mechanism experiment"
	@$(MAKE) mechanism >> $(REPRODUCE_LOG) 2>&1
	@echo "[4/5] DONE:    mechanism experiment"

	@echo "[5/5] RUNNING: temporal experiment"
	@$(MAKE) temporal >> $(REPRODUCE_LOG) 2>&1
	@echo "[5/5] DONE:    temporal experiment"

	@echo "Reproduction complete."
	@echo "Log file: $(REPRODUCE_LOG)"


clean:
	rm -rf $(RESULTS)
	rm -rf $(FIGURES)
