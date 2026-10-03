export PYTHONUTF8 := 1

PYTHON ?= python

CASE := cases/glover37.json
POWERWORLD_DATA := data/powerworld/glover37
SE3_DATA := data/se3_2025.csv

RESULTS := results/glover37
STATIC_RESULTS := $(RESULTS)/static
VALIDATION_RESULTS := $(RESULTS)/validation
MECHANISM_RESULTS := $(RESULTS)/mechanism
TEMPORAL_RESULTS := $(RESULTS)/temporal
TEMPORAL_ANALYSIS := $(TEMPORAL_RESULTS)/analysis

LOG_DIR := logs
REPRODUCE_LOG := $(LOG_DIR)/reproduce.log

.PHONY: \
	help \
	case \
	static \
	static-analysis \
	static-figures \
	validation \
	mechanism \
	temporal \
	temporal-analysis \
	reproduce \
	clean


help:
	@echo "Conference-paper reproducibility workflow"
	@echo
	@echo "  make case               Rebuild the 37-bus case from PowerWorld exports"
	@echo "  make static             Run the controlled static sensitivity scenarios"
	@echo "  make static-analysis    Analyze static changes relative to the baseline"
	@echo "  make static-figures     Generate static spatial figures"
	@echo "  make validation         Reproduce PowerWorld comparison results"
	@echo "  make mechanism          Reproduce structural-change mechanism analysis"
	@echo "  make temporal           Run the 168-hour temporal experiment"
	@echo "  make temporal-analysis  Reproduce temporal-spatial metrics and figures"
	@echo "  make reproduce          Reproduce the complete numerical evidence chain"
	@echo "  make clean              Remove generated results"


case:
	$(PYTHON) tools/pw2json.py \
		$(POWERWORLD_DATA)/base \
		-o $(CASE) \
		--report $(POWERWORLD_DATA)/conversion_report.md


static:
	$(PYTHON) -m experiments.glover37.run_static --all


static-analysis: static
	$(PYTHON) -m experiments.glover37.analyze_static --all


static-figures: static-analysis
	$(PYTHON) -m experiments.glover37.plot_sld --all


validation: static
	$(PYTHON) -m experiments.glover37.validate_powerworld --all


mechanism:
	$(PYTHON) -m experiments.glover37.analyze_mechanism


temporal:
	$(PYTHON) -m experiments.glover37.run_temporal \
		--se3-file $(SE3_DATA) \
		--timestamp-column timestamp \
		--load-column SE3


temporal-analysis: temporal
	$(PYTHON) -m experiments.glover37.analyze_temporal \
		--input-dir $(TEMPORAL_RESULTS) \
		--output-dir $(TEMPORAL_ANALYSIS)


reproduce:
	@mkdir -p $(LOG_DIR)
	@: > $(REPRODUCE_LOG)
	@echo "Starting full reproduction..."
	@echo "[1/6] RUNNING: case reconstruction"
	@$(MAKE) case >> $(REPRODUCE_LOG) 2>&1
	@echo "[1/6] DONE:    case reconstruction"

	@echo "[2/6] RUNNING: static sensitivity analysis"
	@$(MAKE) static-analysis >> $(REPRODUCE_LOG) 2>&1
	@echo "[2/6] DONE:    static sensitivity analysis"

	@echo "[3/6] RUNNING: PowerWorld validation"
	@$(MAKE) validation >> $(REPRODUCE_LOG) 2>&1
	@echo "[3/6] DONE:    PowerWorld validation"

	@echo "[4/6] RUNNING: mechanism analysis"
	@$(MAKE) mechanism >> $(REPRODUCE_LOG) 2>&1
	@echo "[4/6] DONE:    mechanism analysis"

	@echo "[5/6] RUNNING: temporal experiment"
	@$(MAKE) temporal >> $(REPRODUCE_LOG) 2>&1
	@echo "[5/6] DONE:    temporal experiment"

	@echo "[6/6] RUNNING: temporal analysis"
	@$(MAKE) temporal-analysis >> $(REPRODUCE_LOG) 2>&1
	@echo "[6/6] DONE:    temporal analysis"

	@echo "Reproduction complete."
	@echo "Log file: $(REPRODUCE_LOG)"

clean:
	rm -rf $(RESULTS)