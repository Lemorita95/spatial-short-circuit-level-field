# Spatial Sensitivity of Short-Circuit Levels to Network Changes

Reproducible code, public data, and results for the conference-paper study of bus-level short-circuit-level (SCL) variation in the Glover 37-bus benchmark, and for the follow-up comparison of four system-strength screening metrics (SCR, WSCR, ESCR, SDSCR) at three IBR sites added to the same benchmark.

## Reproduce

```bash
pip install -r requirements.txt
make reproduce
```

Or run the experiments separately:

```bash
make static
make validation
make mechanism
make temporal
make strength
```

Run the solver checks, and browse the system-strength results:

```bash
make test
make explorer
```

`make explorer` serves the repository on localhost and opens [`tools/explorer.html`](tools/explorer.html), which reads `results/glover37/strength/` each time it loads (stop with Ctrl-C).

Two variables can be set on any target: `PYTHON` (for example `PYTHON=.venv/bin/python`) and `VOLTAGE_PRESET`, which selects the voltage band used to flag out-of-band operating states. See the [`Makefile`](Makefile) for individual run / analyze / plot stages.

## Repository map

- [`models/`](models/): network model, Newton–Raphson power flow (with generator reactive limits and tap controllers), short-circuit solver, voltage limits, and the system-strength metrics
- [`experiments/glover37/`](experiments/glover37/): experiment, analysis, and plotting scripts
- [`cases/glover37.json`](cases/glover37.json): benchmark network model, generated from the PowerWorld exports by [`tools/pw2json.py`](tools/pw2json.py)
- [`data/powerworld/glover37/`](data/powerworld/glover37/): PowerWorld reference data
- [`data/se3_2025.csv`](data/se3_2025.csv): public temporal scaling signal
- [`results/glover37/`](results/glover37/): retained numerical results, with system-strength results in `strength/`
- [`figures/glover37/`](figures/glover37/): generated paper figures
- [`tools/`](tools/): case conversion, table export, and the results explorer
- [`tests/`](tests/): targeted solver checks

The SE3 demand series is used only as a temporal scaling signal; the Glover 37-bus benchmark is not an electrical model of SE3.

License: [`LICENSE`](LICENSE)
