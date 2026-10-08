# Spatial Sensitivity of Short-Circuit Levels to Network Changes

Reproducible code, public data, and results for the conference-paper study of bus-level short-circuit-level (SCL) variation in the Glover 37-bus benchmark.

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
```

See the [`Makefile`](Makefile) for individual run / analyze / plot stages.

## Repository map

- [`experiments/glover37/`](experiments/glover37/) — experiment, analysis, and plotting scripts
- [`cases/glover37.json`](cases/glover37.json) — benchmark network model
- [`data/powerworld/glover37/`](data/powerworld/glover37/) — PowerWorld reference data
- [`data/se3_2025.csv`](data/se3_2025.csv) — public temporal scaling signal
- [`results/glover37/`](results/glover37/) — retained numerical results
- [`figures/glover37/`](figures/glover37/) — generated paper figures

The SE3 demand series is used only as a temporal scaling signal; the Glover 37-bus benchmark is not an electrical model of SE3.

License: [`LICENSE`](LICENSE)
