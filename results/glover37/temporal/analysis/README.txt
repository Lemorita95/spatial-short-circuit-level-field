Temporal-spatial SCL analysis
=============================

Core design
-----------
- Bus-wise spatial information is preserved.
- No system-wide mean change is used as a primary temporal metric.
- Heatmap normalization is relative to each bus's temporal mean, not hour 0.
- Contingency analysis is kept as a dedicated block.

Mean-normalized definition
--------------------------
delta_i(t) = 100 * [SCL_i(t) - mean_t(SCL_i)] / mean_t(SCL_i)

Quick diagnostics
-----------------
Buses with intact rank changes: 31/37
Bus pairs with at least one intact ordering reversal: 71
Buses whose critical hour changes under contingency: 23/37
Longest fixed-commitment interval used for load diagnostic: hours 86-167 (82 h)

Highest relative temporal ranges (intact)
----------------------------------------
Bus 14: 60.794%
Bus 50: 49.269%
Bus 34: 42.274%
Bus 20: 39.530%
Bus 48: 32.491%
Bus 33: 23.525%
Bus 21: 22.943%
Bus 44: 22.345%
Bus 54: 20.158%
Bus 15: 19.564%

Targeted figures
----------------
figure_intact_mean_normalized_heatmap.png
figure_case_bus13_bus14_rank_reversal.png
figure_case_bus5_bus34_sensitivity.png
figure_case_bus44_contingency_interaction.png
figure_intact_critical_hours.png
figure_intact_rank_heatmap.png
figure_contingency_effect_heatmap.png
figure_diagnostic_contingency_load_relation.png (diagnostic; not automatically a paper figure)

Important interpretation boundary
---------------------------------
The contingency/load correlation output is descriptive. It is calculated within the longest interval with unchanged generator commitment to avoid mixing commitment transitions into the relation. Correlation does not establish that load itself causes the SCL response; load scaling also changes dispatch, voltage, losses and other state variables.