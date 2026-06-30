# Revision Figure Set

Main-text candidate figures:
- `fig03_seven_method_outcomes`: seven-method three-seed benchmark outcomes.
- `fig04_core_5seed_stability`: five-seed success, collision, and timeout outcomes for TD3, SAC, and HW_GA_SAC.
- `fig05_core_path_stretch`: successful-path stretch under the core five-seed protocol.
- `fig06_city_split_success`: unseen city1-4 versus training replay city42 success.
- `fig07_safety_ablation_success`: camera, LiDAR-sector, and handcrafted-safety ablation success.
- `fig08_sensor_stress_success`: synthetic sensor noise and observation latency stress test.

Appendix candidate figures:
- `figA1_success_return`: seven-method success-return relationship.
- `figA2_city_layout_topdown`: city42 and city1-4 top-down layout montage.
- `figA3_sensor_noise_latency_grid`: full sensor noise, latency, and combined stress grid.
- `figA4_extended_unseen_city_generalization`: city43-52 extended unseen-layout generalization.
- `figA5_wind_robustness`: no-wind versus strong-wind success under the simplified force model.

Notes:
- Small hollow points in `fig04_core_5seed_stability` show individual training-seed outcomes; bars and error bars show mean +/- sample standard deviation.
- `fig08_sensor_stress_success` uses SAC and HW_GA_SAC only to keep the main-text stress figure compact.
- Numeric labels in `fig08_sensor_stress_success` are absolute success rates, not changes relative to the clean condition.
- `fig08_sensor_stress_success` uses the clean condition of the stress-test protocol; it is not identical to the main five-seed evaluation protocol.
- The higher success values under sensor noise or noise+latency should be treated as evaluation variability, not evidence that degradation improves the policy.
- Hollow points in `fig05_core_path_stretch` are seed-level values; bars and error bars show mean +/- sample standard deviation.
- `figA2_city_layout_topdown` uses discrete height bins to make building-height differences visually separable.
- Figure 1 and Figure 2 are architecture diagrams from the manuscript and should be handled in the manuscript figure assets, not in this data plotting script.