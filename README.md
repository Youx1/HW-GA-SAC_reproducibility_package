# HW_GA_SAC Minimal Reproducibility Package

This package contains the processed data and lightweight scripts needed to reproduce the numerical tables and figures reported for the HW_GA_SAC manuscript revision.

It is intentionally not a full training or evaluation code release. Model checkpoints, replay buffers, raw training logs, server scripts, and the complete MuJoCo training/evaluation pipeline are excluded to keep the public package compact and maintainable.

## Contents

- `configs/`: final YAML configuration files for the reported methods and HW_GA_SAC ablations.
- `data/evaluation/`: per-episode, per-seed, per-city, and aggregated CSV/JSON results for the paper experiments.
- `data/indexes/`: lightweight file inventories and result indexes.
- `data/derived/random_path_records/`: randomly sampled lightweight path records corresponding to the representative trajectory data mentioned in the manuscript; these CSV files contain only step, position, and target-distance columns for qualitative path-behavior visualization.
- `figures/paper_figures/`: manuscript figure PDFs using the same filenames as the submitted LaTeX source, with additional PNG/SVG exports for convenience.
- `figures/city_layouts/`: top-down city-layout visualizations.
- `scripts/`: plotting/table-generation scripts for reproducing paper figures from the processed CSV data.

## Included Experiments

- Seven-method main comparison with five seeds.
- Core-method seed stability and city split summaries.
- HW_GA_SAC safety ablations.
- Sensor noise and latency stress tests.
- Inference profiling summaries.
- Extended unseen-city and wind robustness summaries.
- Randomly sampled path-record examples for qualitative path-behavior visualization.
- City layout visualizations.

## Minimal Environment

Use `environment.yml` to create a lightweight plotting/statistics environment:

```bash
conda env create -f environment.yml
conda activate hw_ga_sac_repro
```

The scripts are intended for processed-data reproduction only. They do not require MuJoCo or GPU training dependencies.

## Data Availability Wording

Suggested manuscript wording:

> The processed experimental data required to reproduce the tables and figures in this study are publicly available in the accompanying reproducibility package. The package includes final configuration files, per-episode and aggregated CSV/JSON results, representative trajectory data as randomly sampled lightweight path records, city-layout visualizations, sensor-perturbation summaries, and inference-profiling summaries. Raw training logs, replay buffers, model checkpoints, and the full MuJoCo training/evaluation pipeline are not included because of storage and maintenance constraints, but can be made available from the corresponding author upon reasonable request.