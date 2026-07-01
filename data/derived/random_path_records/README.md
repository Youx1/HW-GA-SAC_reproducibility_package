# Random Path Records

This folder contains randomly sampled lightweight path records for qualitative path-behavior inspection. The records correspond to the representative trajectory data noted in the manuscript data-availability statement, but they are not full simulator logs.

Each CSV contains only:

- `step`
- `x`
- `y`
- `z`
- `distance`

The files do not include actions, rewards, images, range observations, model checkpoints, replay buffers, or simulator internal state. The paired SAC and HW_GA_SAC records in `random_path_record_index.csv` use the same evaluation seeds. In the archived records, both methods finish within the goal-distance threshold.