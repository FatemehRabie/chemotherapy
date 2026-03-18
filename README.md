# Integrated PDE and Reinforcement Learning Framework for Personalized Cancer Treatment Optimization

Current version: `1.4.2`

This repository couples a 3D reaction-diffusion tumor simulator with Gymnasium and Stable-Baselines3 to study chemotherapy scheduling policies. It includes:

- preprocessing of GDSC2 dose-response data;
- a custom Gymnasium environment driven by a PDE simulator;
- PPO, TRPO, and A2C training pipelines, plus 3D CNN policy variants;
- heuristic baselines for comparison;
- evaluation utilities that write plots, CSV summaries, LaTeX tables, and text reports;
- a standalone visualization script for side-by-side treatment trajectory renders.

The repository already contains committed datasets, logs, checkpoints, plots, and tables. That makes it easy to inspect prior runs, but it also makes the working tree large.

## Repository layout

```text
.
|-- main.py
|-- generate_visualizations.py
|-- custom_policies.py
|-- requirements.txt
|-- VERSION
|-- GDSC2_fitted_dose_response_27Oct23.xlsx
|-- Filtered_GDSC2_No_Duplicates_Averaged.xlsx
|-- baselines/
|-- env/
|-- simulator/
|-- utils/
|-- logs_*/
`-- results/
```

Key code locations:

- `main.py` orchestrates preprocessing, sweeps, ablations, optional hyperparameter search, and the final comparison run.
- `env/reaction_diffusion.py` defines the Gymnasium environment.
- `simulator/dynamical_system.py` and `simulator/params.py` define the PDE dynamics and global simulation constants.
- `utils/training.py` builds and trains RL agents.
- `utils/evaluation.py` handles evaluation, plotting, out-of-sample runs, CSV export, LaTeX export, and runtime reports.
- `custom_policies.py` defines the 3D CNN feature extractor and policy used by `*_CNN` algorithms.
- `baselines/` contains heuristic control policies and factory helpers.
- `generate_visualizations.py` renders an already-trained agent and baseline regimens for qualitative comparison.

## Environment summary

The environment is registered as `ReactionDiffusion-v0` and models four 3D fields on a `32 x 32 x 32` grid:

- channel 0: normal cells
- channel 1: tumor cells
- channel 2: immune cells
- channel 3: chemotherapy concentration

Observation space:

- `variables`: `Box(shape=(4, 32, 32, 32), dtype=float32)`
- `cell_line`: `Discrete(num_cell_lines)`

Action space:

- `MultiDiscrete([10, 11, num_drugs])`
- action component 0: duration bucket
- action component 1: dose bucket
- action component 2: drug index
- the simulator applies duration as `(duration + 1) * PDE_number_of_steps`
- the simulator applies dose as `0.1 * dose_bucket`

Important simulator defaults from `simulator/params.py`:

- `s_disc = 32`
- `episode_time = 400`
- `PDE_number_of_steps = 4`
- `PDE_step_length = 0.25`
- `PDE_number_of_substeps = 3`
- `observation_noise_level = 0.05`
- `noise_scale = 0.1`

The environment reads `Filtered_GDSC2_No_Duplicates_Averaged.xlsx` directly from the repository root. If that file is missing or renamed, training, evaluation, and visualization will fail until the path is updated.

Environment variants used by the pipeline:

- `baseline`: default stochastic environment settings
- `single-cell-line`: pins the environment to a single cell line, using `--pinned-cell-line` or the first valid cell line found in the processed dataset
- `no-noise`: sets both observation noise and process noise to `0.0`

## Data flow

Two Excel files matter in the default workflow:

- `GDSC2_fitted_dose_response_27Oct23.xlsx`: raw input workbook used by `main.py`
- `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`: processed workbook written by `main.py` and consumed by the environment

`main.py` reads the raw workbook, keeps `CELL_LINE_NAME`, `DRUG_NAME`, and `AUC`, filters to complete cell lines, averages duplicate `(cell line, drug)` pairs, and writes the processed workbook back to the repository root.

## Installation

Create an environment and install the project dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The repository expects the raw GDSC2 workbook to be present in the project root unless you override `--file`.

## Quick start

Run the default end-to-end pipeline:

```bash
python main.py
```

By default, this uses:

- base RL algorithms: `PPO`, `TRPO`, `A2C`
- CNN variants: `PPO_CNN`, `TRPO_CNN`, `A2C_CNN`
- heuristic baselines for the final comparison: `FixedSchedule`, `ProportionalControl`, `RandomPolicy`
- betas: `0.0 0.001 0.01 0.1`
- reduced sweep steps: `5000`
- final training steps: `20000`
- rollout steps per update: `16`
- vectorized environments: `4`
- evaluation episodes: `20`
- parallel evaluation workers: `2`
- seed: `19`
- device: `auto`

## What `main.py` actually does

The current workflow in `main.py` is:

1. Preprocess the raw GDSC2 workbook into `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.
2. Run a reduced-step beta sweep for the base RL algorithms (`PPO`, `TRPO`, `A2C`) under the `baseline` environment.
3. Pick the best beta for each base RL algorithm from in-sample aggregate metrics.
4. Run the CNN variants (`PPO_CNN`, `TRPO_CNN`, `A2C_CNN`) using the selected beta for their corresponding base algorithm.
5. Run the `single-cell-line` and `no-noise` ablation environments using the selected best betas.
6. Optionally run hyperparameter search for the best-performing learned algorithm if it is enabled in the config.
7. Run a final comparison between the best learned policy and the heuristic baselines using full training steps, writing results under `<best_env_label>-final`.

This is more than a single training job. Even with the reduced sweep settings, the default pipeline still launches multiple experiment families.

## Algorithms

Supported learned algorithms:

- `PPO`
- `TRPO`
- `A2C`
- `PPO_CNN`
- `TRPO_CNN`
- `A2C_CNN`

Supported heuristic baselines:

- `FixedSchedule`
- `ProportionalControl`
- `RandomPolicy`

Notes on `beta`:

- For `PPO` and `A2C`, `beta` is used as the entropy coefficient unless explicitly overridden during hyperparameter search.
- For `TRPO`, `beta` is used as `target_kl` unless explicitly overridden.
- A beta of `0.0` does not mean "no regularization forever": `utils/training.py` switches into an adaptive schedule that starts from `0.5` and decays every 5000 timesteps.

## Command-line interface

Main entry point:

```bash
python main.py [options]
```

Important options:

- `--file`: path to the raw GDSC2 workbook
- `--params-file`: JSON config file with overrides
- `--algos`: algorithm list
- `--betas`: beta values to sweep
- `--total-steps`: full training steps for the final comparison run
- `--reduced-total-steps`: reduced training steps for sweep and ablation stages
- `--num-steps`: rollout steps per update
- `--num-envs`: number of vectorized training environments
- `--parallel-workers`: concurrent workers for evaluation
- `--seed`: random seed
- `--device`: model device such as `cpu`, `cuda`, or `auto`
- `--eval-episodes`: evaluation episodes per checkpoint/final evaluation
- `--out-of-sample-cell-lines`: held-out cell line names for out-of-sample evaluation
- `--out-of-sample-diffusions`: JSON-encoded list of diffusion triplets
- `--out-of-sample-drugs`: held-out drug names for out-of-sample evaluation
- `--pinned-cell-line`: cell line name used for the `single-cell-line` variant

Examples:

Run a smaller local sweep:

```bash
python main.py --algos PPO TRPO A2C --betas 0.0 0.01 --reduced-total-steps 2048 --total-steps 4096 --eval-episodes 5 --parallel-workers 1
```

Force a specific device:

```bash
python main.py --device cuda --parallel-workers 4
```

Specify out-of-sample diffusion settings from the CLI:

```bash
python main.py --out-of-sample-diffusions "[[0.002, 0.001, 0.001], [0.0005, 0.0005, 0.0005]]"
```

Use a named cell line for the pinned-cell-line ablation:

```bash
python main.py --pinned-cell-line "CELL_LINE_NAME_HERE"
```

CLI note:

- For `--out-of-sample-cell-lines`, `--out-of-sample-drugs`, and `--pinned-cell-line`, the CLI passes strings, so you should use names from the workbook.
- Numeric indices are more convenient in a JSON params file, where integers stay integers.
- If no out-of-sample targets are provided, evaluation still runs with random cell lines, drugs, and diffusion settings.

## JSON config file

`--params-file` lets you store overrides in JSON. Example:

```json
{
  "algos": ["PPO", "TRPO", "A2C", "PPO_CNN"],
  "betas": [0.0, 0.01],
  "total_steps": 12000,
  "reduced_total_steps": 3000,
  "num_steps": 16,
  "number_of_envs": 4,
  "parallel_workers": 1,
  "seed": 19,
  "device": "auto",
  "eval_episodes": 10,
  "pinned_cell_line": 0,
  "out_of_sample": {
    "cell_lines": [0],
    "diffusions": [[0.002, 0.001, 0.001]],
    "drugs": [0]
  },
  "hyperparam_search": {
    "enabled": true,
    "mode": "grid",
    "max_trials": 3,
    "max_seconds": 900
  }
}
```

Run with:

```bash
python main.py --params-file params.json
```

CLI arguments override values from the JSON file.

## Outputs

The training and evaluation pipeline writes results under `results/` and logs/checkpoints under `logs_*`.

Typical structure:

```text
results/
|-- out_of_sample_plan.txt
|-- baseline/
|   |-- *_training_baseline.txt
|   |-- *_out_of_sample_baseline.txt
|   |-- aggregate_metrics_beta_*.csv
|   |-- runtime_profile_beta_*.txt
|   |-- plots/
|   |   |-- reward_curves/
|   |   |-- aggregate/
|   |   |-- episodes/
|   |   `-- out_of_sample/
|   `-- tables/
|       |-- aggregate_metrics_beta_*.tex
|       `-- best_by_split_beta_*.tex
|-- no-noise/
`-- single-cell-line/
```

Typical contents:

- `results/out_of_sample_plan.txt`: records the held-out evaluation targets that were requested
- `*_training_<label>.txt`: in-sample training/evaluation summaries
- `*_out_of_sample_<label>.txt`: out-of-sample summaries
- `aggregate_metrics_beta_*.csv`: combined metrics across algorithms and splits
- `runtime_profile_beta_*.txt`: wall-clock and cache-use summaries
- `results/<label>/plots/reward_curves/`: smoothed reward curves from `evaluations.npz`
- `results/<label>/plots/aggregate/`: bar charts and heatmaps of aggregate performance
- `results/<label>/plots/episodes/`: action, dose, drug-type, concentration, and reward plots for sampled episodes
- `results/<label>/plots/out_of_sample/`: equivalent plots for out-of-sample episode traces
- `results/<label>/tables/`: LaTeX exports of aggregate metrics and best-by-split summaries

Typical `logs_<algo>_<beta>_<label>/` directories contain:

- `best_model.zip`
- `evaluations.npz`
- logger outputs such as `progress.csv`

## `generate_visualizations.py`

`generate_visualizations.py` is a post-hoc qualitative comparison script. It does not retrain a model. Instead, it:

1. registers `ReactionDiffusion-v0` if needed;
2. creates a single environment instance;
3. loads a trained PPO checkpoint from `logs_PPO_0.0_baseline/best_model.zip`;
4. runs one seeded episode for:
   - the trained PPO agent,
   - an MTD-style fixed schedule baseline,
   - a metronomic fixed schedule baseline,
   - a proportional-control baseline,
   - a random policy baseline;
5. writes side-by-side 2D and 3D visualizations of tumor, immune, and drug dynamics.

Run it with:

```bash
python generate_visualizations.py
```

Current outputs:

- `treatment_comparison_2d.png`
- `3d_renders/*.png`

What the script currently assumes:

- the processed workbook `Filtered_GDSC2_No_Duplicates_Averaged.xlsx` exists in the repository root;
- `logs_PPO_0.0_baseline/best_model.zip` exists;
- the model you want to visualize is a PPO checkpoint, because the script currently calls `PPO.load(...)`.

If you want to visualize a different trained algorithm, update the checkpoint path and load the corresponding class at the bottom of the script.

The script renders four episode checkpoints by default:

- `0%`
- `33%`
- `66%`
- `100%`

For each regimen, the 2D figure includes:

- a tumor mid-slice;
- an immune mid-slice;
- a drug mid-slice;
- a time-series panel of total tumor, immune, and drug mass over episode progress.

The 3D render export writes separate tumor, immune, and drug scatter plots for each regimen/timepoint pair into `3d_renders/`.

## Runtime and reproducibility notes

- The default workflow is computationally expensive because it performs sweeps, CNN follow-up runs, ablations, and a final comparison.
- Increasing `parallel_workers` speeds up evaluation but also causes plot generation to be deferred until the end of the run.
- The repository does not currently include an automated test suite or CI workflow.
- There are no notebooks in the current repository snapshot; the workflow is script-driven.
- The committed artifacts are useful for inspection, but they also make it easy to confuse historical outputs with freshly generated results. Check timestamps and output labels if you are comparing runs.

## Recommended first steps

If you are new to the repository, this order is the least confusing:

1. Read `main.py` and `utils/evaluation.py` to understand the experiment flow.
2. Inspect the existing `results/` and `logs_*` directories before starting a fresh run.
3. Run a reduced local sweep with smaller `--reduced-total-steps`, `--total-steps`, and `--eval-episodes`.
4. Use `python generate_visualizations.py` to produce qualitative figures from the committed PPO checkpoint.
