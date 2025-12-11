# Integrated PDE and Reinforcement Learning Framework for Personalized Cancer Treatment Optimization

This repository provides tools for processing chemotherapy-related data, training reinforcement learning (RL) models, and evaluating their performance in simulation environments.

**Current version: 1.4.0** (previously 1.3.2)

* a 3-dimensional reaction-diffusion tumour simulator wrapped as an OpenAI Gymnasium environment;
* training utilities for three deep-RL algorithms (PPO, TRPO, A2C) implemented with **Stable-Baselines3** / **sb3-contrib**, plus a CCN (convolutional context network) policy variant;
* classical control baselines (fixed schedule, proportional control, random policy) to contextualize RL performance;
* ablation switches to remove observation noise or pin to a single cell line, enabling component attribution studies;
* processed GDSC2 dose–response data and both in-sample and held-out (out-of-sample) evaluation pipelines;
* experiment logs, model checkpoints and evaluation notebooks that reproduce all tables and figures in the paper.

## What’s new in 1.4.0

- Faster end-to-end experiments: default training steps reduced by ~50% (20k main runs / 5k sweeps) with an 8-env rollout, optional parallel evaluation workers, and automatic GPU selection (`--device` override available).
- Improved output organization: plots are grouped into `results/<label>/plots/` subfolders (reward curves, aggregates, episodes, out-of-sample) and LaTeX tables are written to `results/<label>/tables/` alongside CSV metrics.
- Richer reporting: aggregate heatmaps for experiment vs. algorithm performance, runtime profiles that note the active device, and clearer README guidance on configuring speed/quality trade-offs.

## Quick Start

### Prerequisites

1. Install required Python dependencies:
   ```bash
   pip install stable-baselines3 sb3-contrib gymnasium phi-flow torch pandas numpy matplotlib seaborn
   ```
2. Download the GDSC2 dataset (`GDSC2_fitted_dose_response_27Oct23.xlsx`) from [CancerRxGene](https://www.cancerrxgene.org/) and place it in the root directory of the repository.

### Steps to Run

#### 1. Process data and train RL models

The default workflow cleans the GDSC dataset, trains PPO/TRPO/A2C agents (20k steps with 8 parallel environments by default), and evaluates them with 50 evaluation episodes per checkpoint using auto-selected GPU/CPU acceleration:

```bash
python main.py
```

Use command-line flags to override the defaults (e.g., increase evaluation episodes to 75 and configure held-out test settings):

```bash
python main.py \
  --eval-episodes 75 \
  --out-of-sample-cell-lines HL-60 MOLT-4 \
  --out-of-sample-diffusions "[[0.002, 0.001, 0.001], [0.0005, 0.0005, 0.0005]]"
```

You can also store overrides in a JSON params file:

```json
{
  "eval_episodes": 100,
  "algos": ["PPO", "A2C"],
  "out_of_sample": {
    "cell_lines": ["HL-60"],
    "diffusions": [[0.002, 0.001, 0.001]]
  }
}
```

Run with the params file and further CLI overrides if needed:

```bash
python main.py --params-file params.json --betas 0.0 0.01 --num-envs 8
```

Speed/placement-related flags:

```bash
# Force a specific accelerator and bump evaluation parallelism
python main.py --device cuda --parallel-workers 4
```

### Outputs

- **Processed Data**: Saved as `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.
- **Trained RL Models**: Evaluated with performance metrics, for example the files in the folder `logs_PPO_0.01`.
- **Visualization**: Training performance plots grouped under `results/<label>/plots/` (reward curves, aggregate bars/heatmaps, per-episode traces, out-of-sample traces).
- **Tables**: CSV and LaTeX summaries of aggregate metrics under `results/<label>/tables/` for easy paper/report inclusion.
- **Sample test runs**: Saved in the folder `results`.
- **Summary of training metrics**: Such as wall-clock times, final and best-checkpoint performance, for example `A2C_0.01_training.txt`.

### Experiments run (in order)

The logged experiments follow the default beta sweep order (`[0.0, 0.001, 0.01, 0.1]`) across base algorithms (`PPO`, `TRPO`, `A2C`). Each run produced the corresponding `*_training.txt` file and episode plots in `results/`:

1. PPO with beta = 0.0 (`PPO_0.0_training.txt`)
2. TRPO with beta = 0.0 (`TRPO_0.0_training.txt`)
3. A2C with beta = 0.0 (`A2C_0.0_training.txt`)
4. PPO with beta = 0.001 (`PPO_0.001_training.txt`)
5. TRPO with beta = 0.001 (`TRPO_0.001_training.txt`)
6. A2C with beta = 0.001 (`A2C_0.001_training.txt`)
7. PPO with beta = 0.01 (`PPO_0.01_training.txt`)
8. TRPO with beta = 0.01 (`TRPO_0.01_training.txt`)
9. A2C with beta = 0.01 (`A2C_0.01_training.txt`)
10. PPO with beta = 0.1 (`PPO_0.1_training.txt`)
11. TRPO with beta = 0.1 (`TRPO_0.1_training.txt`)
12. A2C with beta = 0.1 (`A2C_0.1_training.txt`)

### Keeping large experiments tractable

The evaluation pipeline supports several runtime-friendly options that make wide sweeps or repeated runs easier to manage:

- **Environment caching**: `evaluate` now reuses a single registered `ReactionDiffusion-v0` spec and caches evaluation environments between algorithms to avoid repeated instantiation overhead.
- **Parallel runs**: Set `parallel_workers` greater than 1 when calling `evaluate` to process algorithms concurrently (plot generation is deferred automatically in this mode).
- **Plot throttling**: Control episode plotting with `defer_plots=True` to perform all rendering at the end of the run and `plot_episode_stride` (default `1`) to subsample episodes when the environment set is large.
- **Runtime profiling**: Each evaluation writes a `runtime_profile_beta_<beta>_<label>.txt` file alongside other results that captures wall time, cache effectiveness, and per-algorithm runtimes so you can spot slow configurations quickly.
