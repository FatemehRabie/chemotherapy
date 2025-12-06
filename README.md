# Integrated PDE and Reinforcement Learning Framework for Personalized Cancer Treatment Optimization

This repository provides tools for processing chemotherapy-related data, training reinforcement learning (RL) models, and evaluating their performance in simulation environments.

* a 3-dimensional reaction-diffusion tumour simulator wrapped as an OpenAI Gymnasium environment;
* training utilities for three deep-RL algorithms (PPO, TRPO, A2C) implemented with **Stable-Baselines3** / **sb3-contrib**;
* processed GDSC2 dose–response data;
* experiment logs, model checkpoints and evaluation notebooks that reproduce all tables and figures in the paper.

## Quick Start

### Prerequisites

1. Install required Python dependencies:
   ```bash
   pip install stable-baselines3 sb3-contrib gymnasium phi-flow torch pandas numpy matplotlib seaborn
   ```
2. Download the GDSC2 dataset (`GDSC2_fitted_dose_response_27Oct23.xlsx`) from [CancerRxGene](https://www.cancerrxgene.org/) and place it in the root directory of the repository.

### Steps to Run

#### 1. Process data and train RL models

The default workflow cleans the GDSC dataset, trains PPO/TRPO/A2C agents, and evaluates them with 50 evaluation episodes per checkpoint:

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

### Outputs

- **Processed Data**: Saved as `Filtered_GDSC2_No_Duplicates_Averaged.xlsx`.
- **Trained RL Models**: Evaluated with performance metrics, for example the files in the folder `logs_PPO_0.01`.
- **Visualization**: Training performance plots, for example `rewards_beta_0.01.png`.
- **Sample test runs**: Saved in the folder `results`.
- **Summary of training metrics**: Such as wall-clock times, final and best-checkpoint performance, for example `A2C_0.01_training.txt`.

### Keeping large experiments tractable

The evaluation pipeline supports several runtime-friendly options that make wide sweeps or repeated runs easier to manage:

- **Environment caching**: `evaluate` now reuses a single registered `ReactionDiffusion-v0` spec and caches evaluation environments between algorithms to avoid repeated instantiation overhead.
- **Parallel runs**: Set `parallel_workers` greater than 1 when calling `evaluate` to process algorithms concurrently (plot generation is deferred automatically in this mode).
- **Plot throttling**: Control episode plotting with `defer_plots=True` to perform all rendering at the end of the run and `plot_episode_stride` (default `1`) to subsample episodes when the environment set is large.
- **Runtime profiling**: Each evaluation writes a `runtime_profile_beta_<beta>_<label>.txt` file alongside other results that captures wall time, cache effectiveness, and per-algorithm runtimes so you can spot slow configurations quickly.
