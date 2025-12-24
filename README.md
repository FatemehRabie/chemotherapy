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

- Faster end-to-end experiments: default training steps reduced by ~50% (20k main runs / 5k sweeps) with a 4-env rollout using 16-step trajectories, optional parallel evaluation workers, and automatic GPU selection (`--device` override available).
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

The default workflow cleans the GDSC dataset, trains PPO/TRPO/A2C agents (20k steps with 4 parallel environments by default), and evaluates them with 20 evaluation episodes per checkpoint using auto-selected GPU/CPU acceleration:

```bash
python main.py
```

The reaction–diffusion simulator now runs on a (32 × 32 × 32) spatial grid (`s_disc = 32`) with a quarter-length PDE time step (`PDE_step_length = 0.25`, from `PDE_number_of_steps = 4`), so both space and time discretizations are explicit in the defaults.

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

Core runtime hyperparameters (CLI flag → config key) and how they affect speed/quality:

- `--num-steps` → `num_steps` (default: 16): rollout length per policy update. Shorter rollouts (smaller values) speed up each update and lower memory use at the cost of higher gradient noise.
- `--num-envs` → `number_of_envs` (default: 4): parallel vector environments for on-policy rollouts. Fewer environments reduce CPU load and coordination overhead but provide fewer samples per update.
- `--eval-episodes` → `eval_episodes` (default: 20): episodes run during evaluation checkpoints. Lower counts shorten evaluation time but make metrics noisier.

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

## Full Experiment Report (Logs and Results Snapshot)

The following section is an exhaustive, **verbatim** capture of all text-based logs and results generated by the current repository runs (excluding incomplete CNN runs as requested). This section is intended to make every experiment detail available directly in this README without referencing any other file.

**Notes:**
- CNN runs are excluded (e.g., `logs_PPO_CNN_0.0_baseline`).
- Binary artifacts (e.g., `.npz`, `.zip`) are intentionally omitted per request.

### Experiment overview (summary extracted from results)

**Experiment label:** baseline  
**Environment kwargs:** default  
**Out-of-sample evaluation targets:** Cell lines = random; Drugs = random; Diffusion regimes = random  
**Algorithms:** A2C, PPO, TRPO  
**Noise levels (beta):** 0.0, 0.001, 0.01, 0.1  
**Evaluation episodes:** 20 (all reported training/out-of-sample evaluations)  
**Training steps (max `time/total_timesteps`):**
- A2C: 4608 timesteps (all betas)
- PPO: 5056 timesteps (all betas)
- TRPO: 5056 timesteps (all betas)

### Training results (in-sample)

| Algo | Beta (noise) | Mean reward (last model) | Mean reward (best model) | Runtime | Eval episodes |
|---|---:|---|---|---|---:|
| A2C | 0.0 | -183.16 ± 0.92 | -183.02 ± 1.14 | 10.47 hours | 20 |
| A2C | 0.001 | -183.04 ± 1.21 | -183.15 ± 1.13 | 10.79 hours | 20 |
| A2C | 0.01 | -183.22 ± 1.10 | -183.25 ± 1.12 | 8.45 hours | 20 |
| A2C | 0.1 | -182.68 ± 0.99 | -176.45 ± 61.49 | 8.09 hours | 20 |
| PPO | 0.0 | -56.11 ± 1.79 | -65.41 ± 4.30 | 16.30 hours | 20 |
| PPO | 0.001 | -183.56 ± 0.79 | -183.04 ± 1.16 | 11.62 hours | 20 |
| PPO | 0.01 | -183.17 ± 0.98 | -183.06 ± 0.77 | 11.67 hours | 20 |
| PPO | 0.1 | -183.50 ± 1.31 | -110.16 ± 7.20 | 13.46 hours | 20 |
| TRPO | 0.0 | -182.84 ± 1.15 | -183.14 ± 1.00 | 11.23 hours | 20 |
| TRPO | 0.001 | -183.01 ± 0.99 | -182.93 ± 1.20 | 17.10 hours | 20 |
| TRPO | 0.01 | -183.37 ± 1.34 | -90.62 ± 84.90 | 15.79 hours | 20 |
| TRPO | 0.1 | -187.32 ± 3.32 | -183.13 ± 0.95 | 13.40 hours | 20 |

### Out-of-sample results

| Algo | Beta (noise) | Out-of-sample mean reward (best model) | Std/variance | Eval episodes | Cell lines | Diffusions | Drugs |
|---|---:|---|---|---:|---|---|---|
| A2C | 0.0 | -183.09 | ± 1.14 | 20 | random | random | random |
| A2C | 0.001 | -183.23 | ± 1.11 | 20 | random | random | random |
| A2C | 0.01 | -183.33 | ± 1.10 | 20 | random | random | random |
| A2C | 0.1 | -184.41 | ± 69.65 | 20 | random | random | random |
| PPO | 0.0 | -64.62 | ± 5.60 | 20 | random | random | random |
| PPO | 0.001 | -183.29 | ± 1.22 | 20 | random | random | random |
| PPO | 0.01 | -183.35 | ± 1.31 | 20 | random | random | random |
| PPO | 0.1 | -111.35 | ± 8.18 | 20 | random | random | random |
| TRPO | 0.0 | -183.02 | ± 0.93 | 20 | random | random | random |
| TRPO | 0.001 | -183.36 | ± 1.20 | 20 | random | random | random |
| TRPO | 0.01 | -109.03 | ± 90.65 | 20 | random | random | random |
| TRPO | 0.1 | -183.18 | ± 1.07 | 20 | random | random | random |

### Aggregate metrics (CSV summaries)

#### Beta = 0.001
| Algo | Split | Mean reward | Std reward | Context |
|---|---|---:|---:|---|
| PPO | in_sample | -183.04424135 | 1.1594728422348752 | standard |
| PPO | out_of_sample | -183.2947248 | 1.215936047761496 | cell_lines=random, diffusions=random, drugs=random |
| TRPO | in_sample | -182.92539195 | 1.2003535651463906 | standard |
| TRPO | out_of_sample | -183.36401845 | 1.1979494587154136 | cell_lines=random, diffusions=random, drugs=random |
| A2C | in_sample | -183.1495587 | 1.1328069456677123 | standard |
| A2C | out_of_sample | -183.2346203 | 1.1115439702692889 | cell_lines=random, diffusions=random, drugs=random |

#### Beta = 0.01
| Algo | Split | Mean reward | Std reward | Context |
|---|---|---:|---:|---|
| PPO | in_sample | -183.06073505 | 0.7741866244893063 | standard |
| PPO | out_of_sample | -183.34875425 | 1.3101357108541791 | cell_lines=random, diffusions=random, drugs=random |
| TRPO | in_sample | -90.6227288 | 84.90233097130373 | standard |
| TRPO | out_of_sample | -109.02980755 | 90.65039824885484 | cell_lines=random, diffusions=random, drugs=random |
| A2C | in_sample | -183.24970785 | 1.1191620744173878 | standard |
| A2C | out_of_sample | -183.32754265 | 1.0996380132428734 | cell_lines=random, diffusions=random, drugs=random |

#### Beta = 0.0
| Algo | Split | Mean reward | Std reward | Context |
|---|---|---:|---:|---|
| TRPO | in_sample | -183.137246 | 0.9993568168750339 | standard |
| TRPO | out_of_sample | -183.0176288 | 0.9258979513447796 | cell_lines=random, diffusions=random, drugs=random |
| PPO | in_sample | -65.4145617 | 4.3023773648114485 | standard |
| PPO | out_of_sample | -64.6204703 | 5.5985069618575904 | cell_lines=random, diffusions=random, drugs=random |
| A2C | in_sample | -183.02007495 | 1.142244227826844 | standard |
| A2C | out_of_sample | -183.0908157 | 1.1390763696889732 | cell_lines=random, diffusions=random, drugs=random |

#### Beta = 0.1
| Algo | Split | Mean reward | Std reward | Context |
|---|---|---:|---:|---|
| PPO | in_sample | -110.15758165 | 7.196616204796746 | standard |
| PPO | out_of_sample | -111.3541769 | 8.184170169724528 | cell_lines=random, diffusions=random, drugs=random |
| TRPO | in_sample | -183.1262258 | 0.9531631691739638 | standard |
| TRPO | out_of_sample | -183.18278195 | 1.0653470012379767 | cell_lines=random, diffusions=random, drugs=random |
| A2C | in_sample | -176.44972455 | 61.48613725081005 | standard |
| A2C | out_of_sample | -184.4132524 | 69.64735626874041 | cell_lines=random, diffusions=random, drugs=random |

### Runtime profiles (verbatim)
#### runtime_profile_beta_0.001_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
PPO: 11.62 hours
TRPO: 17.10 hours
A2C: 10.79 hours
standard_env_acquisition: 0.17 seconds
oos_env_acquisition: 0.32 seconds
standard_env_acquisition: 0.17 seconds
PPO_runtime: 78196.70 seconds
oos_env_acquisition: 0.13 seconds
TRPO_runtime: 97820.23 seconds
standard_env_acquisition: 0.12 seconds
oos_env_acquisition: 0.06 seconds
A2C_runtime: 55717.83 seconds
Total wall time: 37.20 hours
```

#### runtime_profile_beta_0.01_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
PPO: 11.67 hours
TRPO: 15.79 hours
A2C: 8.45 hours
standard_env_acquisition: 0.39 seconds
oos_env_acquisition: 0.12 seconds
standard_env_acquisition: 0.19 seconds
oos_env_acquisition: 0.13 seconds
PPO_runtime: 78185.78 seconds
TRPO_runtime: 80939.58 seconds
standard_env_acquisition: 0.11 seconds
oos_env_acquisition: 0.06 seconds
A2C_runtime: 47301.49 seconds
Total wall time: 34.86 hours
```

#### runtime_profile_beta_0.0_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
TRPO: 11.23 hours
PPO: 16.30 hours
A2C: 10.47 hours
standard_env_acquisition: 0.19 seconds
oos_env_acquisition: 0.13 seconds
standard_env_acquisition: 0.18 seconds
TRPO_runtime: 76634.11 seconds
oos_env_acquisition: 0.12 seconds
PPO_runtime: 94310.71 seconds
standard_env_acquisition: 0.11 seconds
oos_env_acquisition: 0.07 seconds
A2C_runtime: 54621.05 seconds
Total wall time: 36.46 hours
```

#### runtime_profile_beta_0.1_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
PPO: 13.46 hours
TRPO: 13.40 hours
A2C: 8.09 hours
standard_env_acquisition: 0.26 seconds
standard_env_acquisition: 0.13 seconds
oos_env_acquisition: 0.11 seconds
oos_env_acquisition: 0.12 seconds
PPO_runtime: 83689.98 seconds
TRPO_runtime: 84344.75 seconds
standard_env_acquisition: 0.10 seconds
oos_env_acquisition: 0.07 seconds
A2C_runtime: 46016.56 seconds
Total wall time: 36.03 hours
```

### Full log and results file contents (verbatim)

#### logs_A2C_0.001_baseline/log.txt

```
Logging to ./logs_A2C_0.001_baseline
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -182     |
| time/                 |          |
|    total_timesteps    | 512      |
| train/                |          |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 7        |
|    policy_loss        | -415     |
|    value_loss         | 2.39e+03 |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 1024      |
| train/                |           |
|    entropy_loss       | -10.2     |
|    explained_variance | -2.38e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 15        |
|    policy_loss        | -380      |
|    value_loss         | 1.85e+03  |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 1536     |
| train/                |          |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 23       |
|    policy_loss        | -435     |
|    value_loss         | 2.43e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 400      |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 2048     |
| train/                |          |
|    entropy_loss       | -10.1    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 31       |
|    policy_loss        | -365     |
|    value_loss         | 1.78e+03 |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -184      |
| time/                 |           |
|    total_timesteps    | 2560      |
| train/                |           |
|    entropy_loss       | -10.1     |
|    explained_variance | -1.19e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 39        |
|    policy_loss        | -250      |
|    value_loss         | 1.09e+03  |
-------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 3072      |
| train/                |           |
|    entropy_loss       | -10.1     |
|    explained_variance | -2.38e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 47        |
|    policy_loss        | -451      |
|    value_loss         | 2.68e+03  |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 3584     |
| train/                |          |
|    entropy_loss       | -10      |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 55       |
|    policy_loss        | -415     |
|    value_loss         | 2.24e+03 |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 4096      |
| train/                |           |
|    entropy_loss       | -10       |
|    explained_variance | -2.38e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 63        |
|    policy_loss        | -313      |
|    value_loss         | 1.48e+03  |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 4608     |
| train/                |          |
|    entropy_loss       | -9.91    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 71       |
|    policy_loss        | -327     |
|    value_loss         | 1.47e+03 |
------------------------------------
```

#### logs_A2C_0.001_baseline/progress.csv

```
train/explained_variance,train/n_updates,time/total_timesteps,train/learning_rate,train/entropy_loss,eval/mean_ep_length,train/policy_loss,eval/mean_reward,train/value_loss
0.0,7,512,0.0003,-10.321634292602539,80.0,-415.0943603515625,-182.34522139999999,2389.598876953125
-2.384185791015625e-07,15,1024,0.0003,-10.24338150024414,80.0,-379.88299560546875,-183.48513659999998,1854.885986328125
0.0,23,1536,0.0003,-10.190153121948242,80.0,-434.7960205078125,-183.1433224,2427.43017578125
0.0,31,2048,0.0003,-10.130709648132324,400.0,-364.5068359375,-182.8376694,1778.57861328125
-1.1920928955078125e-07,39,2560,0.0003,-10.108536720275879,80.0,-249.75804138183594,-183.8868214,1091.81005859375
-2.384185791015625e-07,47,3072,0.0003,-10.070161819458008,80.0,-451.1073303222656,-182.64109420000003,2675.70947265625
0.0,55,3584,0.0003,-10.049569129943848,80.0,-414.62457275390625,-182.96732860000003,2241.98583984375
-2.384185791015625e-07,63,4096,0.0003,-10.00152587890625,80.0,-312.8813781738281,-182.6215732,1477.567626953125
0.0,71,4608,0.0003,-9.91028118133545,80.0,-326.9285583496094,-182.9839048,1467.604248046875
```

#### logs_A2C_0.01_baseline/log.txt

```
Logging to ./logs_A2C_0.01_baseline
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 512      |
| train/                |          |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 7        |
|    policy_loss        | -486     |
|    value_loss         | 3.27e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 1024     |
| train/                |          |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 15       |
|    policy_loss        | -342     |
|    value_loss         | 1.65e+03 |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 1536      |
| train/                |           |
|    entropy_loss       | -10.2     |
|    explained_variance | -1.19e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 23        |
|    policy_loss        | -500      |
|    value_loss         | 2.96e+03  |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 2048     |
| train/                |          |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 31       |
|    policy_loss        | -373     |
|    value_loss         | 1.8e+03  |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 2560      |
| train/                |           |
|    entropy_loss       | -10.1     |
|    explained_variance | -1.19e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 39        |
|    policy_loss        | -287      |
|    value_loss         | 1.3e+03   |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -182     |
| time/                 |          |
|    total_timesteps    | 3072     |
| train/                |          |
|    entropy_loss       | -10.1    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 47       |
|    policy_loss        | -471     |
|    value_loss         | 2.95e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 3584     |
| train/                |          |
|    entropy_loss       | -10.1    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 55       |
|    policy_loss        | -405     |
|    value_loss         | 2.15e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -184     |
| time/                 |          |
|    total_timesteps    | 4096     |
| train/                |          |
|    entropy_loss       | -10.1    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 63       |
|    policy_loss        | -211     |
|    value_loss         | 833      |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -182     |
| time/                 |          |
|    total_timesteps    | 4608     |
| train/                |          |
|    entropy_loss       | -9.97    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 71       |
|    policy_loss        | -342     |
|    value_loss         | 1.62e+03 |
------------------------------------
```

#### logs_A2C_0.01_baseline/progress.csv

```
train/explained_variance,train/n_updates,time/total_timesteps,train/learning_rate,train/entropy_loss,eval/mean_ep_length,train/policy_loss,eval/mean_reward,train/value_loss
0.0,7,512,0.0003,-10.327014923095703,80.0,-485.6667175292969,-182.754709,3269.590087890625
0.0,15,1024,0.0003,-10.251798629760742,80.0,-341.51324462890625,-183.4035958,1647.33837890625
-1.1920928955078125e-07,23,1536,0.0003,-10.209508895874023,80.0,-500.17279052734375,-183.17484280000002,2964.14794921875
0.0,31,2048,0.0003,-10.162559509277344,80.0,-372.6938781738281,-182.89408699999998,1803.91357421875
-1.1920928955078125e-07,39,2560,0.0003,-10.123132705688477,80.0,-287.4178771972656,-182.88558139999998,1303.509521484375
0.0,47,3072,0.0003,-10.107158660888672,80.0,-471.4856262207031,-182.11473619999998,2946.38671875
0.0,55,3584,0.0003,-10.091238021850586,80.0,-405.4434814453125,-182.5077014,2154.7685546875
0.0,63,4096,0.0003,-10.052495002746582,80.0,-211.44479370117188,-183.835934,833.4022827148438
0.0,71,4608,0.0003,-9.973920822143555,80.0,-342.1588134765625,-182.4213052,1622.4111328125
```

#### logs_A2C_0.0_baseline/log.txt

```
Logging to ./logs_A2C_0.0_baseline
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -268     |
| time/                 |          |
|    total_timesteps    | 512      |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 7        |
|    policy_loss        | -394     |
|    value_loss         | 2.13e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 1024     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 15       |
|    policy_loss        | -341     |
|    value_loss         | 1.7e+03  |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 1536     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.2    |
|    explained_variance | 5.96e-08 |
|    learning_rate      | 0.0003   |
|    n_updates          | 23       |
|    policy_loss        | -447     |
|    value_loss         | 2.66e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 58       |
|    mean_reward        | -186     |
| time/                 |          |
|    total_timesteps    | 2048     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 31       |
|    policy_loss        | -377     |
|    value_loss         | 1.87e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 2560     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 39       |
|    policy_loss        | -376     |
|    value_loss         | 2.01e+03 |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 3072      |
| train/                |           |
|    adaptive_beta      | 0.5       |
|    entropy_loss       | -10.2     |
|    explained_variance | -1.19e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 47        |
|    policy_loss        | -443      |
|    value_loss         | 2.61e+03  |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 58       |
|    mean_reward        | -186     |
| time/                 |          |
|    total_timesteps    | 3584     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 55       |
|    policy_loss        | -380     |
|    value_loss         | 1.95e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 58       |
|    mean_reward        | -186     |
| time/                 |          |
|    total_timesteps    | 4096     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 63       |
|    policy_loss        | -273     |
|    value_loss         | 1.34e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 4608     |
| train/                |          |
|    adaptive_beta      | 0.5      |
|    entropy_loss       | -10.1    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 71       |
|    policy_loss        | -340     |
|    value_loss         | 1.55e+03 |
------------------------------------
```

#### logs_A2C_0.0_baseline/progress.csv

```
train/adaptive_beta,train/explained_variance,train/n_updates,time/total_timesteps,train/learning_rate,train/entropy_loss,eval/mean_ep_length,train/policy_loss,eval/mean_reward,train/value_loss
0.5,0.0,7,512,0.0003,-10.323141098022461,80.0,-394.4951171875,-268.4367542,2127.57861328125
0.5,0.0,15,1024,0.0003,-10.267898559570312,80.0,-341.0807800292969,-183.11362400000002,1696.659912109375
0.5,5.960464477539063e-08,23,1536,0.0003,-10.233842849731445,80.0,-446.7894287109375,-182.8891038,2657.2578125
0.5,0.0,31,2048,0.0003,-10.203926086425781,58.0,-376.79693603515625,-186.2180036,1866.2381591796875
0.5,0.0,39,2560,0.0003,-10.18746566772461,80.0,-376.4490661621094,-183.183348,2010.609619140625
0.5,-1.1920928955078125e-07,47,3072,0.0003,-10.171223640441895,80.0,-442.666015625,-182.9035442,2606.757080078125
0.5,0.0,55,3584,0.0003,-10.16633129119873,58.0,-379.57342529296875,-186.12383340000002,1948.66455078125
0.5,0.0,63,4096,0.0003,-10.15328311920166,58.0,-273.0685119628906,-185.73179199999998,1343.6552734375
0.5,0.0,71,4608,0.0003,-10.093599319458008,80.0,-340.38714599609375,-182.71608199999997,1549.9210205078125
```

#### logs_A2C_0.1_baseline/log.txt

```
Logging to ./logs_A2C_0.1_baseline
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -150     |
| time/                 |          |
|    total_timesteps    | 512      |
| train/                |          |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 7        |
|    policy_loss        | -479     |
|    value_loss         | 3.25e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -184     |
| time/                 |          |
|    total_timesteps    | 1024     |
| train/                |          |
|    entropy_loss       | -10.3    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 15       |
|    policy_loss        | -459     |
|    value_loss         | 2.52e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -182     |
| time/                 |          |
|    total_timesteps    | 1536     |
| train/                |          |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 23       |
|    policy_loss        | -426     |
|    value_loss         | 2.38e+03 |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 400      |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 2048     |
| train/                |          |
|    entropy_loss       | -10.2    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 31       |
|    policy_loss        | -392     |
|    value_loss         | 2.02e+03 |
------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 400       |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 2560      |
| train/                |           |
|    entropy_loss       | -10.2     |
|    explained_variance | -2.38e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 39        |
|    policy_loss        | -287      |
|    value_loss         | 1.29e+03  |
-------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 3072      |
| train/                |           |
|    entropy_loss       | -10.1     |
|    explained_variance | -1.19e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 47        |
|    policy_loss        | -442      |
|    value_loss         | 2.56e+03  |
-------------------------------------
-------------------------------------
| eval/                 |           |
|    mean_ep_length     | 80        |
|    mean_reward        | -183      |
| time/                 |           |
|    total_timesteps    | 3584      |
| train/                |           |
|    entropy_loss       | -10.1     |
|    explained_variance | -1.19e-07 |
|    learning_rate      | 0.0003    |
|    n_updates          | 55        |
|    policy_loss        | -403      |
|    value_loss         | 2.15e+03  |
-------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -183     |
| time/                 |          |
|    total_timesteps    | 4096     |
| train/                |          |
|    entropy_loss       | -10.1    |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 63       |
|    policy_loss        | -219     |
|    value_loss         | 948      |
------------------------------------
------------------------------------
| eval/                 |          |
|    mean_ep_length     | 80       |
|    mean_reward        | -184     |
| time/                 |          |
|    total_timesteps    | 4608     |
| train/                |          |
|    entropy_loss       | -10      |
|    explained_variance | 0        |
|    learning_rate      | 0.0003   |
|    n_updates          | 71       |
|    policy_loss        | -329     |
|    value_loss         | 1.44e+03 |
------------------------------------
```

#### logs_A2C_0.1_baseline/progress.csv

```
train/explained_variance,train/n_updates,time/total_timesteps,train/learning_rate,train/entropy_loss,eval/mean_ep_length,train/policy_loss,eval/mean_reward,train/value_loss
0.0,7,512,0.0003,-10.326486587524414,80.0,-478.83917236328125,-149.5028366,3254.569580078125
0.0,15,1024,0.0003,-10.276077270507812,80.0,-458.8841552734375,-183.9525932,2516.883544921875
0.0,23,1536,0.0003,-10.239139556884766,80.0,-425.8084716796875,-182.48456620000002,2381.38427734375
0.0,31,2048,0.0003,-10.205982208251953,400.0,-392.40582275390625,-182.5431266,2024.3316650390625
-2.384185791015625e-07,39,2560,0.0003,-10.161223411560059,400.0,-287.13555908203125,-183.1880178,1294.3544921875
-1.1920928955078125e-07,47,3072,0.0003,-10.133308410644531,80.0,-441.91717529296875,-182.645351,2560.59765625
-1.1920928955078125e-07,55,3584,0.0003,-10.125617980957031,80.0,-402.94427490234375,-182.8246648,2145.5439453125
0.0,63,4096,0.0003,-10.069011688232422,80.0,-219.4763946533203,-183.408775,947.8653564453125
0.0,71,4608,0.0003,-10.013643264770508,80.0,-329.0965576171875,-183.73038119999998,1439.1522216796875
```

#### logs_PPO_0.001_baseline/log.txt

```
Logging to ./logs_PPO_0.001_baseline
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 14       |
|    ep_rew_mean     | -80.9    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 425      |
|    total_timesteps | 64       |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 14         |
|    ep_rew_mean          | -80.9      |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 2          |
|    time_elapsed         | 886        |
|    total_timesteps      | 128        |
| train/                  |            |
|    approx_kl            | 0.01066758 |
|    clip_fraction        | 0.0234     |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.4      |
|    explained_variance   | 0.00236    |
|    learning_rate        | 0.0003     |
|    loss                 | 676        |
|    n_updates            | 10         |
|    policy_gradient_loss | -0.0449    |
|    value_loss           | 1.4e+03    |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 22          |
|    ep_rew_mean          | -125        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 3           |
|    time_elapsed         | 1349        |
|    total_timesteps      | 192         |
| train/                  |             |
|    approx_kl            | 0.010377924 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 698         |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.0453     |
|    value_loss           | 1.41e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 22          |
|    ep_rew_mean          | -125        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 4           |
|    time_elapsed         | 1764        |
|    total_timesteps      | 256         |
| train/                  |             |
|    approx_kl            | 0.007921223 |
|    clip_fraction        | 0.025       |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 750         |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0414     |
|    value_loss           | 1.51e+03    |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 45.2       |
|    ep_rew_mean          | -240       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 5          |
|    time_elapsed         | 2178       |
|    total_timesteps      | 320        |
| train/                  |            |
|    approx_kl            | 0.00744278 |
|    clip_fraction        | 0.00781    |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.3      |
|    explained_variance   | 5.96e-08   |
|    learning_rate        | 0.0003     |
|    loss                 | 703        |
|    n_updates            | 40         |
|    policy_gradient_loss | -0.0396    |
|    value_loss           | 1.42e+03   |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 52.4        |
|    ep_rew_mean          | -282        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 6           |
|    time_elapsed         | 2657        |
|    total_timesteps      | 384         |
| train/                  |             |
|    approx_kl            | 0.008388444 |
|    clip_fraction        | 0.0281      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 527         |
|    n_updates            | 50          |
|    policy_gradient_loss | -0.0381     |
|    value_loss           | 1.06e+03    |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 48.5       |
|    ep_rew_mean          | -258       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 7          |
|    time_elapsed         | 3096       |
|    total_timesteps      | 448        |
| train/                  |            |
|    approx_kl            | 0.00933009 |
|    clip_fraction        | 0.0328     |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.3      |
|    explained_variance   | 5.96e-08   |
|    learning_rate        | 0.0003     |
|    loss                 | 1.02e+03   |
|    n_updates            | 60         |
|    policy_gradient_loss | -0.0388    |
|    value_loss           | 2.06e+03   |
----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 133         |
|    mean_reward          | -309        |
| time/                   |             |
|    total_timesteps      | 512         |
| train/                  |             |
|    approx_kl            | 0.005723533 |
|    clip_fraction        | 0.00469     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 647         |
|    n_updates            | 70          |
|    policy_gradient_loss | -0.0348     |
|    value_loss           | 1.3e+03     |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 47.1     |
|    ep_rew_mean     | -256     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6320     |
|    total_timesteps | 512      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.1        |
|    ep_rew_mean          | -239        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 9           |
|    time_elapsed         | 6729        |
|    total_timesteps      | 576         |
| train/                  |             |
|    approx_kl            | 0.006100571 |
|    clip_fraction        | 0.0141      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 870         |
|    n_updates            | 80          |
|    policy_gradient_loss | -0.0329     |
|    value_loss           | 1.75e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 50.9         |
|    ep_rew_mean          | -272         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 10           |
|    time_elapsed         | 7135         |
|    total_timesteps      | 640          |
| train/                  |              |
|    approx_kl            | 0.0069114286 |
|    clip_fraction        | 0.0141       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.2        |
|    explained_variance   | 1.79e-07     |
|    learning_rate        | 0.0003       |
|    loss                 | 649          |
|    n_updates            | 90           |
|    policy_gradient_loss | -0.0343      |
|    value_loss           | 1.31e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 50.9        |
|    ep_rew_mean          | -272        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 11          |
|    time_elapsed         | 7490        |
|    total_timesteps      | 704         |
| train/                  |             |
|    approx_kl            | 0.010347788 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 569         |
|    n_updates            | 100         |
|    policy_gradient_loss | -0.0433     |
|    value_loss           | 1.14e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 50.9        |
|    ep_rew_mean          | -272        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 12          |
|    time_elapsed         | 7883        |
|    total_timesteps      | 768         |
| train/                  |             |
|    approx_kl            | 0.011766544 |
|    clip_fraction        | 0.0578      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 378         |
|    n_updates            | 110         |
|    policy_gradient_loss | -0.0467     |
|    value_loss           | 763         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 53.8        |
|    ep_rew_mean          | -285        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 13          |
|    time_elapsed         | 8283        |
|    total_timesteps      | 832         |
| train/                  |             |
|    approx_kl            | 0.007705616 |
|    clip_fraction        | 0.025       |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 534         |
|    n_updates            | 120         |
|    policy_gradient_loss | -0.0348     |
|    value_loss           | 1.07e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56.4        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 14          |
|    time_elapsed         | 8663        |
|    total_timesteps      | 896         |
| train/                  |             |
|    approx_kl            | 0.010335488 |
|    clip_fraction        | 0.0359      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 510         |
|    n_updates            | 130         |
|    policy_gradient_loss | -0.0399     |
|    value_loss           | 1.03e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.7        |
|    ep_rew_mean          | -303        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 15          |
|    time_elapsed         | 9052        |
|    total_timesteps      | 960         |
| train/                  |             |
|    approx_kl            | 0.012943729 |
|    clip_fraction        | 0.0516      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 386         |
|    n_updates            | 140         |
|    policy_gradient_loss | -0.0521     |
|    value_loss           | 778         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 200         |
|    mean_reward          | -271        |
| time/                   |             |
|    total_timesteps      | 1024        |
| train/                  |             |
|    approx_kl            | 0.014150036 |
|    clip_fraction        | 0.0719      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 441         |
|    n_updates            | 150         |
|    policy_gradient_loss | -0.0382     |
|    value_loss           | 887         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 61       |
|    ep_rew_mean     | -307     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 12151    |
|    total_timesteps | 1024     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.8        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 17          |
|    time_elapsed         | 12520       |
|    total_timesteps      | 1088        |
| train/                  |             |
|    approx_kl            | 0.007732114 |
|    clip_fraction        | 0.0156      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 386         |
|    n_updates            | 160         |
|    policy_gradient_loss | -0.037      |
|    value_loss           | 776         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.8        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 18          |
|    time_elapsed         | 12839       |
|    total_timesteps      | 1152        |
| train/                  |             |
|    approx_kl            | 0.012087294 |
|    clip_fraction        | 0.05        |
|    clip_range           | 0.2         |
|    entropy_loss         | -10         |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 386         |
|    n_updates            | 170         |
|    policy_gradient_loss | -0.0469     |
|    value_loss           | 777         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.8        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 19          |
|    time_elapsed         | 13203       |
|    total_timesteps      | 1216        |
| train/                  |             |
|    approx_kl            | 0.010458134 |
|    clip_fraction        | 0.0328      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10         |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 393         |
|    n_updates            | 180         |
|    policy_gradient_loss | -0.0413     |
|    value_loss           | 791         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 62.3        |
|    ep_rew_mean          | -296        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 20          |
|    time_elapsed         | 13547       |
|    total_timesteps      | 1280        |
| train/                  |             |
|    approx_kl            | 0.013777612 |
|    clip_fraction        | 0.0625      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.99       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 319         |
|    n_updates            | 190         |
|    policy_gradient_loss | -0.0456     |
|    value_loss           | 642         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 62.3        |
|    ep_rew_mean          | -296        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 21          |
|    time_elapsed         | 13859       |
|    total_timesteps      | 1344        |
| train/                  |             |
|    approx_kl            | 0.015737891 |
|    clip_fraction        | 0.0641      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.92       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 281         |
|    n_updates            | 200         |
|    policy_gradient_loss | -0.0509     |
|    value_loss           | 565         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 60.5        |
|    ep_rew_mean          | -285        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 22          |
|    time_elapsed         | 14151       |
|    total_timesteps      | 1408        |
| train/                  |             |
|    approx_kl            | 0.019862168 |
|    clip_fraction        | 0.0984      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.84       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 274         |
|    n_updates            | 210         |
|    policy_gradient_loss | -0.0512     |
|    value_loss           | 553         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 62.6        |
|    ep_rew_mean          | -289        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 23          |
|    time_elapsed         | 14438       |
|    total_timesteps      | 1472        |
| train/                  |             |
|    approx_kl            | 0.018446798 |
|    clip_fraction        | 0.0906      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.74       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 281         |
|    n_updates            | 220         |
|    policy_gradient_loss | -0.0549     |
|    value_loss           | 566         |
-----------------------------------------
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 400        |
|    mean_reward          | -182       |
| time/                   |            |
|    total_timesteps      | 1536       |
| train/                  |            |
|    approx_kl            | 0.03601335 |
|    clip_fraction        | 0.189      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.61      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 170        |
|    n_updates            | 230        |
|    policy_gradient_loss | -0.0586    |
|    value_loss           | 344        |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 62.6     |
|    ep_rew_mean     | -289     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 17536    |
|    total_timesteps | 1536     |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 62.6       |
|    ep_rew_mean          | -289       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 25         |
|    time_elapsed         | 17767      |
|    total_timesteps      | 1600       |
| train/                  |            |
|    approx_kl            | 0.02479199 |
|    clip_fraction        | 0.141      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.44      |
|    explained_variance   | 1.79e-07   |
|    learning_rate        | 0.0003     |
|    loss                 | 118        |
|    n_updates            | 240        |
|    policy_gradient_loss | -0.0578    |
|    value_loss           | 239        |
----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 61.5       |
|    ep_rew_mean          | -279       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 26         |
|    time_elapsed         | 18015      |
|    total_timesteps      | 1664       |
| train/                  |            |
|    approx_kl            | 0.03269295 |
|    clip_fraction        | 0.191      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.33      |
|    explained_variance   | 1.79e-07   |
|    learning_rate        | 0.0003     |
|    loss                 | 106        |
|    n_updates            | 250        |
|    policy_gradient_loss | -0.0655    |
|    value_loss           | 215        |
----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 61.5       |
|    ep_rew_mean          | -279       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 27         |
|    time_elapsed         | 18266      |
|    total_timesteps      | 1728       |
| train/                  |            |
|    approx_kl            | 0.01563073 |
|    clip_fraction        | 0.0672     |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.29      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 156        |
|    n_updates            | 260        |
|    policy_gradient_loss | -0.0505    |
|    value_loss           | 314        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 67.4        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 28          |
|    time_elapsed         | 18499       |
|    total_timesteps      | 1792        |
| train/                  |             |
|    approx_kl            | 0.018480416 |
|    clip_fraction        | 0.103       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.24       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 145         |
|    n_updates            | 270         |
|    policy_gradient_loss | -0.0573     |
|    value_loss           | 294         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 67.4        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 29          |
|    time_elapsed         | 18711       |
|    total_timesteps      | 1856        |
| train/                  |             |
|    approx_kl            | 0.021068525 |
|    clip_fraction        | 0.108       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.26       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 70.6        |
|    n_updates            | 280         |
|    policy_gradient_loss | -0.0517     |
|    value_loss           | 143         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 70.3        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 30          |
|    time_elapsed         | 18896       |
|    total_timesteps      | 1920        |
| train/                  |             |
|    approx_kl            | 0.024044054 |
|    clip_fraction        | 0.111       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.28       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 116         |
|    n_updates            | 290         |
|    policy_gradient_loss | -0.0589     |
|    value_loss           | 235         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 70.3       |
|    ep_rew_mean          | -287       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 31         |
|    time_elapsed         | 19118      |
|    total_timesteps      | 1984       |
| train/                  |            |
|    approx_kl            | 0.03198516 |
|    clip_fraction        | 0.188      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.13      |
|    explained_variance   | -1.19e-07  |
|    learning_rate        | 0.0003     |
|    loss                 | 82.8       |
|    n_updates            | 300        |
|    policy_gradient_loss | -0.0684    |
|    value_loss           | 168        |
----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 2048        |
| train/                  |             |
|    approx_kl            | 0.015331095 |
|    clip_fraction        | 0.0672      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.02       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 111         |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.0422     |
|    value_loss           | 224         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 70.3     |
|    ep_rew_mean     | -287     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 22181    |
|    total_timesteps | 2048     |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 70.3       |
|    ep_rew_mean          | -287       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 33         |
|    time_elapsed         | 22409      |
|    total_timesteps      | 2112       |
| train/                  |            |
|    approx_kl            | 0.03908418 |
|    clip_fraction        | 0.223      |
|    clip_range           | 0.2        |
|    entropy_loss         | -8.91      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 77.4       |
|    n_updates            | 320        |
|    policy_gradient_loss | -0.0671    |
|    value_loss           | 157        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 70.3        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 34          |
|    time_elapsed         | 22592       |
|    total_timesteps      | 2176        |
| train/                  |             |
|    approx_kl            | 0.022339385 |
|    clip_fraction        | 0.113       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.77       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 88.9        |
|    n_updates            | 330         |
|    policy_gradient_loss | -0.0577     |
|    value_loss           | 180         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 70.3        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 35          |
|    time_elapsed         | 22791       |
|    total_timesteps      | 2240        |
| train/                  |             |
|    approx_kl            | 0.022428777 |
|    clip_fraction        | 0.108       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.63       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 70.7        |
|    n_updates            | 340         |
|    policy_gradient_loss | -0.0655     |
|    value_loss           | 144         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 70.3        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 36          |
|    time_elapsed         | 22982       |
|    total_timesteps      | 2304        |
| train/                  |             |
|    approx_kl            | 0.016905047 |
|    clip_fraction        | 0.0844      |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.53       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 54.6        |
|    n_updates            | 350         |
|    policy_gradient_loss | -0.0531     |
|    value_loss           | 111         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 74.4       |
|    ep_rew_mean          | -287       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 37         |
|    time_elapsed         | 23153      |
|    total_timesteps      | 2368       |
| train/                  |            |
|    approx_kl            | 0.02328303 |
|    clip_fraction        | 0.117      |
|    clip_range           | 0.2        |
|    entropy_loss         | -8.45      |
|    explained_variance   | -1.19e-07  |
|    learning_rate        | 0.0003     |
|    loss                 | 57.7       |
|    n_updates            | 360        |
|    policy_gradient_loss | -0.0585    |
|    value_loss           | 117        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 77.8        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 38          |
|    time_elapsed         | 23304       |
|    total_timesteps      | 2432        |
| train/                  |             |
|    approx_kl            | 0.044985138 |
|    clip_fraction        | 0.253       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.25       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 49.8        |
|    n_updates            | 370         |
|    policy_gradient_loss | -0.0642     |
|    value_loss           | 101         |
-----------------------------------------
---------------------------------------
| rollout/                |           |
|    ep_len_mean          | 77.8      |
|    ep_rew_mean          | -287      |
| time/                   |           |
|    fps                  | 0         |
|    iterations           | 39        |
|    time_elapsed         | 23470     |
|    total_timesteps      | 2496      |
| train/                  |           |
|    approx_kl            | 0.0398153 |
|    clip_fraction        | 0.228     |
|    clip_range           | 0.2       |
|    entropy_loss         | -8.17     |
|    explained_variance   | 0         |
|    learning_rate        | 0.0003    |
|    loss                 | 23.1      |
|    n_updates            | 380       |
|    policy_gradient_loss | -0.0748   |
|    value_loss           | 47.3      |
---------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 2560        |
| train/                  |             |
|    approx_kl            | 0.027839454 |
|    clip_fraction        | 0.164       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.11       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 41.4        |
|    n_updates            | 390         |
|    policy_gradient_loss | -0.0597     |
|    value_loss           | 84.6        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 81.6     |
|    ep_rew_mean     | -287     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 26452    |
|    total_timesteps | 2560     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 81.6        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 41          |
|    time_elapsed         | 26623       |
|    total_timesteps      | 2624        |
| train/                  |             |
|    approx_kl            | 0.020205291 |
|    clip_fraction        | 0.102       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.99       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 40.1        |
|    n_updates            | 400         |
|    policy_gradient_loss | -0.0571     |
|    value_loss           | 81.9        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 85.1       |
|    ep_rew_mean          | -286       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 42         |
|    time_elapsed         | 26801      |
|    total_timesteps      | 2688       |
| train/                  |            |
|    approx_kl            | 0.02349067 |
|    clip_fraction        | 0.128      |
|    clip_range           | 0.2        |
|    entropy_loss         | -7.85      |
|    explained_variance   | 5.96e-08   |
|    learning_rate        | 0.0003     |
|    loss                 | 50.4       |
|    n_updates            | 410        |
|    policy_gradient_loss | -0.0545    |
|    value_loss           | 103        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 43          |
|    time_elapsed         | 26928       |
|    total_timesteps      | 2752        |
| train/                  |             |
|    approx_kl            | 0.024109166 |
|    clip_fraction        | 0.127       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.7        |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 41.9        |
|    n_updates            | 420         |
|    policy_gradient_loss | -0.0597     |
|    value_loss           | 85.3        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 85.1       |
|    ep_rew_mean          | -286       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 44         |
|    time_elapsed         | 27081      |
|    total_timesteps      | 2816       |
| train/                  |            |
|    approx_kl            | 0.04591059 |
|    clip_fraction        | 0.244      |
|    clip_range           | 0.2        |
|    entropy_loss         | -7.55      |
|    explained_variance   | -1.19e-07  |
|    learning_rate        | 0.0003     |
|    loss                 | 19         |
|    n_updates            | 430        |
|    policy_gradient_loss | -0.0774    |
|    value_loss           | 39.2       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 45          |
|    time_elapsed         | 27229       |
|    total_timesteps      | 2880        |
| train/                  |             |
|    approx_kl            | 0.021040674 |
|    clip_fraction        | 0.0969      |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.42       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 41          |
|    n_updates            | 440         |
|    policy_gradient_loss | -0.0538     |
|    value_loss           | 83.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 46          |
|    time_elapsed         | 27338       |
|    total_timesteps      | 2944        |
| train/                  |             |
|    approx_kl            | 0.039559312 |
|    clip_fraction        | 0.202       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.26       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 30.1        |
|    n_updates            | 450         |
|    policy_gradient_loss | -0.0767     |
|    value_loss           | 61.6        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 47          |
|    time_elapsed         | 27450       |
|    total_timesteps      | 3008        |
| train/                  |             |
|    approx_kl            | 0.049130376 |
|    clip_fraction        | 0.248       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.99       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 9.9         |
|    n_updates            | 460         |
|    policy_gradient_loss | -0.0805     |
|    value_loss           | 20.8        |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 3072        |
| train/                  |             |
|    approx_kl            | 0.029982774 |
|    clip_fraction        | 0.175       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.74       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 14          |
|    n_updates            | 470         |
|    policy_gradient_loss | -0.0583     |
|    value_loss           | 29.1        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 85.1     |
|    ep_rew_mean     | -286     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 30398    |
|    total_timesteps | 3072     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 49          |
|    time_elapsed         | 30510       |
|    total_timesteps      | 3136        |
| train/                  |             |
|    approx_kl            | 0.031461973 |
|    clip_fraction        | 0.17        |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.62       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 17.1        |
|    n_updates            | 480         |
|    policy_gradient_loss | -0.0635     |
|    value_loss           | 35.4        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 50          |
|    time_elapsed         | 30617       |
|    total_timesteps      | 3200        |
| train/                  |             |
|    approx_kl            | 0.031113055 |
|    clip_fraction        | 0.163       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.55       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 14.9        |
|    n_updates            | 490         |
|    policy_gradient_loss | -0.0601     |
|    value_loss           | 30.9        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 51          |
|    time_elapsed         | 30721       |
|    total_timesteps      | 3264        |
| train/                  |             |
|    approx_kl            | 0.033674337 |
|    clip_fraction        | 0.188       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.52       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 12.8        |
|    n_updates            | 500         |
|    policy_gradient_loss | -0.0733     |
|    value_loss           | 26.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 52          |
|    time_elapsed         | 30827       |
|    total_timesteps      | 3328        |
| train/                  |             |
|    approx_kl            | 0.033911195 |
|    clip_fraction        | 0.188       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.45       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 14          |
|    n_updates            | 510         |
|    policy_gradient_loss | -0.0627     |
|    value_loss           | 29.1        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 85.1       |
|    ep_rew_mean          | -286       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 53         |
|    time_elapsed         | 30924      |
|    total_timesteps      | 3392       |
| train/                  |            |
|    approx_kl            | 0.03337422 |
|    clip_fraction        | 0.203      |
|    clip_range           | 0.2        |
|    entropy_loss         | -6.36      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 12.1       |
|    n_updates            | 520        |
|    policy_gradient_loss | -0.0687    |
|    value_loss           | 25.3       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 91.5        |
|    ep_rew_mean          | -284        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 54          |
|    time_elapsed         | 31018       |
|    total_timesteps      | 3456        |
| train/                  |             |
|    approx_kl            | 0.038641103 |
|    clip_fraction        | 0.223       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.3        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 11.4        |
|    n_updates            | 530         |
|    policy_gradient_loss | -0.0752     |
|    value_loss           | 23.8        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 97.1       |
|    ep_rew_mean          | -283       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 55         |
|    time_elapsed         | 31109      |
|    total_timesteps      | 3520       |
| train/                  |            |
|    approx_kl            | 0.06282024 |
|    clip_fraction        | 0.339      |
|    clip_range           | 0.2        |
|    entropy_loss         | -6.15      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 17.2       |
|    n_updates            | 540        |
|    policy_gradient_loss | -0.0763    |
|    value_loss           | 35         |
----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 3584        |
| train/                  |             |
|    approx_kl            | 0.018131059 |
|    clip_fraction        | 0.0922      |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.04       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 12.6        |
|    n_updates            | 550         |
|    policy_gradient_loss | -0.0407     |
|    value_loss           | 26          |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 97.1     |
|    ep_rew_mean     | -283     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 34031    |
|    total_timesteps | 3584     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 97.1        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 57          |
|    time_elapsed         | 34122       |
|    total_timesteps      | 3648        |
| train/                  |             |
|    approx_kl            | 0.027804287 |
|    clip_fraction        | 0.13        |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.03       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 10.1        |
|    n_updates            | 560         |
|    policy_gradient_loss | -0.0595     |
|    value_loss           | 21.2        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 58          |
|    time_elapsed         | 34216       |
|    total_timesteps      | 3712        |
| train/                  |             |
|    approx_kl            | 0.030136464 |
|    clip_fraction        | 0.152       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.06       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 9.34        |
|    n_updates            | 570         |
|    policy_gradient_loss | -0.058      |
|    value_loss           | 19.6        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 59          |
|    time_elapsed         | 34310       |
|    total_timesteps      | 3776        |
| train/                  |             |
|    approx_kl            | 0.063181914 |
|    clip_fraction        | 0.328       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.05       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 16.8        |
|    n_updates            | 580         |
|    policy_gradient_loss | -0.0723     |
|    value_loss           | 34          |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 60          |
|    time_elapsed         | 34402       |
|    total_timesteps      | 3840        |
| train/                  |             |
|    approx_kl            | 0.017944407 |
|    clip_fraction        | 0.1         |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.03       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 9.79        |
|    n_updates            | 590         |
|    policy_gradient_loss | -0.0312     |
|    value_loss           | 20.4        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 61          |
|    time_elapsed         | 34494       |
|    total_timesteps      | 3904        |
| train/                  |             |
|    approx_kl            | 0.019797083 |
|    clip_fraction        | 0.0953      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.95       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 7.53        |
|    n_updates            | 600         |
|    policy_gradient_loss | -0.0449     |
|    value_loss           | 15.9        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -283       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 62         |
|    time_elapsed         | 34595      |
|    total_timesteps      | 3968       |
| train/                  |            |
|    approx_kl            | 0.02780189 |
|    clip_fraction        | 0.15       |
|    clip_range           | 0.2        |
|    entropy_loss         | -5.97      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 8.06       |
|    n_updates            | 610        |
|    policy_gradient_loss | -0.0589    |
|    value_loss           | 17         |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 63          |
|    time_elapsed         | 34689       |
|    total_timesteps      | 4032        |
| train/                  |             |
|    approx_kl            | 0.021823034 |
|    clip_fraction        | 0.116       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.9        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 8.68        |
|    n_updates            | 620         |
|    policy_gradient_loss | -0.05       |
|    value_loss           | 18.2        |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.026968703 |
|    clip_fraction        | 0.114       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.84       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 14.3        |
|    n_updates            | 630         |
|    policy_gradient_loss | -0.0491     |
|    value_loss           | 29.2        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 111      |
|    ep_rew_mean     | -281     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 37613    |
|    total_timesteps | 4096     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 65          |
|    time_elapsed         | 37709       |
|    total_timesteps      | 4160        |
| train/                  |             |
|    approx_kl            | 0.023035713 |
|    clip_fraction        | 0.119       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.83       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 8.84        |
|    n_updates            | 640         |
|    policy_gradient_loss | -0.0601     |
|    value_loss           | 18.6        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 66          |
|    time_elapsed         | 37802       |
|    total_timesteps      | 4224        |
| train/                  |             |
|    approx_kl            | 0.024265915 |
|    clip_fraction        | 0.127       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.78       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 9.67        |
|    n_updates            | 650         |
|    policy_gradient_loss | -0.056      |
|    value_loss           | 20.3        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 67          |
|    time_elapsed         | 37892       |
|    total_timesteps      | 4288        |
| train/                  |             |
|    approx_kl            | 0.023206677 |
|    clip_fraction        | 0.106       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.77       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 9.88        |
|    n_updates            | 660         |
|    policy_gradient_loss | -0.0569     |
|    value_loss           | 20.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 68          |
|    time_elapsed         | 37983       |
|    total_timesteps      | 4352        |
| train/                  |             |
|    approx_kl            | 0.047301535 |
|    clip_fraction        | 0.244       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.67       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 8.52        |
|    n_updates            | 670         |
|    policy_gradient_loss | -0.0642     |
|    value_loss           | 17.9        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 69          |
|    time_elapsed         | 38074       |
|    total_timesteps      | 4416        |
| train/                  |             |
|    approx_kl            | 0.035239927 |
|    clip_fraction        | 0.219       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.55       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 6.91        |
|    n_updates            | 680         |
|    policy_gradient_loss | -0.0736     |
|    value_loss           | 14.7        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 111        |
|    ep_rew_mean          | -281       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 70         |
|    time_elapsed         | 38166      |
|    total_timesteps      | 4480       |
| train/                  |            |
|    approx_kl            | 0.02203437 |
|    clip_fraction        | 0.116      |
|    clip_range           | 0.2        |
|    entropy_loss         | -5.51      |
|    explained_variance   | 5.96e-08   |
|    learning_rate        | 0.0003     |
|    loss                 | 7.92       |
|    n_updates            | 690        |
|    policy_gradient_loss | -0.0419    |
|    value_loss           | 16.7       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 71          |
|    time_elapsed         | 38265       |
|    total_timesteps      | 4544        |
| train/                  |             |
|    approx_kl            | 0.026603311 |
|    clip_fraction        | 0.119       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.53       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 8.03        |
|    n_updates            | 700         |
|    policy_gradient_loss | -0.0519     |
|    value_loss           | 16.9        |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 4608        |
| train/                  |             |
|    approx_kl            | 0.022957962 |
|    clip_fraction        | 0.106       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.44       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 9.19        |
|    n_updates            | 710         |
|    policy_gradient_loss | -0.0434     |
|    value_loss           | 19.3        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 111      |
|    ep_rew_mean     | -281     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 41197    |
|    total_timesteps | 4608     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 73          |
|    time_elapsed         | 41288       |
|    total_timesteps      | 4672        |
| train/                  |             |
|    approx_kl            | 0.019447748 |
|    clip_fraction        | 0.0984      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.39       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 7.97        |
|    n_updates            | 720         |
|    policy_gradient_loss | -0.0387     |
|    value_loss           | 16.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 74          |
|    time_elapsed         | 41380       |
|    total_timesteps      | 4736        |
| train/                  |             |
|    approx_kl            | 0.018282127 |
|    clip_fraction        | 0.0875      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.28       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 7.59        |
|    n_updates            | 730         |
|    policy_gradient_loss | -0.0435     |
|    value_loss           | 16          |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 75          |
|    time_elapsed         | 41475       |
|    total_timesteps      | 4800        |
| train/                  |             |
|    approx_kl            | 0.020621981 |
|    clip_fraction        | 0.0984      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.22       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 6.9         |
|    n_updates            | 740         |
|    policy_gradient_loss | -0.0491     |
|    value_loss           | 14.6        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 76          |
|    time_elapsed         | 41570       |
|    total_timesteps      | 4864        |
| train/                  |             |
|    approx_kl            | 0.026766445 |
|    clip_fraction        | 0.141       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.11       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 7.85        |
|    n_updates            | 750         |
|    policy_gradient_loss | -0.0537     |
|    value_loss           | 16.5        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 77          |
|    time_elapsed         | 41661       |
|    total_timesteps      | 4928        |
| train/                  |             |
|    approx_kl            | 0.024439313 |
|    clip_fraction        | 0.141       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5          |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 6.92        |
|    n_updates            | 760         |
|    policy_gradient_loss | -0.0523     |
|    value_loss           | 14.6        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 111         |
|    ep_rew_mean          | -281        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 78          |
|    time_elapsed         | 41753       |
|    total_timesteps      | 4992        |
| train/                  |             |
|    approx_kl            | 0.020683143 |
|    clip_fraction        | 0.103       |
|    clip_range           | 0.2         |
|    entropy_loss         | -4.87       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 6.35        |
|    n_updates            | 770         |
|    policy_gradient_loss | -0.0444     |
|    value_loss           | 13.5        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 127         |
|    ep_rew_mean          | -278        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 79          |
|    time_elapsed         | 41844       |
|    total_timesteps      | 5056        |
| train/                  |             |
|    approx_kl            | 0.017178137 |
|    clip_fraction        | 0.0766      |
|    clip_range           | 0.2         |
|    entropy_loss         | -4.75       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 6.03        |
|    n_updates            | 780         |
|    policy_gradient_loss | -0.0448     |
|    value_loss           | 12.8        |
-----------------------------------------
```

#### logs_PPO_0.001_baseline/progress.csv

```
time/iterations,time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,time/fps,time/time_elapsed,train/explained_variance,train/approx_kl,train/clip_fraction,train/loss,train/policy_gradient_loss,train/n_updates,train/learning_rate,train/entropy_loss,train/clip_range,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,64,14.0,-80.931749,0,425,,,,,,,,,,,,
2,128,14.0,-80.931749,0,886,0.0023565292358398438,0.01066758,0.0234375,676.1712036132812,-0.04487760029733181,10,0.0003,-10.35523042678833,0.2,1396.2471923828125,,
3,192,22.0,-125.0221945,0,1349,1.1920928955078125e-07,0.010377924,0.03125,698.4393920898438,-0.04525396507233381,20,0.0003,-10.344770812988282,0.2,1411.720654296875,,
4,256,22.0,-125.0221945,0,1764,-1.1920928955078125e-07,0.007921223,0.025,749.556884765625,-0.04136621728539467,30,0.0003,-10.328810596466065,0.2,1512.6734497070313,,
5,320,45.25,-239.65107775,0,2178,5.960464477539063e-08,0.00744278,0.0078125,702.8438720703125,-0.0396339075639844,40,0.0003,-10.316028213500976,0.2,1417.1686401367188,,
6,384,52.4,-281.76184700000005,0,2657,0.0,0.008388444,0.028125,526.6373901367188,-0.03812917172908783,50,0.0003,-10.307358264923096,0.2,1062.5441040039063,,
7,448,48.5,-257.9257548333334,0,3096,5.960464477539063e-08,0.00933009,0.0328125,1022.2207641601562,-0.03882765434682369,60,0.0003,-10.292426490783692,0.2,2056.247106933594,,
,512,,,,,1.7881393432617188e-07,0.005723533,0.0046875,647.0514526367188,-0.034751695767045024,70,0.0003,-10.278273487091065,0.2,1302.8543212890625,-309.0159138,132.6
8,512,47.125,-255.68802499999998,0,6320,,,,,,,,,,,,
9,576,44.111111111111114,-239.11681011111108,0,6729,5.960464477539063e-08,0.006100571,0.0140625,869.9312744140625,-0.03285973835736513,80,0.0003,-10.277079677581787,0.2,1749.1548095703124,,
10,640,50.90909090909091,-271.69744490909085,0,7135,1.7881393432617188e-07,0.0069114286,0.0140625,649.0543823242188,-0.03431780710816383,90,0.0003,-10.244952201843262,0.2,1305.5642578125,,
11,704,50.90909090909091,-271.69744490909085,0,7490,-1.1920928955078125e-07,0.010347788,0.03125,568.7550048828125,-0.04330454403534532,100,0.0003,-10.217523860931397,0.2,1144.1238403320312,,
12,768,50.90909090909091,-271.69744490909085,0,7883,1.1920928955078125e-07,0.011766544,0.0578125,378.4637145996094,-0.04666231237351894,110,0.0003,-10.186844158172608,0.2,762.6350646972656,,
13,832,53.75,-285.31403291666663,0,8283,0.0,0.007705616,0.025,533.952880859375,-0.034754921682178974,120,0.0003,-10.17165403366089,0.2,1074.155810546875,,
14,896,56.38461538461539,-293.2452416923077,0,8663,-1.1920928955078125e-07,0.010335488,0.0359375,510.0661926269531,-0.03992632366716862,130,0.0003,-10.157701873779297,0.2,1025.8604248046875,,
15,960,58.714285714285715,-302.8134945714286,0,9052,1.1920928955078125e-07,0.012943729,0.0515625,386.4285583496094,-0.0521384509280324,140,0.0003,-10.125921726226807,0.2,777.9657531738281,,
,1024,,,,,0.0,0.014150036,0.071875,440.68487548828125,-0.0381946325302124,150,0.0003,-10.089711380004882,0.2,886.8273681640625,-271.1628538,200.0
16,1024,61.0,-307.2428283333333,0,12151,,,,,,,,,,,,
17,1088,57.75,-289.95158987499997,0,12520,1.1920928955078125e-07,0.007732114,0.015625,385.7244567871094,-0.03696484062820673,160,0.0003,-10.069101715087891,0.2,776.2856750488281,,
18,1152,57.75,-289.95158987499997,0,12839,-1.1920928955078125e-07,0.012087294,0.05,386.2665710449219,-0.04689995963126421,170,0.0003,-10.04317502975464,0.2,777.1708984375,,
19,1216,57.75,-289.95158987499997,0,13203,-1.1920928955078125e-07,0.010458134,0.0328125,392.7790222167969,-0.04125814400613308,180,0.0003,-10.018542766571045,0.2,790.5501220703125,,
20,1280,62.31578947368421,-295.56936684210524,0,13547,0.0,0.013777612,0.0625,318.8118591308594,-0.04561955071985722,190,0.0003,-9.991140937805175,0.2,641.8338989257812,,
21,1344,62.31578947368421,-295.56936684210524,0,13859,0.0,0.015737891,0.0640625,280.6841125488281,-0.05086729675531387,200,0.0003,-9.924252605438232,0.2,565.1430603027344,,
22,1408,60.55,-285.20833964999997,0,14151,-1.1920928955078125e-07,0.019862168,0.0984375,274.27978515625,-0.05115817245095968,210,0.0003,-9.84021463394165,0.2,552.657470703125,,
23,1472,62.57142857142857,-288.8748598095238,0,14438,1.7881393432617188e-07,0.018446798,0.090625,280.86370849609375,-0.054903632309287784,220,0.0003,-9.735432529449463,0.2,565.6432312011718,,
,1536,,,,,0.0,0.03601335,0.1890625,170.29110717773438,-0.05857964679598808,230,0.0003,-9.612387371063232,0.2,343.64564819335936,-182.3833692,400.0
24,1536,62.57142857142857,-288.8748598095238,0,17536,,,,,,,,,,,,
25,1600,62.57142857142857,-288.8748598095238,0,17767,1.7881393432617188e-07,0.02479199,0.140625,118.341552734375,-0.05779877230525017,240,0.0003,-9.444404411315919,0.2,239.3946319580078,,
26,1664,61.54545454545455,-279.2559337272727,0,18015,1.7881393432617188e-07,0.03269295,0.190625,106.31593322753906,-0.06554505676031112,250,0.0003,-9.33465518951416,0.2,215.3544448852539,,
27,1728,61.54545454545455,-279.2559337272727,0,18266,0.0,0.01563073,0.0671875,155.564453125,-0.050496689043939114,260,0.0003,-9.288826179504394,0.2,313.8494079589844,,
28,1792,67.41666666666667,-286.0989309166667,0,18499,1.1920928955078125e-07,0.018480416,0.103125,145.2804412841797,-0.05726625081151724,270,0.0003,-9.237160682678223,0.2,293.5495574951172,,
29,1856,67.41666666666667,-286.0989309166667,0,18711,0.0,0.021068525,0.1078125,70.55125427246094,-0.05170203000307083,280,0.0003,-9.260340023040772,0.2,143.12869415283203,,
30,1920,70.28,-286.5137444,0,18896,0.0,0.024044054,0.1109375,115.9473876953125,-0.058888149447739124,290,0.0003,-9.283000278472901,0.2,234.51983795166015,,
31,1984,70.28,-286.5137444,0,19118,-1.1920928955078125e-07,0.03198516,0.1875,82.8201904296875,-0.06837963983416558,300,0.0003,-9.126916122436523,0.2,167.73932342529298,,
,2048,,,,,1.1920928955078125e-07,0.015331095,0.0671875,110.79556274414062,-0.042187807336449626,310,0.0003,-9.02089490890503,0.2,224.2593017578125,-183.50346019999998,400.0
32,2048,70.28,-286.5137444,0,22181,,,,,,,,,,,,
33,2112,70.28,-286.5137444,0,22409,0.0,0.03908418,0.2234375,77.41106414794922,-0.06710424963384867,320,0.0003,-8.90566053390503,0.2,157.04663848876953,,
34,2176,70.28,-286.5137444,0,22592,-1.1920928955078125e-07,0.022339385,0.1125,88.94001007080078,-0.05772438980638981,330,0.0003,-8.768190383911133,0.2,180.33865814208986,,
35,2240,70.28,-286.5137444,0,22791,0.0,0.022428777,0.1078125,70.68074798583984,-0.06548334956169129,340,0.0003,-8.625446033477782,0.2,143.51550903320313,,
36,2304,70.28,-286.5137444,0,22982,0.0,0.016905047,0.084375,54.575218200683594,-0.05311267301440239,350,0.0003,-8.534307289123536,0.2,111.05680541992187,,
37,2368,74.38461538461539,-287.0148689230769,0,23153,-1.1920928955078125e-07,0.02328303,0.1171875,57.7479248046875,-0.05854783840477466,360,0.0003,-8.451761817932129,0.2,117.43111267089844,,
38,2432,77.81481481481481,-287.4707921111111,0,23304,0.0,0.044985138,0.253125,49.78737258911133,-0.06423825770616531,370,0.0003,-8.246609973907471,0.2,101.04267807006836,,
39,2496,77.81481481481481,-287.4707921111111,0,23470,0.0,0.0398153,0.228125,23.083398818969727,-0.07482690773904324,380,0.0003,-8.166127777099609,0.2,47.27712631225586,,
,2560,,,,,-1.1920928955078125e-07,0.027839454,0.1640625,41.43293762207031,-0.05967306792736053,390,0.0003,-8.113980674743653,0.2,84.55892181396484,-182.77224900000002,400.0
40,2560,81.57142857142857,-286.99539878571426,0,26452,,,,,,,,,,,,
41,2624,81.57142857142857,-286.99539878571426,0,26623,0.0,0.020205291,0.1015625,40.1297721862793,-0.05707571972161531,400,0.0003,-7.9897936344146725,0.2,81.85218505859375,,
42,2688,85.10344827586206,-285.9892193448275,0,26801,5.960464477539063e-08,0.02349067,0.128125,50.37944412231445,-0.054522193502634764,410,0.0003,-7.845370483398438,0.2,102.52865219116211,,
43,2752,85.10344827586206,-285.9892193448275,0,26928,-1.1920928955078125e-07,0.024109166,0.1265625,41.918766021728516,-0.05970817357301712,420,0.0003,-7.703430414199829,0.2,85.26776809692383,,
44,2816,85.10344827586206,-285.9892193448275,0,27081,-1.1920928955078125e-07,0.04591059,0.24375,18.989643096923828,-0.07741947658360004,430,0.0003,-7.549782180786133,0.2,39.246345138549806,,
45,2880,85.10344827586206,-285.9892193448275,0,27229,0.0,0.021040674,0.096875,41.016788482666016,-0.05375417321920395,440,0.0003,-7.424979543685913,0.2,83.66963119506836,,
46,2944,85.10344827586206,-285.9892193448275,0,27338,0.0,0.039559312,0.2015625,30.082746505737305,-0.0767431154847145,450,0.0003,-7.256072854995727,0.2,61.633917236328124,,
47,3008,85.10344827586206,-285.9892193448275,0,27450,-1.1920928955078125e-07,0.049130376,0.2484375,9.902323722839355,-0.08052497822791338,460,0.0003,-6.994247531890869,0.2,20.78281726837158,,
,3072,,,,,0.0,0.029982774,0.175,14.002896308898926,-0.05834050439298153,470,0.0003,-6.7442710399627686,0.2,29.066589164733887,-182.77257920000002,400.0
48,3072,85.10344827586206,-285.9892193448275,0,30398,,,,,,,,,,,,
49,3136,85.10344827586206,-285.9892193448275,0,30510,-1.1920928955078125e-07,0.031461973,0.1703125,17.1268253326416,-0.06349703930318355,480,0.0003,-6.619525671005249,0.2,35.44088287353516,,
50,3200,85.10344827586206,-285.9892193448275,0,30617,5.960464477539063e-08,0.031113055,0.1625,14.894349098205566,-0.0600965260528028,490,0.0003,-6.553417348861695,0.2,30.889623641967773,,
51,3264,85.10344827586206,-285.9892193448275,0,30721,-1.1920928955078125e-07,0.033674337,0.1875,12.80873966217041,-0.0733257457613945,500,0.0003,-6.521224403381348,0.2,26.694685173034667,,
52,3328,85.10344827586206,-285.9892193448275,0,30827,0.0,0.033911195,0.1875,14.048120498657227,-0.06271658614277839,510,0.0003,-6.449989652633667,0.2,29.127000617980958,,
53,3392,85.10344827586206,-285.9892193448275,0,30924,0.0,0.03337422,0.203125,12.116095542907715,-0.06867782361805438,520,0.0003,-6.35713210105896,0.2,25.260320091247557,,
54,3456,91.46666666666667,-284.3912018666666,0,31018,0.0,0.038641103,0.2234375,11.385379791259766,-0.07520371414721012,530,0.0003,-6.300806999206543,0.2,23.81939067840576,,
55,3520,97.09677419354838,-283.4681317741935,0,31109,0.0,0.06282024,0.3390625,17.233169555664062,-0.07632681764662266,540,0.0003,-6.152373600006103,0.2,34.95697212219238,,
,3584,,,,,-1.1920928955078125e-07,0.018131059,0.0921875,12.614954948425293,-0.04068926740437746,550,0.0003,-6.038629722595215,0.2,25.963951873779298,-183.40706,400.0
56,3584,97.09677419354838,-283.4681317741935,0,34031,,,,,,,,,,,,
57,3648,97.09677419354838,-283.4681317741935,0,34122,0.0,0.027804287,0.1296875,10.095118522644043,-0.05949217453598976,560,0.0003,-6.034123516082763,0.2,21.150206565856934,,
58,3712,103.53125,-282.6728814375,0,34216,0.0,0.030136464,0.1515625,9.3436279296875,-0.05804337859153748,570,0.0003,-6.061284399032592,0.2,19.59343738555908,,
59,3776,103.53125,-282.6728814375,0,34310,0.0,0.063181914,0.328125,16.82830047607422,-0.07225181944668294,580,0.0003,-6.046624851226807,0.2,34.008976745605466,,
60,3840,103.53125,-282.6728814375,0,34402,0.0,0.017944407,0.1,9.790243148803711,-0.03122078850865364,590,0.0003,-6.02831335067749,0.2,20.437459182739257,,
61,3904,103.53125,-282.6728814375,0,34494,0.0,0.019797083,0.0953125,7.5298051834106445,-0.04488174580037594,600,0.0003,-5.948784112930298,0.2,15.891407012939453,,
62,3968,103.53125,-282.6728814375,0,34595,0.0,0.02780189,0.15,8.06164264678955,-0.0588639210909605,610,0.0003,-5.9719462394714355,0.2,17.00233898162842,,
63,4032,110.51515151515152,-281.3065534848485,0,34689,0.0,0.021823034,0.115625,8.682883262634277,-0.049968904629349706,620,0.0003,-5.904045534133911,0.2,18.23204174041748,,
,4096,,,,,0.0,0.026968703,0.1140625,14.277002334594727,-0.049054985865950584,630,0.0003,-5.842593050003051,0.2,29.216875648498537,-182.79319019999997,400.0
64,4096,110.51515151515152,-281.3065534848485,0,37613,,,,,,,,,,,,
65,4160,110.51515151515152,-281.3065534848485,0,37709,1.1920928955078125e-07,0.023035713,0.11875,8.843849182128906,-0.060087927244603635,640,0.0003,-5.830121088027954,0.2,18.61305980682373,,
66,4224,110.51515151515152,-281.3065534848485,0,37802,0.0,0.024265915,0.1265625,9.668349266052246,-0.055961475148797034,650,0.0003,-5.783435964584351,0.2,20.251771926879883,,
67,4288,110.51515151515152,-281.3065534848485,0,37892,1.1920928955078125e-07,0.023206677,0.10625,9.875944137573242,-0.056911615934222934,660,0.0003,-5.766378259658813,0.2,20.69276866912842,,
68,4352,110.51515151515152,-281.3065534848485,0,37983,0.0,0.047301535,0.24375,8.524406433105469,-0.06424493603408336,670,0.0003,-5.667219877243042,0.2,17.945811080932618,,
69,4416,110.51515151515152,-281.3065534848485,0,38074,0.0,0.035239927,0.21875,6.912813186645508,-0.07358060553669929,680,0.0003,-5.549051332473755,0.2,14.70722827911377,,
70,4480,110.51515151515152,-281.3065534848485,0,38166,5.960464477539063e-08,0.02203437,0.115625,7.92331600189209,-0.0418980436399579,690,0.0003,-5.505994129180908,0.2,16.683637046813963,,
71,4544,110.51515151515152,-281.3065534848485,0,38265,-1.1920928955078125e-07,0.026603311,0.11875,8.032673835754395,-0.05185915138572454,700,0.0003,-5.525634336471557,0.2,16.921680450439453,,
,4608,,,,,0.0,0.022957962,0.10625,9.193990707397461,-0.04341089520603418,710,0.0003,-5.436012125015258,0.2,19.260218811035156,-183.38051160000003,400.0
72,4608,110.51515151515152,-281.3065534848485,0,41197,,,,,,,,,,,,
73,4672,110.51515151515152,-281.3065534848485,0,41288,0.0,0.019447748,0.0984375,7.97336483001709,-0.038703645952045916,720,0.0003,-5.391503620147705,0.2,16.732965660095214,,
74,4736,110.51515151515152,-281.3065534848485,0,41380,-1.1920928955078125e-07,0.018282127,0.0875,7.585076808929443,-0.04347058702260256,730,0.0003,-5.282029724121093,0.2,15.978719997406007,,
75,4800,110.51515151515152,-281.3065534848485,0,41475,0.0,0.020621981,0.0984375,6.8993730545043945,-0.04910749644041061,740,0.0003,-5.218949842453003,0.2,14.595163345336914,,
76,4864,110.51515151515152,-281.3065534848485,0,41570,0.0,0.026766445,0.140625,7.8520989418029785,-0.05365236550569534,750,0.0003,-5.108656120300293,0.2,16.53515386581421,,
77,4928,110.51515151515152,-281.3065534848485,0,41661,5.960464477539063e-08,0.024439313,0.140625,6.923089027404785,-0.05227673761546612,760,0.0003,-4.998711585998535,0.2,14.648934268951416,,
78,4992,110.51515151515152,-281.3065534848485,0,41753,0.0,0.020683143,0.103125,6.3540449142456055,-0.04443833269178867,770,0.0003,-4.8690962314605715,0.2,13.482247638702393,,
79,5056,126.85714285714286,-277.9331477428571,0,41844,1.1920928955078125e-07,0.017178137,0.0765625,6.03033971786499,-0.04481028914451599,780,0.0003,-4.748990154266357,0.2,12.809585094451904,,
```

#### logs_PPO_0.01_baseline/log.txt

```
Logging to ./logs_PPO_0.01_baseline
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 14       |
|    ep_rew_mean     | -80.9    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 412      |
|    total_timesteps | 64       |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 14          |
|    ep_rew_mean          | -80.9       |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 2           |
|    time_elapsed         | 865         |
|    total_timesteps      | 128         |
| train/                  |             |
|    approx_kl            | 0.012337673 |
|    clip_fraction        | 0.0266      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.4       |
|    explained_variance   | 0.00398     |
|    learning_rate        | 0.0003      |
|    loss                 | 629         |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0491     |
|    value_loss           | 1.3e+03     |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 29          |
|    ep_rew_mean          | -133        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 3           |
|    time_elapsed         | 1320        |
|    total_timesteps      | 192         |
| train/                  |             |
|    approx_kl            | 0.013255373 |
|    clip_fraction        | 0.0688      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 607         |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.0509     |
|    value_loss           | 1.23e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 29          |
|    ep_rew_mean          | -133        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 4           |
|    time_elapsed         | 1745        |
|    total_timesteps      | 256         |
| train/                  |             |
|    approx_kl            | 0.009837223 |
|    clip_fraction        | 0.0344      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 601         |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0479     |
|    value_loss           | 1.21e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 39.8        |
|    ep_rew_mean          | -197        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 5           |
|    time_elapsed         | 2196        |
|    total_timesteps      | 320         |
| train/                  |             |
|    approx_kl            | 0.008149838 |
|    clip_fraction        | 0.0375      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 674         |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0398     |
|    value_loss           | 1.36e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 48          |
|    ep_rew_mean          | -244        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 6           |
|    time_elapsed         | 2661        |
|    total_timesteps      | 384         |
| train/                  |             |
|    approx_kl            | 0.007254501 |
|    clip_fraction        | 0.0172      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 624         |
|    n_updates            | 50          |
|    policy_gradient_loss | -0.038      |
|    value_loss           | 1.26e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 41.4        |
|    ep_rew_mean          | -207        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 7           |
|    time_elapsed         | 3065        |
|    total_timesteps      | 448         |
| train/                  |             |
|    approx_kl            | 0.009552674 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 898         |
|    n_updates            | 60          |
|    policy_gradient_loss | -0.0377     |
|    value_loss           | 1.81e+03    |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 200         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 512         |
| train/                  |             |
|    approx_kl            | 0.006872235 |
|    clip_fraction        | 0.0141      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 476         |
|    n_updates            | 70          |
|    policy_gradient_loss | -0.0378     |
|    value_loss           | 960         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 48.9     |
|    ep_rew_mean     | -252     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6271     |
|    total_timesteps | 512      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 47.3        |
|    ep_rew_mean          | -241        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 9           |
|    time_elapsed         | 6672        |
|    total_timesteps      | 576         |
| train/                  |             |
|    approx_kl            | 0.013299373 |
|    clip_fraction        | 0.0578      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 696         |
|    n_updates            | 80          |
|    policy_gradient_loss | -0.0528     |
|    value_loss           | 1.4e+03     |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 47.3         |
|    ep_rew_mean          | -241         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 10           |
|    time_elapsed         | 7050         |
|    total_timesteps      | 640          |
| train/                  |              |
|    approx_kl            | 0.0121987695 |
|    clip_fraction        | 0.0625       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.2        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 369          |
|    n_updates            | 90           |
|    policy_gradient_loss | -0.0447      |
|    value_loss           | 743          |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 48.9        |
|    ep_rew_mean          | -248        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 11          |
|    time_elapsed         | 7451        |
|    total_timesteps      | 704         |
| train/                  |             |
|    approx_kl            | 0.014821068 |
|    clip_fraction        | 0.0641      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 342         |
|    n_updates            | 100         |
|    policy_gradient_loss | -0.0496     |
|    value_loss           | 689         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 48.9        |
|    ep_rew_mean          | -248        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 12          |
|    time_elapsed         | 7840        |
|    total_timesteps      | 768         |
| train/                  |             |
|    approx_kl            | 0.009226799 |
|    clip_fraction        | 0.0219      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 393         |
|    n_updates            | 110         |
|    policy_gradient_loss | -0.0377     |
|    value_loss           | 793         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 52.2        |
|    ep_rew_mean          | -257        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 13          |
|    time_elapsed         | 8170        |
|    total_timesteps      | 832         |
| train/                  |             |
|    approx_kl            | 0.009213571 |
|    clip_fraction        | 0.0219      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 468         |
|    n_updates            | 120         |
|    policy_gradient_loss | -0.0372     |
|    value_loss           | 943         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.6        |
|    ep_rew_mean          | -273        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 14          |
|    time_elapsed         | 8562        |
|    total_timesteps      | 896         |
| train/                  |             |
|    approx_kl            | 0.011879311 |
|    clip_fraction        | 0.0672      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 286         |
|    n_updates            | 130         |
|    policy_gradient_loss | -0.0434     |
|    value_loss           | 577         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.6        |
|    ep_rew_mean          | -273        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 15          |
|    time_elapsed         | 8960        |
|    total_timesteps      | 960         |
| train/                  |             |
|    approx_kl            | 0.011848001 |
|    clip_fraction        | 0.0563      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 312         |
|    n_updates            | 140         |
|    policy_gradient_loss | -0.0474     |
|    value_loss           | 627         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 200         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 1024        |
| train/                  |             |
|    approx_kl            | 0.006678966 |
|    clip_fraction        | 0.0234      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 507         |
|    n_updates            | 150         |
|    policy_gradient_loss | -0.0358     |
|    value_loss           | 1.02e+03    |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 57.6     |
|    ep_rew_mean     | -273     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 12043    |
|    total_timesteps | 1024     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 60.5        |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 17          |
|    time_elapsed         | 12422       |
|    total_timesteps      | 1088        |
| train/                  |             |
|    approx_kl            | 0.010574155 |
|    clip_fraction        | 0.0516      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 361         |
|    n_updates            | 160         |
|    policy_gradient_loss | -0.0411     |
|    value_loss           | 726         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 60.5        |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 18          |
|    time_elapsed         | 12768       |
|    total_timesteps      | 1152        |
| train/                  |             |
|    approx_kl            | 0.009285669 |
|    clip_fraction        | 0.025       |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 526         |
|    n_updates            | 170         |
|    policy_gradient_loss | -0.0343     |
|    value_loss           | 1.06e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 62.8         |
|    ep_rew_mean          | -286         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 19           |
|    time_elapsed         | 13047        |
|    total_timesteps      | 1216         |
| train/                  |              |
|    approx_kl            | 0.0051592477 |
|    clip_fraction        | 0.0141       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.1        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 396          |
|    n_updates            | 180          |
|    policy_gradient_loss | -0.0252      |
|    value_loss           | 797          |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 64.5        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 20          |
|    time_elapsed         | 13387       |
|    total_timesteps      | 1280        |
| train/                  |             |
|    approx_kl            | 0.019144475 |
|    clip_fraction        | 0.106       |
|    clip_range           | 0.2         |
|    entropy_loss         | -10         |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 272         |
|    n_updates            | 190         |
|    policy_gradient_loss | -0.05       |
|    value_loss           | 548         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 67.1        |
|    ep_rew_mean          | -298        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 21          |
|    time_elapsed         | 13681       |
|    total_timesteps      | 1344        |
| train/                  |             |
|    approx_kl            | 0.010530417 |
|    clip_fraction        | 0.0437      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.96       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 286         |
|    n_updates            | 200         |
|    policy_gradient_loss | -0.0364     |
|    value_loss           | 577         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 67.1        |
|    ep_rew_mean          | -298        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 22          |
|    time_elapsed         | 13983       |
|    total_timesteps      | 1408        |
| train/                  |             |
|    approx_kl            | 0.017948698 |
|    clip_fraction        | 0.0922      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.91       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 201         |
|    n_updates            | 210         |
|    policy_gradient_loss | -0.0539     |
|    value_loss           | 406         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 67.1        |
|    ep_rew_mean          | -298        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 23          |
|    time_elapsed         | 14274       |
|    total_timesteps      | 1472        |
| train/                  |             |
|    approx_kl            | 0.021898726 |
|    clip_fraction        | 0.117       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.79       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 290         |
|    n_updates            | 220         |
|    policy_gradient_loss | -0.0581     |
|    value_loss           | 584         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 1536        |
| train/                  |             |
|    approx_kl            | 0.015252838 |
|    clip_fraction        | 0.0734      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.71       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 208         |
|    n_updates            | 230         |
|    policy_gradient_loss | -0.0464     |
|    value_loss           | 420         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 67.2     |
|    ep_rew_mean     | -293     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 17364    |
|    total_timesteps | 1536     |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 68.2       |
|    ep_rew_mean          | -290       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 25         |
|    time_elapsed         | 17602      |
|    total_timesteps      | 1600       |
| train/                  |            |
|    approx_kl            | 0.03271477 |
|    clip_fraction        | 0.172      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.56      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 131        |
|    n_updates            | 240        |
|    policy_gradient_loss | -0.0665    |
|    value_loss           | 265        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 71.5        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 26          |
|    time_elapsed         | 17833       |
|    total_timesteps      | 1664        |
| train/                  |             |
|    approx_kl            | 0.018527184 |
|    clip_fraction        | 0.103       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.44       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 87.2        |
|    n_updates            | 250         |
|    policy_gradient_loss | -0.0599     |
|    value_loss           | 177         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 72.3       |
|    ep_rew_mean          | -290       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 27         |
|    time_elapsed         | 18072      |
|    total_timesteps      | 1728       |
| train/                  |            |
|    approx_kl            | 0.02827879 |
|    clip_fraction        | 0.163      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.43      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 93.7       |
|    n_updates            | 260        |
|    policy_gradient_loss | -0.0607    |
|    value_loss           | 190        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 72.3        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 28          |
|    time_elapsed         | 18299       |
|    total_timesteps      | 1792        |
| train/                  |             |
|    approx_kl            | 0.021897698 |
|    clip_fraction        | 0.102       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.3        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 141         |
|    n_updates            | 270         |
|    policy_gradient_loss | -0.0551     |
|    value_loss           | 286         |
-----------------------------------------
---------------------------------------
| rollout/                |           |
|    ep_len_mean          | 72.3      |
|    ep_rew_mean          | -290      |
| time/                   |           |
|    fps                  | 0         |
|    iterations           | 29        |
|    time_elapsed         | 18521     |
|    total_timesteps      | 1856      |
| train/                  |           |
|    approx_kl            | 0.0308385 |
|    clip_fraction        | 0.177     |
|    clip_range           | 0.2       |
|    entropy_loss         | -9.23     |
|    explained_variance   | -1.19e-07 |
|    learning_rate        | 0.0003    |
|    loss                 | 96.7      |
|    n_updates            | 280       |
|    policy_gradient_loss | -0.0677   |
|    value_loss           | 196       |
---------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 72.3        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 30          |
|    time_elapsed         | 18783       |
|    total_timesteps      | 1920        |
| train/                  |             |
|    approx_kl            | 0.016395954 |
|    clip_fraction        | 0.0813      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.17       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 83.7        |
|    n_updates            | 290         |
|    policy_gradient_loss | -0.0504     |
|    value_loss           | 170         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 72.3       |
|    ep_rew_mean          | -290       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 31         |
|    time_elapsed         | 18974      |
|    total_timesteps      | 1984       |
| train/                  |            |
|    approx_kl            | 0.01536713 |
|    clip_fraction        | 0.0719     |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.1       |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 197        |
|    n_updates            | 300        |
|    policy_gradient_loss | -0.0449    |
|    value_loss           | 397        |
----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 2048        |
| train/                  |             |
|    approx_kl            | 0.025281565 |
|    clip_fraction        | 0.144       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.99       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 88.5        |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.0644     |
|    value_loss           | 179         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 72.3     |
|    ep_rew_mean     | -290     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 22040    |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 75.9        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 33          |
|    time_elapsed         | 22247       |
|    total_timesteps      | 2112        |
| train/                  |             |
|    approx_kl            | 0.020967763 |
|    clip_fraction        | 0.0953      |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.85       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 116         |
|    n_updates            | 320         |
|    policy_gradient_loss | -0.0618     |
|    value_loss           | 234         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 75.9        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 34          |
|    time_elapsed         | 22444       |
|    total_timesteps      | 2176        |
| train/                  |             |
|    approx_kl            | 0.039487112 |
|    clip_fraction        | 0.25        |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.73       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 35.5        |
|    n_updates            | 330         |
|    policy_gradient_loss | -0.0728     |
|    value_loss           | 72.5        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 79.2       |
|    ep_rew_mean          | -291       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 35         |
|    time_elapsed         | 22664      |
|    total_timesteps      | 2240       |
| train/                  |            |
|    approx_kl            | 0.03676312 |
|    clip_fraction        | 0.23       |
|    clip_range           | 0.2        |
|    entropy_loss         | -8.62      |
|    explained_variance   | -1.19e-07  |
|    learning_rate        | 0.0003     |
|    loss                 | 51.8       |
|    n_updates            | 340        |
|    policy_gradient_loss | -0.056     |
|    value_loss           | 106        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.3        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 36          |
|    time_elapsed         | 22852       |
|    total_timesteps      | 2304        |
| train/                  |             |
|    approx_kl            | 0.028057288 |
|    clip_fraction        | 0.148       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.44       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 91.2        |
|    n_updates            | 350         |
|    policy_gradient_loss | -0.0615     |
|    value_loss           | 185         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 85.4       |
|    ep_rew_mean          | -291       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 37         |
|    time_elapsed         | 23022      |
|    total_timesteps      | 2368       |
| train/                  |            |
|    approx_kl            | 0.02848405 |
|    clip_fraction        | 0.144      |
|    clip_range           | 0.2        |
|    entropy_loss         | -8.31      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 52.7       |
|    n_updates            | 360        |
|    policy_gradient_loss | -0.0665    |
|    value_loss           | 107        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 38          |
|    time_elapsed         | 23205       |
|    total_timesteps      | 2432        |
| train/                  |             |
|    approx_kl            | 0.039110675 |
|    clip_fraction        | 0.234       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.39       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 24.5        |
|    n_updates            | 370         |
|    policy_gradient_loss | -0.0738     |
|    value_loss           | 50.4        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 39          |
|    time_elapsed         | 23381       |
|    total_timesteps      | 2496        |
| train/                  |             |
|    approx_kl            | 0.024425887 |
|    clip_fraction        | 0.128       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.44       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 52.6        |
|    n_updates            | 380         |
|    policy_gradient_loss | -0.0518     |
|    value_loss           | 107         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 2560        |
| train/                  |             |
|    approx_kl            | 0.013451718 |
|    clip_fraction        | 0.0547      |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.42       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 73.9        |
|    n_updates            | 390         |
|    policy_gradient_loss | -0.0496     |
|    value_loss           | 150         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 85.4     |
|    ep_rew_mean     | -291     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 26355    |
|    total_timesteps | 2560     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 41          |
|    time_elapsed         | 26515       |
|    total_timesteps      | 2624        |
| train/                  |             |
|    approx_kl            | 0.025055602 |
|    clip_fraction        | 0.141       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.51       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 55.7        |
|    n_updates            | 400         |
|    policy_gradient_loss | -0.0634     |
|    value_loss           | 113         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 42          |
|    time_elapsed         | 26672       |
|    total_timesteps      | 2688        |
| train/                  |             |
|    approx_kl            | 0.021403098 |
|    clip_fraction        | 0.108       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.39       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 53          |
|    n_updates            | 410         |
|    policy_gradient_loss | -0.0646     |
|    value_loss           | 108         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 43          |
|    time_elapsed         | 26806       |
|    total_timesteps      | 2752        |
| train/                  |             |
|    approx_kl            | 0.016373124 |
|    clip_fraction        | 0.0703      |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.25       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 40.5        |
|    n_updates            | 420         |
|    policy_gradient_loss | -0.0529     |
|    value_loss           | 82.9        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 44          |
|    time_elapsed         | 26959       |
|    total_timesteps      | 2816        |
| train/                  |             |
|    approx_kl            | 0.033447925 |
|    clip_fraction        | 0.178       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.17       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 31.5        |
|    n_updates            | 430         |
|    policy_gradient_loss | -0.0751     |
|    value_loss           | 64.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 45          |
|    time_elapsed         | 27119       |
|    total_timesteps      | 2880        |
| train/                  |             |
|    approx_kl            | 0.019272793 |
|    clip_fraction        | 0.102       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.1        |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 39.7        |
|    n_updates            | 440         |
|    policy_gradient_loss | -0.0517     |
|    value_loss           | 81.1        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 46          |
|    time_elapsed         | 27248       |
|    total_timesteps      | 2944        |
| train/                  |             |
|    approx_kl            | 0.024974179 |
|    clip_fraction        | 0.133       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.98       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 38.6        |
|    n_updates            | 450         |
|    policy_gradient_loss | -0.0611     |
|    value_loss           | 79          |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 85.4        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 47          |
|    time_elapsed         | 27392       |
|    total_timesteps      | 3008        |
| train/                  |             |
|    approx_kl            | 0.040451653 |
|    clip_fraction        | 0.203       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.79       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 21.4        |
|    n_updates            | 460         |
|    policy_gradient_loss | -0.071      |
|    value_loss           | 44.2        |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 3072        |
| train/                  |             |
|    approx_kl            | 0.027317137 |
|    clip_fraction        | 0.156       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.6        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 30.7        |
|    n_updates            | 470         |
|    policy_gradient_loss | -0.0661     |
|    value_loss           | 62.9        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 91.1     |
|    ep_rew_mean     | -291     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 30330    |
|    total_timesteps | 3072     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 99.1        |
|    ep_rew_mean          | -289        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 49          |
|    time_elapsed         | 30467       |
|    total_timesteps      | 3136        |
| train/                  |             |
|    approx_kl            | 0.031152105 |
|    clip_fraction        | 0.164       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.57       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 34.2        |
|    n_updates            | 480         |
|    policy_gradient_loss | -0.0766     |
|    value_loss           | 69.6        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 50         |
|    time_elapsed         | 30597      |
|    total_timesteps      | 3200       |
| train/                  |            |
|    approx_kl            | 0.04109685 |
|    clip_fraction        | 0.225      |
|    clip_range           | 0.2        |
|    entropy_loss         | -7.47      |
|    explained_variance   | -1.19e-07  |
|    learning_rate        | 0.0003     |
|    loss                 | 34.2       |
|    n_updates            | 490        |
|    policy_gradient_loss | -0.0738    |
|    value_loss           | 69.3       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 51          |
|    time_elapsed         | 30705       |
|    total_timesteps      | 3264        |
| train/                  |             |
|    approx_kl            | 0.041215673 |
|    clip_fraction        | 0.217       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.39       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 22.7        |
|    n_updates            | 500         |
|    policy_gradient_loss | -0.0702     |
|    value_loss           | 46.5        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 52         |
|    time_elapsed         | 30816      |
|    total_timesteps      | 3328       |
| train/                  |            |
|    approx_kl            | 0.02742074 |
|    clip_fraction        | 0.136      |
|    clip_range           | 0.2        |
|    entropy_loss         | -7.33      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 14.8       |
|    n_updates            | 510        |
|    policy_gradient_loss | -0.0623    |
|    value_loss           | 30.9       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 53          |
|    time_elapsed         | 30935       |
|    total_timesteps      | 3392        |
| train/                  |             |
|    approx_kl            | 0.022933966 |
|    clip_fraction        | 0.109       |
|    clip_range           | 0.2         |
|    entropy_loss         | -7.24       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 18.3        |
|    n_updates            | 520         |
|    policy_gradient_loss | -0.0588     |
|    value_loss           | 38          |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 54         |
|    time_elapsed         | 31045      |
|    total_timesteps      | 3456       |
| train/                  |            |
|    approx_kl            | 0.03433932 |
|    clip_fraction        | 0.197      |
|    clip_range           | 0.2        |
|    entropy_loss         | -7.09      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 14.9       |
|    n_updates            | 530        |
|    policy_gradient_loss | -0.0685    |
|    value_loss           | 31.1       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 55          |
|    time_elapsed         | 31151       |
|    total_timesteps      | 3520        |
| train/                  |             |
|    approx_kl            | 0.029980943 |
|    clip_fraction        | 0.18        |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.93       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 16.3        |
|    n_updates            | 540         |
|    policy_gradient_loss | -0.0636     |
|    value_loss           | 33.9        |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -182        |
| time/                   |             |
|    total_timesteps      | 3584        |
| train/                  |             |
|    approx_kl            | 0.025252137 |
|    clip_fraction        | 0.134       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.79       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 12.8        |
|    n_updates            | 550         |
|    policy_gradient_loss | -0.0673     |
|    value_loss           | 26.9        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 104      |
|    ep_rew_mean     | -288     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 34093    |
|    total_timesteps | 3584     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 57          |
|    time_elapsed         | 34209       |
|    total_timesteps      | 3648        |
| train/                  |             |
|    approx_kl            | 0.033944964 |
|    clip_fraction        | 0.156       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.65       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 17.1        |
|    n_updates            | 560         |
|    policy_gradient_loss | -0.0667     |
|    value_loss           | 35.5        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 58          |
|    time_elapsed         | 34329       |
|    total_timesteps      | 3712        |
| train/                  |             |
|    approx_kl            | 0.023318913 |
|    clip_fraction        | 0.138       |
|    clip_range           | 0.2         |
|    entropy_loss         | -6.45       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 15.1        |
|    n_updates            | 570         |
|    policy_gradient_loss | -0.0504     |
|    value_loss           | 31.3        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 59         |
|    time_elapsed         | 34456      |
|    total_timesteps      | 3776       |
| train/                  |            |
|    approx_kl            | 0.02731265 |
|    clip_fraction        | 0.141      |
|    clip_range           | 0.2        |
|    entropy_loss         | -6.29      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 12.2       |
|    n_updates            | 580        |
|    policy_gradient_loss | -0.0629    |
|    value_loss           | 25.5       |
----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 60         |
|    time_elapsed         | 34557      |
|    total_timesteps      | 3840       |
| train/                  |            |
|    approx_kl            | 0.02043181 |
|    clip_fraction        | 0.0875     |
|    clip_range           | 0.2        |
|    entropy_loss         | -6.21      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 13.9       |
|    n_updates            | 590        |
|    policy_gradient_loss | -0.0476    |
|    value_loss           | 29         |
----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 61         |
|    time_elapsed         | 34661      |
|    total_timesteps      | 3904       |
| train/                  |            |
|    approx_kl            | 0.03502047 |
|    clip_fraction        | 0.197      |
|    clip_range           | 0.2        |
|    entropy_loss         | -6.1       |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 9.46       |
|    n_updates            | 600        |
|    policy_gradient_loss | -0.081     |
|    value_loss           | 20         |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 62          |
|    time_elapsed         | 34753       |
|    total_timesteps      | 3968        |
| train/                  |             |
|    approx_kl            | 0.031027105 |
|    clip_fraction        | 0.167       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.93       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 8.7         |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.0639     |
|    value_loss           | 18.4        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 104        |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 63         |
|    time_elapsed         | 34862      |
|    total_timesteps      | 4032       |
| train/                  |            |
|    approx_kl            | 0.03215894 |
|    clip_fraction        | 0.166      |
|    clip_range           | 0.2        |
|    entropy_loss         | -5.78      |
|    explained_variance   | 5.96e-08   |
|    learning_rate        | 0.0003     |
|    loss                 | 6.24       |
|    n_updates            | 620        |
|    policy_gradient_loss | -0.0618    |
|    value_loss           | 13.4       |
----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.032552026 |
|    clip_fraction        | 0.198       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.81       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 9.08        |
|    n_updates            | 630         |
|    policy_gradient_loss | -0.0727     |
|    value_loss           | 19.2        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 104      |
|    ep_rew_mean     | -288     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 37779    |
|    total_timesteps | 4096     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 65          |
|    time_elapsed         | 37884       |
|    total_timesteps      | 4160        |
| train/                  |             |
|    approx_kl            | 0.036718592 |
|    clip_fraction        | 0.202       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.84       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 9.33        |
|    n_updates            | 640         |
|    policy_gradient_loss | -0.0663     |
|    value_loss           | 19.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 66          |
|    time_elapsed         | 37989       |
|    total_timesteps      | 4224        |
| train/                  |             |
|    approx_kl            | 0.026313182 |
|    clip_fraction        | 0.147       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.72       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 7.26        |
|    n_updates            | 650         |
|    policy_gradient_loss | -0.0559     |
|    value_loss           | 15.5        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 67          |
|    time_elapsed         | 38087       |
|    total_timesteps      | 4288        |
| train/                  |             |
|    approx_kl            | 0.024443105 |
|    clip_fraction        | 0.119       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.66       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 12.6        |
|    n_updates            | 660         |
|    policy_gradient_loss | -0.0519     |
|    value_loss           | 26.2        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 104         |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 68          |
|    time_elapsed         | 38198       |
|    total_timesteps      | 4352        |
| train/                  |             |
|    approx_kl            | 0.027610064 |
|    clip_fraction        | 0.142       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.63       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 8.99        |
|    n_updates            | 670         |
|    policy_gradient_loss | -0.0698     |
|    value_loss           | 19.1        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 123         |
|    ep_rew_mean          | -282        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 69          |
|    time_elapsed         | 38291       |
|    total_timesteps      | 4416        |
| train/                  |             |
|    approx_kl            | 0.026144825 |
|    clip_fraction        | 0.145       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.65       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 10.3        |
|    n_updates            | 680         |
|    policy_gradient_loss | -0.0572     |
|    value_loss           | 21.6        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 123        |
|    ep_rew_mean          | -282       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 70         |
|    time_elapsed         | 38385      |
|    total_timesteps      | 4480       |
| train/                  |            |
|    approx_kl            | 0.10080221 |
|    clip_fraction        | 0.491      |
|    clip_range           | 0.2        |
|    entropy_loss         | -5.55      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 28.2       |
|    n_updates            | 690        |
|    policy_gradient_loss | -0.0988    |
|    value_loss           | 56.8       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 71          |
|    time_elapsed         | 38487       |
|    total_timesteps      | 4544        |
| train/                  |             |
|    approx_kl            | 0.020173997 |
|    clip_fraction        | 0.0969      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.45       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 7.25        |
|    n_updates            | 700         |
|    policy_gradient_loss | -0.0465     |
|    value_loss           | 15          |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -183        |
| time/                   |             |
|    total_timesteps      | 4608        |
| train/                  |             |
|    approx_kl            | 0.014706491 |
|    clip_fraction        | 0.0688      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.41       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 28.8        |
|    n_updates            | 710         |
|    policy_gradient_loss | -0.0367     |
|    value_loss           | 58.4        |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 129      |
|    ep_rew_mean     | -280     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 41362    |
|    total_timesteps | 4608     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 73          |
|    time_elapsed         | 41455       |
|    total_timesteps      | 4672        |
| train/                  |             |
|    approx_kl            | 0.016405586 |
|    clip_fraction        | 0.1         |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.39       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 6.81        |
|    n_updates            | 720         |
|    policy_gradient_loss | -0.04       |
|    value_loss           | 14.4        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 74          |
|    time_elapsed         | 41546       |
|    total_timesteps      | 4736        |
| train/                  |             |
|    approx_kl            | 0.017981078 |
|    clip_fraction        | 0.0984      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.43       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 6.44        |
|    n_updates            | 730         |
|    policy_gradient_loss | -0.0433     |
|    value_loss           | 13.7        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 75          |
|    time_elapsed         | 41637       |
|    total_timesteps      | 4800        |
| train/                  |             |
|    approx_kl            | 0.017677587 |
|    clip_fraction        | 0.0891      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.46       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 6.11        |
|    n_updates            | 740         |
|    policy_gradient_loss | -0.0486     |
|    value_loss           | 13.1        |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 129        |
|    ep_rew_mean          | -280       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 76         |
|    time_elapsed         | 41729      |
|    total_timesteps      | 4864       |
| train/                  |            |
|    approx_kl            | 0.01820511 |
|    clip_fraction        | 0.0859     |
|    clip_range           | 0.2        |
|    entropy_loss         | -5.41      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 6.4        |
|    n_updates            | 750        |
|    policy_gradient_loss | -0.0422    |
|    value_loss           | 13.6       |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 77          |
|    time_elapsed         | 41824       |
|    total_timesteps      | 4928        |
| train/                  |             |
|    approx_kl            | 0.025982507 |
|    clip_fraction        | 0.152       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.36       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 5.53        |
|    n_updates            | 760         |
|    policy_gradient_loss | -0.0582     |
|    value_loss           | 11.9        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 78          |
|    time_elapsed         | 41914       |
|    total_timesteps      | 4992        |
| train/                  |             |
|    approx_kl            | 0.021784266 |
|    clip_fraction        | 0.116       |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.34       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 8.89        |
|    n_updates            | 770         |
|    policy_gradient_loss | -0.0476     |
|    value_loss           | 18.8        |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 129         |
|    ep_rew_mean          | -280        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 79          |
|    time_elapsed         | 42010       |
|    total_timesteps      | 5056        |
| train/                  |             |
|    approx_kl            | 0.016031437 |
|    clip_fraction        | 0.0922      |
|    clip_range           | 0.2         |
|    entropy_loss         | -5.37       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 5.72        |
|    n_updates            | 780         |
|    policy_gradient_loss | -0.0415     |
|    value_loss           | 12.3        |
-----------------------------------------
```

#### logs_PPO_0.01_baseline/progress.csv

```
time/iterations,time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,time/fps,time/time_elapsed,train/explained_variance,train/approx_kl,train/clip_fraction,train/loss,train/policy_gradient_loss,train/n_updates,train/learning_rate,train/entropy_loss,train/clip_range,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,64,14.0,-80.931749,0,412,,,,,,,,,,,,
2,128,14.0,-80.931749,0,865,0.003979742527008057,0.012337673,0.0265625,628.91748046875,-0.04907322861254215,10,0.0003,-10.355007457733155,0.2,1297.018505859375,,
3,192,29.0,-132.80558933333333,0,1320,0.0,0.013255373,0.06875,606.8843994140625,-0.05092278588563204,20,0.0003,-10.344224643707275,0.2,1227.8317504882812,,
4,256,29.0,-132.80558933333333,0,1745,0.0,0.009837223,0.034375,601.4386596679688,-0.04792186804115772,30,0.0003,-10.328796672821046,0.2,1214.4789916992188,,
5,320,39.75,-197.20570225,0,2196,0.0,0.008149838,0.0375,673.8716430664062,-0.03976563923060894,40,0.0003,-10.315431594848633,0.2,1359.19287109375,,
6,384,48.0,-244.1633508,0,2661,5.960464477539063e-08,0.007254501,0.0171875,623.639404296875,-0.03804940581321716,50,0.0003,-10.307303333282471,0.2,1256.9391967773438,,
7,448,41.42857142857143,-207.1337487142857,0,3065,5.960464477539063e-08,0.009552674,0.03125,898.4871215820312,-0.037687619030475614,60,0.0003,-10.288030815124511,0.2,1808.1204956054687,,
,512,,,,,0.0,0.006872235,0.0140625,476.2591247558594,-0.037837601453065875,70,0.0003,-10.273942279815675,0.2,959.7382507324219,-183.02501539999997,200.0
8,512,48.888888888888886,-251.66440466666668,0,6271,,,,,,,,,,,,
9,576,47.3,-241.42072930000003,0,6672,0.0,0.013299373,0.0578125,696.4805908203125,-0.05275816712528467,80,0.0003,-10.25428285598755,0.2,1400.8256469726562,,
10,640,47.3,-241.42072930000003,0,7050,0.0,0.0121987695,0.0625,368.6083984375,-0.04465885907411575,90,0.0003,-10.21614933013916,0.2,743.0037475585938,,
11,704,48.90909090909091,-247.5471283636364,0,7451,0.0,0.014821068,0.0640625,341.5318908691406,-0.04960230588912964,100,0.0003,-10.186589622497559,0.2,688.6086120605469,,
12,768,48.90909090909091,-247.5471283636364,0,7840,-1.1920928955078125e-07,0.009226799,0.021875,393.4744567871094,-0.037748335674405095,110,0.0003,-10.145546817779541,0.2,792.5029479980469,,
13,832,52.166666666666664,-257.3996548333334,0,8170,-1.1920928955078125e-07,0.009213571,0.021875,468.1217041015625,-0.037196427956223486,120,0.0003,-10.148195838928222,0.2,942.6124206542969,,
14,896,57.57142857142857,-272.96893121428576,0,8562,1.1920928955078125e-07,0.011879311,0.0671875,286.1368408203125,-0.04338397495448589,130,0.0003,-10.160109615325927,0.2,576.9631103515625,,
15,960,57.57142857142857,-272.96893121428576,0,8960,-1.1920928955078125e-07,0.011848001,0.05625,311.50079345703125,-0.04742661034688354,140,0.0003,-10.148009014129638,0.2,627.4384155273438,,
,1024,,,,,5.960464477539063e-08,0.006678966,0.0234375,507.4692077636719,-0.035843107104301455,150,0.0003,-10.118657875061036,0.2,1021.0230041503906,-183.6530272,200.0
16,1024,57.57142857142857,-272.96893121428576,0,12043,,,,,,,,,,,,
17,1088,60.46666666666667,-279.5452128666667,0,12422,1.7881393432617188e-07,0.010574155,0.0515625,360.577880859375,-0.041087645664811136,160,0.0003,-10.110985279083252,0.2,726.2289184570312,,
18,1152,60.46666666666667,-279.5452128666667,0,12768,0.0,0.009285669,0.025,526.0516967773438,-0.03431394025683403,170,0.0003,-10.092340183258056,0.2,1057.87724609375,,
19,1216,62.75,-286.25055406250004,0,13047,0.0,0.0051592477,0.0140625,395.96099853515625,-0.025230436585843563,180,0.0003,-10.064274501800536,0.2,797.1757751464844,,
20,1280,64.47058823529412,-292.6096973529412,0,13387,0.0,0.019144475,0.10625,271.8649597167969,-0.04997660778462887,190,0.0003,-10.02440586090088,0.2,547.6896545410157,,
21,1344,67.11111111111111,-297.6368210555556,0,13681,-1.1920928955078125e-07,0.010530417,0.04375,286.4881286621094,-0.03636628314852715,200,0.0003,-9.964962577819824,0.2,577.2951904296875,,
22,1408,67.11111111111111,-297.6368210555556,0,13983,-1.1920928955078125e-07,0.017948698,0.0921875,201.07151794433594,-0.05390248708426952,210,0.0003,-9.912430763244629,0.2,405.6424987792969,,
23,1472,67.11111111111111,-297.6368210555556,0,14274,-1.1920928955078125e-07,0.021898726,0.1171875,290.048095703125,-0.058060924708843234,220,0.0003,-9.794366931915283,0.2,584.4798217773438,,
,1536,,,,,-1.1920928955078125e-07,0.015252838,0.0734375,208.25350952148438,-0.04643708672374487,230,0.0003,-9.712809658050537,0.2,420.15394592285156,-183.4251402,400.0
24,1536,67.21052631578948,-292.92266289473685,0,17364,,,,,,,,,,,,
25,1600,68.25,-290.4508911,0,17602,0.0,0.03271477,0.171875,130.954833984375,-0.06648496575653554,240,0.0003,-9.56152276992798,0.2,264.712353515625,,
26,1664,71.52380952380952,-292.8413868095238,0,17833,0.0,0.018527184,0.103125,87.24485778808594,-0.05991819240152836,250,0.0003,-9.43775577545166,0.2,176.87929992675782,,
27,1728,72.27272727272727,-290.1854348636364,0,18072,0.0,0.02827879,0.1625,93.69670867919922,-0.0606707664206624,260,0.0003,-9.433082866668702,0.2,189.81277770996093,,
28,1792,72.27272727272727,-290.1854348636364,0,18299,0.0,0.021897698,0.1015625,141.3666229248047,-0.05514770969748497,270,0.0003,-9.29935417175293,0.2,285.50782165527346,,
29,1856,72.27272727272727,-290.1854348636364,0,18521,-1.1920928955078125e-07,0.0308385,0.1765625,96.74530029296875,-0.06773603148758411,280,0.0003,-9.231454944610595,0.2,196.17371063232423,,
30,1920,72.27272727272727,-290.1854348636364,0,18783,1.1920928955078125e-07,0.016395954,0.08125,83.72001647949219,-0.050429463014006616,290,0.0003,-9.169335842132568,0.2,169.99528961181642,,
31,1984,72.27272727272727,-290.1854348636364,0,18974,0.0,0.01536713,0.071875,196.7268829345703,-0.04491160828620196,300,0.0003,-9.102083873748779,0.2,397.0996887207031,,
,2048,,,,,0.0,0.025281565,0.14375,88.45252227783203,-0.06435097120702267,310,0.0003,-8.986308002471924,0.2,179.32073211669922,-183.6867242,400.0
32,2048,72.27272727272727,-290.1854348636364,0,22040,,,,,,,,,,,,
33,2112,75.8695652173913,-289.931664,0,22247,0.0,0.020967763,0.0953125,115.75106811523438,-0.06176312305033207,320,0.0003,-8.851637363433838,0.2,234.41734008789064,,
34,2176,75.8695652173913,-289.931664,0,22444,0.0,0.039487112,0.25,35.48598861694336,-0.0727888073772192,330,0.0003,-8.729203701019287,0.2,72.49301528930664,,
35,2240,79.16666666666667,-290.5184770833334,0,22664,-1.1920928955078125e-07,0.03676312,0.2296875,51.786312103271484,-0.05602402985095978,340,0.0003,-8.623913192749024,0.2,105.51727523803712,,
36,2304,82.28,-290.392596,0,22852,0.0,0.028057288,0.1484375,91.23846435546875,-0.06148949712514877,350,0.0003,-8.444341087341309,0.2,184.75949249267578,,
37,2368,85.38461538461539,-291.10167688461536,0,23022,0.0,0.02848405,0.14375,52.708797454833984,-0.06650610836222767,360,0.0003,-8.307341861724854,0.2,107.3370864868164,,
38,2432,85.38461538461539,-291.10167688461536,0,23205,0.0,0.039110675,0.234375,24.540691375732422,-0.07378382310271263,370,0.0003,-8.390359878540039,0.2,50.441978454589844,,
39,2496,85.38461538461539,-291.10167688461536,0,23381,1.7881393432617188e-07,0.024425887,0.128125,52.55552291870117,-0.05177561342716217,380,0.0003,-8.444827270507812,0.2,107.10808639526367,,
,2560,,,,,-1.1920928955078125e-07,0.013451718,0.0546875,73.87794494628906,-0.04955791477113962,390,0.0003,-8.421911430358886,0.2,150.15227508544922,-184.0605504,400.0
40,2560,85.38461538461539,-291.10167688461536,0,26355,,,,,,,,,,,,
41,2624,85.38461538461539,-291.10167688461536,0,26515,0.0,0.025055602,0.140625,55.697330474853516,-0.06336319781839847,400,0.0003,-8.511184692382812,0.2,113.44521484375,,
42,2688,85.38461538461539,-291.10167688461536,0,26672,0.0,0.021403098,0.1078125,52.95729446411133,-0.06460088370367885,410,0.0003,-8.385446643829345,0.2,107.99154129028321,,
43,2752,85.38461538461539,-291.10167688461536,0,26806,0.0,0.016373124,0.0703125,40.47648239135742,-0.05288367532193661,420,0.0003,-8.250005340576172,0.2,82.8620590209961,,
44,2816,85.38461538461539,-291.10167688461536,0,26959,0.0,0.033447925,0.178125,31.484161376953125,-0.07505618464201688,430,0.0003,-8.17016668319702,0.2,64.67384033203125,,
45,2880,85.38461538461539,-291.10167688461536,0,27119,1.1920928955078125e-07,0.019272793,0.1015625,39.65752029418945,-0.05174971958622336,440,0.0003,-8.09831953048706,0.2,81.07633895874024,,
46,2944,85.38461538461539,-291.10167688461536,0,27248,0.0,0.024974179,0.1328125,38.62609100341797,-0.06109459176659584,450,0.0003,-7.975163173675537,0.2,79.0313003540039,,
47,3008,85.38461538461539,-291.10167688461536,0,27392,0.0,0.040451653,0.203125,21.375812530517578,-0.07103681936860085,460,0.0003,-7.791160583496094,0.2,44.182224655151366,,
,3072,,,,,0.0,0.027317137,0.15625,30.653657913208008,-0.06613945551216602,470,0.0003,-7.597978496551514,0.2,62.93468742370605,-183.6518112,400.0
48,3072,91.11111111111111,-290.8989007407407,0,30330,,,,,,,,,,,,
49,3136,99.13793103448276,-288.7257088620689,0,30467,0.0,0.031152105,0.1640625,34.17892837524414,-0.0766415998339653,480,0.0003,-7.569041395187378,0.2,69.55157928466797,,
50,3200,103.56666666666666,-287.6900101666667,0,30597,-1.1920928955078125e-07,0.04109685,0.225,34.188018798828125,-0.07378541305661201,490,0.0003,-7.467291450500488,0.2,69.2574333190918,,
51,3264,103.56666666666666,-287.6900101666667,0,30705,0.0,0.041215673,0.2171875,22.746294021606445,-0.0702400729060173,500,0.0003,-7.387417554855347,0.2,46.455340194702146,,
52,3328,103.56666666666666,-287.6900101666667,0,30816,0.0,0.02742074,0.1359375,14.829029083251953,-0.062307053990662095,510,0.0003,-7.326716613769531,0.2,30.929436874389648,,
53,3392,103.56666666666666,-287.6900101666667,0,30935,0.0,0.022933966,0.109375,18.300395965576172,-0.05883920285850763,520,0.0003,-7.237463188171387,0.2,37.982845306396484,,
54,3456,103.56666666666666,-287.6900101666667,0,31045,0.0,0.03433932,0.196875,14.905449867248535,-0.06851069815456867,530,0.0003,-7.087092065811158,0.2,31.057789421081544,,
55,3520,103.56666666666666,-287.6900101666667,0,31151,0.0,0.029980943,0.1796875,16.316165924072266,-0.06362794451415539,540,0.0003,-6.92903962135315,0.2,33.90174827575684,,
,3584,,,,,1.1920928955078125e-07,0.025252137,0.134375,12.839394569396973,-0.06726894993335009,550,0.0003,-6.7915716648101805,0.2,26.903839302062988,-181.95247179999998,400.0
56,3584,103.56666666666666,-287.6900101666667,0,34093,,,,,,,,,,,,
57,3648,103.56666666666666,-287.6900101666667,0,34209,-1.1920928955078125e-07,0.033944964,0.15625,17.11688804626465,-0.06673389934003353,560,0.0003,-6.6540265560150145,0.2,35.495964431762694,,
58,3712,103.56666666666666,-287.6900101666667,0,34329,0.0,0.023318913,0.1375,15.054852485656738,-0.05038740281015634,570,0.0003,-6.446998548507691,0.2,31.261070442199706,,
59,3776,103.56666666666666,-287.6900101666667,0,34456,0.0,0.02731265,0.140625,12.187644958496094,-0.06287376042455435,580,0.0003,-6.285531520843506,0.2,25.491666984558105,,
60,3840,103.56666666666666,-287.6900101666667,0,34557,0.0,0.02043181,0.0875,13.90257453918457,-0.047554020676761864,590,0.0003,-6.209599256515503,0.2,28.987249183654786,,
61,3904,103.56666666666666,-287.6900101666667,0,34661,0.0,0.03502047,0.196875,9.462966918945312,-0.08104117065668107,600,0.0003,-6.096381902694702,0.2,20.01870822906494,,
62,3968,103.56666666666666,-287.6900101666667,0,34753,0.0,0.031027105,0.1671875,8.698741912841797,-0.06389886140823364,610,0.0003,-5.929089546203613,0.2,18.430036354064942,,
63,4032,103.56666666666666,-287.6900101666667,0,34862,5.960464477539063e-08,0.03215894,0.165625,6.243506908416748,-0.0617841936647892,620,0.0003,-5.775619125366211,0.2,13.413726997375488,,
,4096,,,,,0.0,0.032552026,0.1984375,9.076820373535156,-0.07268287725746632,630,0.0003,-5.8089134216308596,0.2,19.243678855895997,-183.3952104,400.0
64,4096,103.56666666666666,-287.6900101666667,0,37779,,,,,,,,,,,,
65,4160,103.56666666666666,-287.6900101666667,0,37884,0.0,0.036718592,0.2015625,9.332679748535156,-0.06630661971867084,640,0.0003,-5.835685920715332,0.2,19.692106246948242,,
66,4224,103.56666666666666,-287.6900101666667,0,37989,0.0,0.026313182,0.146875,7.26155424118042,-0.05591359734535217,650,0.0003,-5.722985935211182,0.2,15.458447360992432,,
67,4288,103.56666666666666,-287.6900101666667,0,38087,0.0,0.024443105,0.11875,12.558932304382324,-0.051860587671399117,660,0.0003,-5.661573982238769,0.2,26.196718215942383,,
68,4352,103.56666666666666,-287.6900101666667,0,38198,0.0,0.027610064,0.1421875,8.986159324645996,-0.06983405947685242,670,0.0003,-5.627637958526611,0.2,19.061157035827637,,
69,4416,123.39393939393939,-282.1051791818182,0,38291,0.0,0.026144825,0.1453125,10.29894733428955,-0.057176553457975385,680,0.0003,-5.652285957336426,0.2,21.629256439208984,,
70,4480,123.39393939393939,-282.1051791818182,0,38385,0.0,0.10080221,0.490625,28.191333770751953,-0.0987531015649438,690,0.0003,-5.545714855194092,0.2,56.756241989135745,,
71,4544,129.44117647058823,-280.18742514705883,0,38487,0.0,0.020173997,0.096875,7.251147747039795,-0.0465473759919405,700,0.0003,-5.447128582000732,0.2,15.00733118057251,,
,4608,,,,,-1.1920928955078125e-07,0.014706491,0.06875,28.760223388671875,-0.036727023869752885,710,0.0003,-5.407768297195434,0.2,58.3737075805664,-183.46075399999998,400.0
72,4608,129.44117647058823,-280.18742514705883,0,41362,,,,,,,,,,,,
73,4672,129.44117647058823,-280.18742514705883,0,41455,0.0,0.016405586,0.1,6.809627532958984,-0.03999975509941578,720,0.0003,-5.393047285079956,0.2,14.433207511901855,,
74,4736,129.44117647058823,-280.18742514705883,0,41546,0.0,0.017981078,0.0984375,6.441025733947754,-0.043339429795742034,730,0.0003,-5.432446575164795,0.2,13.737832927703858,,
75,4800,129.44117647058823,-280.18742514705883,0,41637,0.0,0.017677587,0.0890625,6.111212730407715,-0.04859136864542961,740,0.0003,-5.459812355041504,0.2,13.095634365081787,,
76,4864,129.44117647058823,-280.18742514705883,0,41729,0.0,0.01820511,0.0859375,6.396773338317871,-0.042153197713196276,750,0.0003,-5.406517839431762,0.2,13.641241455078125,,
77,4928,129.44117647058823,-280.18742514705883,0,41824,1.1920928955078125e-07,0.025982507,0.1515625,5.530246257781982,-0.05822089351713657,760,0.0003,-5.362298440933228,0.2,11.915291595458985,,
78,4992,129.44117647058823,-280.18742514705883,0,41914,0.0,0.021784266,0.115625,8.889939308166504,-0.04758907575160265,770,0.0003,-5.3411720275878904,0.2,18.750822448730467,,
79,5056,129.44117647058823,-280.18742514705883,0,42010,0.0,0.016031437,0.0921875,5.722137928009033,-0.04147411286830902,780,0.0003,-5.373646831512451,0.2,12.275255012512208,,
```

#### logs_PPO_0.0_baseline/log.txt

```
Logging to ./logs_PPO_0.0_baseline
---------------------------------
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 477      |
|    total_timesteps | 64       |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 17          |
|    ep_rew_mean          | -80.1       |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 2           |
|    time_elapsed         | 952         |
|    total_timesteps      | 128         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008775144 |
|    clip_fraction        | 0.0172      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.4       |
|    explained_variance   | 0.000827    |
|    learning_rate        | 0.0003      |
|    loss                 | 724         |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0408     |
|    value_loss           | 1.5e+03     |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 19.5         |
|    ep_rew_mean          | -98          |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 3            |
|    time_elapsed         | 1476         |
|    total_timesteps      | 192          |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0089736935 |
|    clip_fraction        | 0.0219       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | -1.19e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 1.06e+03     |
|    n_updates            | 20           |
|    policy_gradient_loss | -0.0403      |
|    value_loss           | 2.14e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 28.2        |
|    ep_rew_mean          | -157        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 4           |
|    time_elapsed         | 1873        |
|    total_timesteps      | 256         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009701499 |
|    clip_fraction        | 0.0172      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 857         |
|    n_updates            | 30          |
|    policy_gradient_loss | -0.0435     |
|    value_loss           | 1.74e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.2        |
|    ep_rew_mean          | -253        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 5           |
|    time_elapsed         | 2361        |
|    total_timesteps      | 320         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008791318 |
|    clip_fraction        | 0.0266      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 524         |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0372     |
|    value_loss           | 1.07e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 44.2         |
|    ep_rew_mean          | -253         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 6            |
|    time_elapsed         | 2833         |
|    total_timesteps      | 384          |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0068506813 |
|    clip_fraction        | 0.0109       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 759          |
|    n_updates            | 50           |
|    policy_gradient_loss | -0.0356      |
|    value_loss           | 1.54e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.2        |
|    ep_rew_mean          | -253        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 7           |
|    time_elapsed         | 3338        |
|    total_timesteps      | 448         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008605803 |
|    clip_fraction        | 0.0344      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 891         |
|    n_updates            | 60          |
|    policy_gradient_loss | -0.0402     |
|    value_loss           | 1.8e+03     |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 134         |
|    mean_reward          | -246        |
| time/                   |             |
|    total_timesteps      | 512         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.004366393 |
|    clip_fraction        | 0.00313     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 961         |
|    n_updates            | 70          |
|    policy_gradient_loss | -0.0257     |
|    value_loss           | 1.94e+03    |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 48.8     |
|    ep_rew_mean     | -273     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6535     |
|    total_timesteps | 512      |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 45          |
|    ep_rew_mean          | -250        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 9           |
|    time_elapsed         | 6996        |
|    total_timesteps      | 576         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008587489 |
|    clip_fraction        | 0.0141      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 612         |
|    n_updates            | 80          |
|    policy_gradient_loss | -0.0394     |
|    value_loss           | 1.24e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.9        |
|    ep_rew_mean          | -252        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 10          |
|    time_elapsed         | 7429        |
|    total_timesteps      | 640         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009217796 |
|    clip_fraction        | 0.0297      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 634         |
|    n_updates            | 90          |
|    policy_gradient_loss | -0.0369     |
|    value_loss           | 1.28e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.1        |
|    ep_rew_mean          | -245        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 11          |
|    time_elapsed         | 7908        |
|    total_timesteps      | 704         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008917995 |
|    clip_fraction        | 0.0141      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 467         |
|    n_updates            | 100         |
|    policy_gradient_loss | -0.0436     |
|    value_loss           | 950         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 41.5        |
|    ep_rew_mean          | -230        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 12          |
|    time_elapsed         | 8311        |
|    total_timesteps      | 768         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.005298556 |
|    clip_fraction        | 0.00469     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 1.08e+03    |
|    n_updates            | 110         |
|    policy_gradient_loss | -0.0302     |
|    value_loss           | 2.17e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 39.9        |
|    ep_rew_mean          | -219        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 13          |
|    time_elapsed         | 8688        |
|    total_timesteps      | 832         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008067757 |
|    clip_fraction        | 0.0234      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 511         |
|    n_updates            | 120         |
|    policy_gradient_loss | -0.0411     |
|    value_loss           | 1.04e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 42.1        |
|    ep_rew_mean          | -233        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 14          |
|    time_elapsed         | 9175        |
|    total_timesteps      | 896         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008645452 |
|    clip_fraction        | 0.0234      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 529         |
|    n_updates            | 130         |
|    policy_gradient_loss | -0.0382     |
|    value_loss           | 1.07e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44          |
|    ep_rew_mean          | -241        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 15          |
|    time_elapsed         | 9605        |
|    total_timesteps      | 960         |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.005753789 |
|    clip_fraction        | 0.0125      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 917         |
|    n_updates            | 140         |
|    policy_gradient_loss | -0.0306     |
|    value_loss           | 1.85e+03    |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 134         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 1024        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.012458517 |
|    clip_fraction        | 0.0406      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 344         |
|    n_updates            | 150         |
|    policy_gradient_loss | -0.0474     |
|    value_loss           | 703         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 44.7     |
|    ep_rew_mean     | -242     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 12792    |
|    total_timesteps | 1024     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 44.7         |
|    ep_rew_mean          | -242         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 17           |
|    time_elapsed         | 13262        |
|    total_timesteps      | 1088         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0065491106 |
|    clip_fraction        | 0.0109       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 667          |
|    n_updates            | 160          |
|    policy_gradient_loss | -0.0299      |
|    value_loss           | 1.35e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.3        |
|    ep_rew_mean          | -239        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 18          |
|    time_elapsed         | 13687       |
|    total_timesteps      | 1152        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.005327685 |
|    clip_fraction        | 0.00625     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 729         |
|    n_updates            | 170         |
|    policy_gradient_loss | -0.0297     |
|    value_loss           | 1.47e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 44.3        |
|    ep_rew_mean          | -239        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 19          |
|    time_elapsed         | 14115       |
|    total_timesteps      | 1216        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009876929 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 571         |
|    n_updates            | 180         |
|    policy_gradient_loss | -0.0415     |
|    value_loss           | 1.16e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 46.5         |
|    ep_rew_mean          | -248         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 20           |
|    time_elapsed         | 14611        |
|    total_timesteps      | 1280         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0073834946 |
|    clip_fraction        | 0.025        |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 712          |
|    n_updates            | 190          |
|    policy_gradient_loss | -0.0317      |
|    value_loss           | 1.44e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 46.5        |
|    ep_rew_mean          | -248        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 21          |
|    time_elapsed         | 15083       |
|    total_timesteps      | 1344        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.004955995 |
|    clip_fraction        | 0.00781     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 857         |
|    n_updates            | 200         |
|    policy_gradient_loss | -0.0231     |
|    value_loss           | 1.73e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 46.5         |
|    ep_rew_mean          | -248         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 22           |
|    time_elapsed         | 15470        |
|    total_timesteps      | 1408         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0051172785 |
|    clip_fraction        | 0.00781      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 752          |
|    n_updates            | 210          |
|    policy_gradient_loss | -0.0291      |
|    value_loss           | 1.52e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 47.9        |
|    ep_rew_mean          | -254        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 23          |
|    time_elapsed         | 15920       |
|    total_timesteps      | 1472        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008940698 |
|    clip_fraction        | 0.0219      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 540         |
|    n_updates            | 220         |
|    policy_gradient_loss | -0.0373     |
|    value_loss           | 1.1e+03     |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -303        |
| time/                   |             |
|    total_timesteps      | 1536        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.007823569 |
|    clip_fraction        | 0.0172      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 565         |
|    n_updates            | 230         |
|    policy_gradient_loss | -0.0351     |
|    value_loss           | 1.15e+03    |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 46.9     |
|    ep_rew_mean     | -250     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 19135    |
|    total_timesteps | 1536     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 49.2         |
|    ep_rew_mean          | -260         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 25           |
|    time_elapsed         | 19543        |
|    total_timesteps      | 1600         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0075978385 |
|    clip_fraction        | 0.0266       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 516          |
|    n_updates            | 240          |
|    policy_gradient_loss | -0.0293      |
|    value_loss           | 1.05e+03     |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 49.2         |
|    ep_rew_mean          | -260         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 26           |
|    time_elapsed         | 20007        |
|    total_timesteps      | 1664         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0051798346 |
|    clip_fraction        | 0.00469      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 1.19e-07     |
|    learning_rate        | 0.0003       |
|    loss                 | 747          |
|    n_updates            | 250          |
|    policy_gradient_loss | -0.0301      |
|    value_loss           | 1.51e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 49.1        |
|    ep_rew_mean          | -259        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 27          |
|    time_elapsed         | 20394       |
|    total_timesteps      | 1728        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.004718489 |
|    clip_fraction        | 0.00156     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 738         |
|    n_updates            | 260         |
|    policy_gradient_loss | -0.0282     |
|    value_loss           | 1.49e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 50.3        |
|    ep_rew_mean          | -265        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 28          |
|    time_elapsed         | 20820       |
|    total_timesteps      | 1792        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.007951127 |
|    clip_fraction        | 0.0187      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 360         |
|    n_updates            | 270         |
|    policy_gradient_loss | -0.039      |
|    value_loss           | 734         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 50.3        |
|    ep_rew_mean          | -265        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 29          |
|    time_elapsed         | 21257       |
|    total_timesteps      | 1856        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008100969 |
|    clip_fraction        | 0.0203      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 581         |
|    n_updates            | 280         |
|    policy_gradient_loss | -0.0374     |
|    value_loss           | 1.18e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 52          |
|    ep_rew_mean          | -273        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 30          |
|    time_elapsed         | 21682       |
|    total_timesteps      | 1920        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.006489272 |
|    clip_fraction        | 0.00781     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 851         |
|    n_updates            | 290         |
|    policy_gradient_loss | -0.0295     |
|    value_loss           | 1.72e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 52          |
|    ep_rew_mean          | -273        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 31          |
|    time_elapsed         | 22082       |
|    total_timesteps      | 1984        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009070935 |
|    clip_fraction        | 0.0375      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 513         |
|    n_updates            | 300         |
|    policy_gradient_loss | -0.0319     |
|    value_loss           | 1.04e+03    |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -383        |
| time/                   |             |
|    total_timesteps      | 2048        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.010139834 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 463         |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.041      |
|    value_loss           | 941         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 52.9     |
|    ep_rew_mean     | -275     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 25275    |
|    total_timesteps | 2048     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 52.9       |
|    ep_rew_mean          | -272       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 33         |
|    time_elapsed         | 25645      |
|    total_timesteps      | 2112       |
| train/                  |            |
|    adaptive_beta        | 0.5        |
|    approx_kl            | 0.01197361 |
|    clip_fraction        | 0.0453     |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.3      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 337        |
|    n_updates            | 320        |
|    policy_gradient_loss | -0.0467    |
|    value_loss           | 687        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 51.9        |
|    ep_rew_mean          | -268        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 34          |
|    time_elapsed         | 26058       |
|    total_timesteps      | 2176        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.014939003 |
|    clip_fraction        | 0.0672      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 300         |
|    n_updates            | 330         |
|    policy_gradient_loss | -0.0521     |
|    value_loss           | 614         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 51.9        |
|    ep_rew_mean          | -268        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 35          |
|    time_elapsed         | 26517       |
|    total_timesteps      | 2240        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009013851 |
|    clip_fraction        | 0.0359      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 512         |
|    n_updates            | 340         |
|    policy_gradient_loss | -0.032      |
|    value_loss           | 1.04e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 53           |
|    ep_rew_mean          | -271         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 36           |
|    time_elapsed         | 26945        |
|    total_timesteps      | 2304         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0043333974 |
|    clip_fraction        | 0.00313      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 5.96e-08     |
|    learning_rate        | 0.0003       |
|    loss                 | 904          |
|    n_updates            | 350          |
|    policy_gradient_loss | -0.0267      |
|    value_loss           | 1.83e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 54.2        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 37          |
|    time_elapsed         | 27371       |
|    total_timesteps      | 2368        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.007663547 |
|    clip_fraction        | 0.0203      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 651         |
|    n_updates            | 360         |
|    policy_gradient_loss | -0.0347     |
|    value_loss           | 1.32e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 54.2        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 38          |
|    time_elapsed         | 27826       |
|    total_timesteps      | 2432        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.004909193 |
|    clip_fraction        | 0.00156     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 679         |
|    n_updates            | 370         |
|    policy_gradient_loss | -0.0294     |
|    value_loss           | 1.37e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 54.8         |
|    ep_rew_mean          | -281         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 39           |
|    time_elapsed         | 28274        |
|    total_timesteps      | 2496         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0050714007 |
|    clip_fraction        | 0.00469      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | -1.19e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 970          |
|    n_updates            | 380          |
|    policy_gradient_loss | -0.0253      |
|    value_loss           | 1.96e+03     |
------------------------------------------
------------------------------------------
| eval/                   |              |
|    mean_ep_length       | 400          |
|    mean_reward          | -183         |
| time/                   |              |
|    total_timesteps      | 2560         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0065006893 |
|    clip_fraction        | 0.0141       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 832          |
|    n_updates            | 390          |
|    policy_gradient_loss | -0.0321      |
|    value_loss           | 1.68e+03     |
------------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 54.8     |
|    ep_rew_mean     | -281     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 31480    |
|    total_timesteps | 2560     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 55.4         |
|    ep_rew_mean          | -283         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 41           |
|    time_elapsed         | 31860        |
|    total_timesteps      | 2624         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0048848456 |
|    clip_fraction        | 0.00625      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.2        |
|    explained_variance   | -2.38e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 788          |
|    n_updates            | 400          |
|    policy_gradient_loss | -0.0268      |
|    value_loss           | 1.59e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56.6        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 42          |
|    time_elapsed         | 32335       |
|    total_timesteps      | 2688        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.010001464 |
|    clip_fraction        | 0.0344      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 567         |
|    n_updates            | 410         |
|    policy_gradient_loss | -0.0368     |
|    value_loss           | 1.15e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56.6        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 43          |
|    time_elapsed         | 32739       |
|    total_timesteps      | 2752        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.006001888 |
|    clip_fraction        | 0.00781     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 834         |
|    n_updates            | 420         |
|    policy_gradient_loss | -0.0299     |
|    value_loss           | 1.68e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.1        |
|    ep_rew_mean          | -294        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 44          |
|    time_elapsed         | 33133       |
|    total_timesteps      | 2816        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.010678176 |
|    clip_fraction        | 0.0281      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 640         |
|    n_updates            | 430         |
|    policy_gradient_loss | -0.039      |
|    value_loss           | 1.3e+03     |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56.2        |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 45          |
|    time_elapsed         | 33523       |
|    total_timesteps      | 2880        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.006745995 |
|    clip_fraction        | 0.0219      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 574         |
|    n_updates            | 440         |
|    policy_gradient_loss | -0.0333     |
|    value_loss           | 1.16e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56.7        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 46          |
|    time_elapsed         | 33933       |
|    total_timesteps      | 2944        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009983155 |
|    clip_fraction        | 0.0266      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 440         |
|    n_updates            | 450         |
|    policy_gradient_loss | -0.0394     |
|    value_loss           | 895         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57          |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 47          |
|    time_elapsed         | 34334       |
|    total_timesteps      | 3008        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.004903609 |
|    clip_fraction        | 0.00781     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 530         |
|    n_updates            | 460         |
|    policy_gradient_loss | -0.0244     |
|    value_loss           | 1.08e+03    |
-----------------------------------------
---------------------------------------
| eval/                   |           |
|    mean_ep_length       | 400       |
|    mean_reward          | -152      |
| time/                   |           |
|    total_timesteps      | 3072      |
| train/                  |           |
|    adaptive_beta        | 0.5       |
|    approx_kl            | 0.0191808 |
|    clip_fraction        | 0.0984    |
|    clip_range           | 0.2       |
|    entropy_loss         | -10.3     |
|    explained_variance   | 5.96e-08  |
|    learning_rate        | 0.0003    |
|    loss                 | 163       |
|    n_updates            | 470       |
|    policy_gradient_loss | -0.0545   |
|    value_loss           | 338       |
---------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 57       |
|    ep_rew_mean     | -291     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 37580    |
|    total_timesteps | 3072     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57          |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 49          |
|    time_elapsed         | 37940       |
|    total_timesteps      | 3136        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.006385441 |
|    clip_fraction        | 0.0125      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 652         |
|    n_updates            | 480         |
|    policy_gradient_loss | -0.0256     |
|    value_loss           | 1.32e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57          |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 50          |
|    time_elapsed         | 38342       |
|    total_timesteps      | 3200        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008175593 |
|    clip_fraction        | 0.0125      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 536         |
|    n_updates            | 490         |
|    policy_gradient_loss | -0.038      |
|    value_loss           | 1.09e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 57.1         |
|    ep_rew_mean          | -291         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 51           |
|    time_elapsed         | 38688        |
|    total_timesteps      | 3264         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0064408993 |
|    clip_fraction        | 0.0109       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.2        |
|    explained_variance   | -1.19e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 662          |
|    n_updates            | 500          |
|    policy_gradient_loss | -0.0298      |
|    value_loss           | 1.34e+03     |
------------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 57.6         |
|    ep_rew_mean          | -293         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 52           |
|    time_elapsed         | 39068        |
|    total_timesteps      | 3328         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0073126256 |
|    clip_fraction        | 0.00937      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.2        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 338          |
|    n_updates            | 510          |
|    policy_gradient_loss | -0.0309      |
|    value_loss           | 689          |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.6        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 53          |
|    time_elapsed         | 39464       |
|    total_timesteps      | 3392        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.004882048 |
|    clip_fraction        | 0.00781     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 457         |
|    n_updates            | 520         |
|    policy_gradient_loss | -0.0217     |
|    value_loss           | 929         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.5        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 54          |
|    time_elapsed         | 39867       |
|    total_timesteps      | 3456        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.005696806 |
|    clip_fraction        | 0.0109      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 702         |
|    n_updates            | 530         |
|    policy_gradient_loss | -0.03       |
|    value_loss           | 1.42e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56.9        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 55          |
|    time_elapsed         | 40266       |
|    total_timesteps      | 3520        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.007261212 |
|    clip_fraction        | 0.00937     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 454         |
|    n_updates            | 540         |
|    policy_gradient_loss | -0.0367     |
|    value_loss           | 922         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 134         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 3584        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.009033485 |
|    clip_fraction        | 0.0297      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 415         |
|    n_updates            | 550         |
|    policy_gradient_loss | -0.0379     |
|    value_loss           | 844         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 57.4     |
|    ep_rew_mean     | -288     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 43418    |
|    total_timesteps | 3584     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 57.4       |
|    ep_rew_mean          | -288       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 57         |
|    time_elapsed         | 43812      |
|    total_timesteps      | 3648       |
| train/                  |            |
|    adaptive_beta        | 0.5        |
|    approx_kl            | 0.00967169 |
|    clip_fraction        | 0.0297     |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.2      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 468        |
|    n_updates            | 560        |
|    policy_gradient_loss | -0.0329    |
|    value_loss           | 951        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.4        |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 58          |
|    time_elapsed         | 44235       |
|    total_timesteps      | 3712        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008976412 |
|    clip_fraction        | 0.0281      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 748         |
|    n_updates            | 570         |
|    policy_gradient_loss | -0.0281     |
|    value_loss           | 1.51e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 57.7         |
|    ep_rew_mean          | -289         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 59           |
|    time_elapsed         | 44602        |
|    total_timesteps      | 3776         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0036636228 |
|    clip_fraction        | 0.00156      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 787          |
|    n_updates            | 580          |
|    policy_gradient_loss | -0.0192      |
|    value_loss           | 1.59e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.2        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 60          |
|    time_elapsed         | 44995       |
|    total_timesteps      | 3840        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.011485878 |
|    clip_fraction        | 0.0672      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 433         |
|    n_updates            | 590         |
|    policy_gradient_loss | -0.0398     |
|    value_loss           | 880         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.9        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 61          |
|    time_elapsed         | 45408       |
|    total_timesteps      | 3904        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008639585 |
|    clip_fraction        | 0.0266      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 373         |
|    n_updates            | 600         |
|    policy_gradient_loss | -0.04       |
|    value_loss           | 761         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.9        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 62          |
|    time_elapsed         | 45825       |
|    total_timesteps      | 3968        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.011541786 |
|    clip_fraction        | 0.0422      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 481         |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.0443     |
|    value_loss           | 978         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.9        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 63          |
|    time_elapsed         | 46238       |
|    total_timesteps      | 4032        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.010120759 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 548         |
|    n_updates            | 620         |
|    policy_gradient_loss | -0.0405     |
|    value_loss           | 1.11e+03    |
-----------------------------------------
------------------------------------------
| eval/                   |              |
|    mean_ep_length       | 399          |
|    mean_reward          | -65.5        |
| time/                   |              |
|    total_timesteps      | 4096         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0049386974 |
|    clip_fraction        | 0.00937      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 1.79e-07     |
|    learning_rate        | 0.0003       |
|    loss                 | 618          |
|    n_updates            | 630          |
|    policy_gradient_loss | -0.0255      |
|    value_loss           | 1.25e+03     |
------------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 59       |
|    ep_rew_mean     | -293     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 49489    |
|    total_timesteps | 4096     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 59.1        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 65          |
|    time_elapsed         | 49899       |
|    total_timesteps      | 4160        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008993767 |
|    clip_fraction        | 0.0297      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 601         |
|    n_updates            | 640         |
|    policy_gradient_loss | -0.0361     |
|    value_loss           | 1.22e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 59.4        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 66          |
|    time_elapsed         | 50364       |
|    total_timesteps      | 4224        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.017608395 |
|    clip_fraction        | 0.0969      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 247         |
|    n_updates            | 650         |
|    policy_gradient_loss | -0.0515     |
|    value_loss           | 507         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 59.1        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 67          |
|    time_elapsed         | 50825       |
|    total_timesteps      | 4288        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.012729534 |
|    clip_fraction        | 0.0641      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 641         |
|    n_updates            | 660         |
|    policy_gradient_loss | -0.0365     |
|    value_loss           | 1.3e+03     |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.4        |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 68          |
|    time_elapsed         | 51260       |
|    total_timesteps      | 4352        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.005877954 |
|    clip_fraction        | 0.00937     |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 656         |
|    n_updates            | 670         |
|    policy_gradient_loss | -0.0288     |
|    value_loss           | 1.33e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 58.6         |
|    ep_rew_mean          | -290         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 69           |
|    time_elapsed         | 51676        |
|    total_timesteps      | 4416         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0067448057 |
|    clip_fraction        | 0.0156       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | -2.38e-07    |
|    learning_rate        | 0.0003       |
|    loss                 | 689          |
|    n_updates            | 680          |
|    policy_gradient_loss | -0.0332      |
|    value_loss           | 1.39e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58          |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 70          |
|    time_elapsed         | 52105       |
|    total_timesteps      | 4480        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.014141137 |
|    clip_fraction        | 0.0656      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 358         |
|    n_updates            | 690         |
|    policy_gradient_loss | -0.0502     |
|    value_loss           | 731         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.9        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 71          |
|    time_elapsed         | 52545       |
|    total_timesteps      | 4544        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.008160599 |
|    clip_fraction        | 0.0281      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 665         |
|    n_updates            | 700         |
|    policy_gradient_loss | -0.0341     |
|    value_loss           | 1.35e+03    |
-----------------------------------------
------------------------------------------
| eval/                   |              |
|    mean_ep_length       | 200          |
|    mean_reward          | -184         |
| time/                   |              |
|    total_timesteps      | 4608         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0066908803 |
|    clip_fraction        | 0.0172       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 5.96e-08     |
|    learning_rate        | 0.0003       |
|    loss                 | 631          |
|    n_updates            | 710          |
|    policy_gradient_loss | -0.0331      |
|    value_loss           | 1.28e+03     |
------------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 58.3     |
|    ep_rew_mean     | -287     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 55724    |
|    total_timesteps | 4608     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.4        |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 73          |
|    time_elapsed         | 56141       |
|    total_timesteps      | 4672        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.007345749 |
|    clip_fraction        | 0.0219      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 866         |
|    n_updates            | 720         |
|    policy_gradient_loss | -0.0349     |
|    value_loss           | 1.75e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 58.4        |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 74          |
|    time_elapsed         | 56520       |
|    total_timesteps      | 4736        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.010898312 |
|    clip_fraction        | 0.0781      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 539         |
|    n_updates            | 730         |
|    policy_gradient_loss | -0.0334     |
|    value_loss           | 1.09e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 58.2         |
|    ep_rew_mean          | -288         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 75           |
|    time_elapsed         | 56919        |
|    total_timesteps      | 4800         |
| train/                  |              |
|    adaptive_beta        | 0.5          |
|    approx_kl            | 0.0049218806 |
|    clip_fraction        | 0.00156      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.2        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 685          |
|    n_updates            | 740          |
|    policy_gradient_loss | -0.0224      |
|    value_loss           | 1.39e+03     |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.9        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 76          |
|    time_elapsed         | 57323       |
|    total_timesteps      | 4864        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.012277812 |
|    clip_fraction        | 0.0484      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 338         |
|    n_updates            | 750         |
|    policy_gradient_loss | -0.0448     |
|    value_loss           | 690         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.5        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 77          |
|    time_elapsed         | 57756       |
|    total_timesteps      | 4928        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.010520387 |
|    clip_fraction        | 0.0328      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 506         |
|    n_updates            | 760         |
|    policy_gradient_loss | -0.04       |
|    value_loss           | 1.03e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.5        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 78          |
|    time_elapsed         | 58171       |
|    total_timesteps      | 4992        |
| train/                  |             |
|    adaptive_beta        | 0.5         |
|    approx_kl            | 0.020131372 |
|    clip_fraction        | 0.111       |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 266         |
|    n_updates            | 770         |
|    policy_gradient_loss | -0.061      |
|    value_loss           | 544         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 57.5       |
|    ep_rew_mean          | -283       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 79         |
|    time_elapsed         | 58661      |
|    total_timesteps      | 5056       |
| train/                  |            |
|    adaptive_beta        | 0.475      |
|    approx_kl            | 0.00860951 |
|    clip_fraction        | 0.0281     |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.3      |
|    explained_variance   | 1.19e-07   |
|    learning_rate        | 0.0003     |
|    loss                 | 775        |
|    n_updates            | 780        |
|    policy_gradient_loss | -0.0238    |
|    value_loss           | 1.57e+03   |
----------------------------------------
```

#### logs_PPO_0.0_baseline/progress.csv

```
time/iterations,train/adaptive_beta,time/total_timesteps,time/fps,time/time_elapsed,train/explained_variance,train/approx_kl,train/clip_fraction,train/loss,rollout/ep_len_mean,train/policy_gradient_loss,train/n_updates,train/learning_rate,train/entropy_loss,train/clip_range,rollout/ep_rew_mean,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,0.5,64,0,477,,,,,,,,,,,,,,
2,0.5,128,0,952,0.0008271336555480957,0.008775144,0.0171875,723.9107666015625,17.0,-0.040817347913980485,10,0.0003,-10.355391216278075,0.2,-80.138998,1504.6678588867187,,
3,0.5,192,0,1476,-1.1920928955078125e-07,0.0089736935,0.021875,1056.9141845703125,19.5,-0.040296001359820366,20,0.0003,-10.346981048583984,0.2,-98.01493049999999,2142.7298583984375,,
4,0.5,256,0,1873,5.960464477539063e-08,0.009701499,0.0171875,856.8851318359375,28.25,-0.04346215948462486,30,0.0003,-10.338719081878661,0.2,-157.036147,1738.7186401367187,,
5,0.5,320,0,2361,1.1920928955078125e-07,0.008791318,0.0265625,524.3536376953125,44.166666666666664,-0.03721475563943386,40,0.0003,-10.33558464050293,0.2,-252.61037716666667,1069.4129638671875,,
6,0.5,384,0,2833,0.0,0.0068506813,0.0109375,758.6131591796875,44.166666666666664,-0.03564730733633041,50,0.0003,-10.328399181365967,0.2,-252.61037716666667,1539.2163818359375,,
7,0.5,448,0,3338,-1.1920928955078125e-07,0.008605803,0.034375,890.9860229492188,44.166666666666664,-0.04021642245352268,60,0.0003,-10.325566482543945,0.2,-252.61037716666667,1803.6217895507812,,
,0.5,512,,,1.1920928955078125e-07,0.004366393,0.003125,960.7277221679688,,-0.025739349797368048,70,0.0003,-10.322179794311523,0.2,,1942.889208984375,-246.32264259999997,134.0
8,0.5,512,0,6535,,,,,48.75,,,,,,-273.03467262500004,,,
9,0.5,576,0,6996,-1.1920928955078125e-07,0.008587489,0.0140625,611.67822265625,45.0,-0.039387839846313,80,0.0003,-10.320841979980468,0.2,-249.55330181818184,1240.9162963867188,,
10,0.5,640,0,7429,-1.1920928955078125e-07,0.009217796,0.0296875,633.792724609375,44.92307692307692,-0.03693355275318026,90,0.0003,-10.321279811859132,0.2,-252.03389892307695,1284.3755493164062,,
11,0.5,704,0,7908,1.1920928955078125e-07,0.008917995,0.0140625,466.64752197265625,44.142857142857146,-0.04362080693244934,100,0.0003,-10.317298889160156,0.2,-245.39562478571432,949.63134765625,,
12,0.5,768,0,8311,0.0,0.005298556,0.0046875,1076.5880126953125,41.53333333333333,-0.030165875144302846,110,0.0003,-10.308085918426514,0.2,-230.3428642666667,2171.8470947265623,,
13,0.5,832,0,8688,0.0,0.008067757,0.0234375,510.7551574707031,39.9375,-0.041080678999423983,120,0.0003,-10.300148677825927,0.2,-219.4606534375,1038.0905883789062,,
14,0.5,896,0,9175,-1.1920928955078125e-07,0.008645452,0.0234375,529.3212890625,42.05882352941177,-0.038243461214005944,130,0.0003,-10.286701583862305,0.2,-232.62599123529412,1074.8246459960938,,
15,0.5,960,0,9605,5.960464477539063e-08,0.005753789,0.0125,916.5422973632812,44.05,-0.030622915737330912,140,0.0003,-10.277848720550537,0.2,-240.57972440000003,1851.1616577148438,,
,0.5,1024,,,1.1920928955078125e-07,0.012458517,0.040625,344.27783203125,,-0.0473692798987031,150,0.0003,-10.287588024139405,0.2,,703.3002136230468,-183.93985600000002,134.0
16,0.5,1024,0,12792,,,,,44.714285714285715,,,,,,-242.04656409523812,,,
17,0.5,1088,0,13262,0.0,0.0065491106,0.0109375,667.0061645507812,44.714285714285715,-0.02994768600910902,160,0.0003,-10.27362127304077,0.2,-242.04656409523812,1350.603857421875,,
18,0.5,1152,0,13687,1.1920928955078125e-07,0.005327685,0.00625,728.674072265625,44.30434782608695,-0.029652186669409275,170,0.0003,-10.266766262054443,0.2,-238.61185573913048,1474.4362182617188,,
19,0.5,1216,0,14115,0.0,0.009876929,0.03125,570.8595581054688,44.30434782608695,-0.041542465426027776,180,0.0003,-10.267069435119629,0.2,-238.61185573913048,1157.7235961914062,,
20,0.5,1280,0,14611,0.0,0.0073834946,0.025,712.466796875,46.5,-0.03166433498263359,190,0.0003,-10.268619060516357,0.2,-247.8496915769231,1441.9086059570313,,
21,0.5,1344,0,15083,0.0,0.004955995,0.0078125,856.6036376953125,46.5,-0.023072408512234688,200,0.0003,-10.283522701263427,0.2,-247.8496915769231,1729.9615112304687,,
22,0.5,1408,0,15470,0.0,0.0051172785,0.0078125,752.3191528320312,46.5,-0.029086767323315145,210,0.0003,-10.290037155151367,0.2,-247.8496915769231,1521.5853637695313,,
23,0.5,1472,0,15920,0.0,0.008940698,0.021875,539.9696044921875,47.892857142857146,-0.037346628680825236,220,0.0003,-10.297621154785157,0.2,-254.28481628571427,1095.7163208007812,,
,0.5,1536,,,0.0,0.007823569,0.0171875,565.3762817382812,,-0.03507373668253422,230,0.0003,-10.298096752166748,0.2,,1146.2746826171874,-302.97866300000004,400.0
24,0.5,1536,0,19135,,,,,46.93103448275862,,,,,,-249.5270776896552,,,
25,0.5,1600,0,19543,0.0,0.0075978385,0.0265625,516.2955932617188,49.193548387096776,-0.02932664342224598,240,0.0003,-10.286420440673828,0.2,-259.9377850645161,1048.0137329101562,,
26,0.5,1664,0,20007,1.1920928955078125e-07,0.0051798346,0.0046875,746.6924438476562,49.193548387096776,-0.03009347729384899,250,0.0003,-10.28032169342041,0.2,-259.9377850645161,1509.679052734375,,
27,0.5,1728,0,20394,-1.1920928955078125e-07,0.004718489,0.0015625,738.4351806640625,49.125,-0.028196763433516025,260,0.0003,-10.273220825195313,0.2,-259.436552125,1493.6058715820313,,
28,0.5,1792,0,20820,-1.1920928955078125e-07,0.007951127,0.01875,359.7951965332031,50.333333333333336,-0.03898058421909809,270,0.0003,-10.276296043395996,0.2,-264.89154275757573,733.9296081542968,,
29,0.5,1856,0,21257,0.0,0.008100969,0.0203125,580.8865966796875,50.333333333333336,-0.037374342978000644,280,0.0003,-10.270807552337647,0.2,-264.89154275757573,1177.6253173828125,,
30,0.5,1920,0,21682,0.0,0.006489272,0.0078125,851.3931274414062,52.0,-0.029454460926353933,290,0.0003,-10.26986379623413,0.2,-272.5511039714285,1720.0216430664063,,
31,0.5,1984,0,22082,0.0,0.009070935,0.0375,512.7886962890625,52.0,-0.03191499374806881,300,0.0003,-10.285608386993408,0.2,-272.5511039714285,1040.7399169921875,,
,0.5,2048,,,0.0,0.010139834,0.03125,462.85687255859375,,-0.04097283165901899,310,0.0003,-10.29005069732666,0.2,,941.0993896484375,-382.5546254,400.0
32,0.5,2048,0,25275,,,,,52.86486486486486,,,,,,-274.6357182972972,,,
33,0.5,2112,0,25645,0.0,0.01197361,0.0453125,336.6268310546875,52.89473684210526,-0.04669506037607789,320,0.0003,-10.283967971801758,0.2,-272.19539928947364,687.4011779785156,,
34,0.5,2176,0,26058,5.960464477539063e-08,0.014939003,0.0671875,299.80670166015625,51.94871794871795,-0.05212688446044922,330,0.0003,-10.275076770782471,0.2,-267.5720086666666,613.619189453125,,
35,0.5,2240,0,26517,0.0,0.009013851,0.0359375,512.107421875,51.94871794871795,-0.0319879150018096,340,0.0003,-10.286543560028075,0.2,-267.5720086666666,1039.6745483398438,,
36,0.5,2304,0,26945,5.960464477539063e-08,0.0043333974,0.003125,904.3197021484375,53.05,-0.02665999261662364,350,0.0003,-10.282144260406493,0.2,-270.65059279999997,1825.8314453125,,
37,0.5,2368,0,27371,5.960464477539063e-08,0.007663547,0.0203125,650.86328125,54.214285714285715,-0.03469570009037852,360,0.0003,-10.271163940429688,0.2,-276.9820093333333,1317.4924438476562,,
38,0.5,2432,0,27826,1.1920928955078125e-07,0.004909193,0.0015625,679.206298828125,54.214285714285715,-0.029382253624498846,370,0.0003,-10.263722324371338,0.2,-276.9820093333333,1373.9478149414062,,
39,0.5,2496,0,28274,-1.1920928955078125e-07,0.0050714007,0.0046875,970.0043334960938,54.83720930232558,-0.025345286168158055,380,0.0003,-10.258830738067626,0.2,-280.91715897674413,1957.4603515625,,
,0.5,2560,,,0.0,0.0065006893,0.0140625,831.699951171875,,-0.032124263048171994,390,0.0003,-10.251629638671876,0.2,,1680.071630859375,-183.0530534,400.0
40,0.5,2560,0,31480,,,,,54.83720930232558,,,,,,-280.91715897674413,,,
41,0.5,2624,0,31860,-2.384185791015625e-07,0.0048848456,0.00625,788.0926513671875,55.40909090909091,-0.026797479949891567,400,0.0003,-10.242005252838135,0.2,-283.30973372727266,1593.03935546875,,
42,0.5,2688,0,32335,-1.1920928955078125e-07,0.010001464,0.034375,566.7413330078125,56.58695652173913,-0.03676584139466286,410,0.0003,-10.244054126739503,0.2,-289.8482002826086,1149.0857788085937,,
43,0.5,2752,0,32739,-1.1920928955078125e-07,0.006001888,0.0078125,834.0075073242188,56.58695652173913,-0.029882955364882946,420,0.0003,-10.253403854370116,0.2,-289.8482002826086,1684.2899536132813,,
44,0.5,2816,0,33133,-1.1920928955078125e-07,0.010678176,0.028125,640.1600952148438,57.06382978723404,-0.038993284478783606,430,0.0003,-10.26132869720459,0.2,-293.54792012765955,1296.333154296875,,
45,0.5,2880,0,33523,0.0,0.006745995,0.021875,574.082275390625,56.1875,-0.03329250887036324,440,0.0003,-10.26620798110962,0.2,-288.42069147916663,1163.8370727539063,,
46,0.5,2944,0,33933,0.0,0.009983155,0.0265625,440.1138000488281,56.714285714285715,-0.039384339563548566,450,0.0003,-10.263336658477783,0.2,-291.0161867346938,894.86015625,,
47,0.5,3008,0,34334,0.0,0.004903609,0.0078125,529.91357421875,56.96153846153846,-0.02443372942507267,460,0.0003,-10.262869262695313,0.2,-290.86076330769225,1075.21875,,
,0.5,3072,,,5.960464477539063e-08,0.0191808,0.0984375,162.5570831298828,,-0.054526195488870145,470,0.0003,-10.26213722229004,0.2,,337.5585235595703,-152.18775019999998,400.0
48,0.5,3072,0,37580,,,,,56.96153846153846,,,,,,-290.86076330769225,,,
49,0.5,3136,0,37940,0.0,0.006385441,0.0125,651.5753173828125,57.0377358490566,-0.025621788389980794,480,0.0003,-10.251581764221191,0.2,-290.50393662264145,1319.1015625,,
50,0.5,3200,0,38342,5.960464477539063e-08,0.008175593,0.0125,536.0147705078125,57.0377358490566,-0.03797983657568693,490,0.0003,-10.238527393341064,0.2,-290.50393662264145,1086.839453125,,
51,0.5,3264,0,38688,-1.1920928955078125e-07,0.0064408993,0.0109375,661.7669067382812,57.05555555555556,-0.029799073189496993,500,0.0003,-10.225236892700195,0.2,-290.9779435555555,1339.5085815429688,,
52,0.5,3328,0,39068,0.0,0.0073126256,0.009375,337.5339050292969,57.56363636363636,-0.030936050415039062,510,0.0003,-10.219273090362549,0.2,-292.6200113636363,689.2230712890625,,
53,0.5,3392,0,39464,1.1920928955078125e-07,0.004882048,0.0078125,456.87841796875,57.56363636363636,-0.021688850596547127,520,0.0003,-10.228518676757812,0.2,-292.6200113636363,928.5195922851562,,
54,0.5,3456,0,39867,0.0,0.005696806,0.0109375,702.350341796875,57.526315789473685,-0.029994112998247148,530,0.0003,-10.227580451965332,0.2,-289.8820496842106,1420.9731689453124,,
55,0.5,3520,0,40266,0.0,0.007261212,0.009375,453.8705749511719,56.93333333333333,-0.03668896965682507,540,0.0003,-10.219633483886719,0.2,-286.02792425000007,922.0819702148438,,
,0.5,3584,,,0.0,0.009033485,0.0296875,415.0466613769531,,-0.03788103684782982,550,0.0003,-10.209865188598632,0.2,,843.8651245117187,-184.0954536,134.0
56,0.5,3584,0,43418,,,,,57.39344262295082,,,,,,-288.41480406557383,,,
57,0.5,3648,0,43812,0.0,0.00967169,0.0296875,468.1830749511719,57.39344262295082,-0.03293892964720726,560,0.0003,-10.219395446777344,0.2,-288.41480406557383,951.2906799316406,,
58,0.5,3712,0,44235,0.0,0.008976412,0.028125,748.3592529296875,57.39344262295082,-0.028138116374611853,570,0.0003,-10.251610851287841,0.2,-288.41480406557383,1513.4375,,
59,0.5,3776,0,44602,0.0,0.0036636228,0.0015625,786.8689575195312,57.74193548387097,-0.01920783668756485,580,0.0003,-10.274611854553223,0.2,-288.8488545645162,1590.05859375,,
60,0.5,3840,0,44995,0.0,0.011485878,0.0671875,432.9615173339844,58.1875,-0.03979852832853794,590,0.0003,-10.276855945587158,0.2,-290.51242106250004,880.4317993164062,,
61,0.5,3904,0,45408,0.0,0.008639585,0.0265625,373.3941955566406,58.86153846153846,-0.04002140872180462,600,0.0003,-10.274800300598145,0.2,-292.6033289538462,760.6007385253906,,
62,0.5,3968,0,45825,1.7881393432617188e-07,0.011541786,0.0421875,481.4848937988281,58.86153846153846,-0.044291557557880876,610,0.0003,-10.271378326416016,0.2,-292.6033289538462,977.7306762695313,,
63,0.5,4032,0,46238,1.1920928955078125e-07,0.010120759,0.03125,547.8778686523438,58.86153846153846,-0.04050256488844752,620,0.0003,-10.266210842132569,0.2,-292.6033289538462,1111.3133056640625,,
,0.5,4096,,,1.7881393432617188e-07,0.0049386974,0.009375,617.5272216796875,,-0.025473016314208508,630,0.0003,-10.257550907135009,0.2,,1250.8161010742188,-65.5333128,399.0
64,0.5,4096,0,49489,,,,,58.96969696969697,,,,,,-293.2371662121213,,,
65,0.5,4160,0,49899,0.0,0.008993767,0.0296875,601.0685424804688,59.072463768115945,-0.03605167884379625,640,0.0003,-10.262525844573975,0.2,-291.4580111594203,1218.1291748046874,,
66,0.5,4224,0,50364,-1.1920928955078125e-07,0.017608395,0.096875,247.03355407714844,59.42857142857143,-0.05151988603174686,650,0.0003,-10.269373893737793,0.2,-293.09463051428577,507.0912750244141,,
67,0.5,4288,0,50825,5.960464477539063e-08,0.012729534,0.0640625,641.298095703125,59.056338028169016,-0.03652660753577948,660,0.0003,-10.271353149414063,0.2,-291.12031601408455,1298.1526245117188,,
68,0.5,4352,0,51260,0.0,0.005877954,0.009375,655.6962890625,58.43055555555556,-0.028795947693288325,670,0.0003,-10.269927024841309,0.2,-288.20818538888886,1327.0997192382813,,
69,0.5,4416,0,51676,-2.384185791015625e-07,0.0067448057,0.015625,689.229736328125,58.567567567567565,-0.03319482635706663,680,0.0003,-10.272200584411621,0.2,-289.7466883783784,1394.0381591796875,,
70,0.5,4480,0,52105,0.0,0.014141137,0.065625,358.2071838378906,57.986666666666665,-0.05015700701624155,690,0.0003,-10.274212837219238,0.2,-286.8023248933334,730.52294921875,,
71,0.5,4544,0,52545,0.0,0.008160599,0.028125,665.1740112304688,57.89473684210526,-0.034060954302549365,700,0.0003,-10.261737728118897,0.2,-286.0858968026316,1345.90625,,
,0.5,4608,,,5.960464477539063e-08,0.0066908803,0.0171875,631.0313720703125,,-0.033133311197161674,710,0.0003,-10.252879238128662,0.2,,1277.50771484375,-184.0400702,200.0
72,0.5,4608,0,55724,,,,,58.25974025974026,,,,,,-287.4932291948052,,,
73,0.5,4672,0,56141,-1.1920928955078125e-07,0.007345749,0.021875,866.4751586914062,58.38461538461539,-0.03486405303701758,720,0.0003,-10.245021724700928,0.2,-288.44553257692314,1749.4297607421875,,
74,0.5,4736,0,56520,0.0,0.010898312,0.078125,539.0206298828125,58.38461538461539,-0.03336950354278088,730,0.0003,-10.232364273071289,0.2,-288.44553257692314,1092.9517578125,,
75,0.5,4800,0,56919,0.0,0.0049218806,0.0015625,685.0012817382812,58.2125,-0.022367911972105504,740,0.0003,-10.229137325286866,0.2,-287.83391140000003,1386.2855712890625,,
76,0.5,4864,0,57323,-1.1920928955078125e-07,0.012277812,0.0484375,338.10821533203125,57.9390243902439,-0.04480511862784624,750,0.0003,-10.264085006713866,0.2,-286.3801418048781,689.7089599609375,,
77,0.5,4928,0,57756,1.1920928955078125e-07,0.010520387,0.0328125,506.11163330078125,57.49411764705882,-0.039998847991228104,760,0.0003,-10.266220474243164,0.2,-282.75796045882356,1026.9126220703124,,
78,0.5,4992,0,58171,-1.1920928955078125e-07,0.020131372,0.1109375,265.7834167480469,57.49411764705882,-0.061036494374275205,770,0.0003,-10.26146640777588,0.2,-282.75796045882356,544.06396484375,,
79,0.475,5056,0,58661,1.1920928955078125e-07,0.00860951,0.028125,774.5276489257812,57.49411764705882,-0.023844985663890837,780,0.0003,-10.265326499938965,0.2,-282.75796045882356,1565.734375,,
```

#### logs_PPO_0.1_baseline/log.txt

```
Logging to ./logs_PPO_0.1_baseline
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 14       |
|    ep_rew_mean     | -80.9    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 425      |
|    total_timesteps | 64       |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 14          |
|    ep_rew_mean          | -80.9       |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 2           |
|    time_elapsed         | 890         |
|    total_timesteps      | 128         |
| train/                  |             |
|    approx_kl            | 0.012552523 |
|    clip_fraction        | 0.0359      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.4       |
|    explained_variance   | 0.00164     |
|    learning_rate        | 0.0003      |
|    loss                 | 628         |
|    n_updates            | 10          |
|    policy_gradient_loss | -0.0491     |
|    value_loss           | 1.3e+03     |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 14          |
|    ep_rew_mean          | -80.9       |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 3           |
|    time_elapsed         | 1373        |
|    total_timesteps      | 192         |
| train/                  |             |
|    approx_kl            | 0.014237673 |
|    clip_fraction        | 0.0703      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 2.38e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 721         |
|    n_updates            | 20          |
|    policy_gradient_loss | -0.0518     |
|    value_loss           | 1.46e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 35.3         |
|    ep_rew_mean          | -172         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 4            |
|    time_elapsed         | 1769         |
|    total_timesteps      | 256          |
| train/                  |              |
|    approx_kl            | 0.0065382356 |
|    clip_fraction        | 0.00625      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 692          |
|    n_updates            | 30           |
|    policy_gradient_loss | -0.0328      |
|    value_loss           | 1.4e+03      |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 45.5        |
|    ep_rew_mean          | -226        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 5           |
|    time_elapsed         | 2167        |
|    total_timesteps      | 320         |
| train/                  |             |
|    approx_kl            | 0.011088308 |
|    clip_fraction        | 0.0469      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 486         |
|    n_updates            | 40          |
|    policy_gradient_loss | -0.0497     |
|    value_loss           | 983         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 53          |
|    ep_rew_mean          | -263        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 6           |
|    time_elapsed         | 2605        |
|    total_timesteps      | 384         |
| train/                  |             |
|    approx_kl            | 0.008641969 |
|    clip_fraction        | 0.0344      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.3       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 541         |
|    n_updates            | 50          |
|    policy_gradient_loss | -0.0352     |
|    value_loss           | 1.09e+03    |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 47.7         |
|    ep_rew_mean          | -234         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 7            |
|    time_elapsed         | 2989         |
|    total_timesteps      | 448          |
| train/                  |              |
|    approx_kl            | 0.0047762934 |
|    clip_fraction        | 0.00313      |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 0            |
|    learning_rate        | 0.0003       |
|    loss                 | 1.06e+03     |
|    n_updates            | 60           |
|    policy_gradient_loss | -0.0296      |
|    value_loss           | 2.13e+03     |
------------------------------------------
------------------------------------------
| eval/                   |              |
|    mean_ep_length       | 134          |
|    mean_reward          | -184         |
| time/                   |              |
|    total_timesteps      | 512          |
| train/                  |              |
|    approx_kl            | 0.0077922577 |
|    clip_fraction        | 0.0172       |
|    clip_range           | 0.2          |
|    entropy_loss         | -10.3        |
|    explained_variance   | 1.19e-07     |
|    learning_rate        | 0.0003       |
|    loss                 | 524          |
|    n_updates            | 70           |
|    policy_gradient_loss | -0.0383      |
|    value_loss           | 1.06e+03     |
------------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 47.7     |
|    ep_rew_mean     | -234     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6155     |
|    total_timesteps | 512      |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 54.3       |
|    ep_rew_mean          | -266       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 9          |
|    time_elapsed         | 6551       |
|    total_timesteps      | 576        |
| train/                  |            |
|    approx_kl            | 0.00980701 |
|    clip_fraction        | 0.0328     |
|    clip_range           | 0.2        |
|    entropy_loss         | -10.3      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 580        |
|    n_updates            | 80         |
|    policy_gradient_loss | -0.0436    |
|    value_loss           | 1.17e+03   |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 54.3        |
|    ep_rew_mean          | -266        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 10          |
|    time_elapsed         | 6964        |
|    total_timesteps      | 640         |
| train/                  |             |
|    approx_kl            | 0.013457211 |
|    clip_fraction        | 0.0609      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 359         |
|    n_updates            | 90          |
|    policy_gradient_loss | -0.0536     |
|    value_loss           | 725         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.7        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 11          |
|    time_elapsed         | 7320        |
|    total_timesteps      | 704         |
| train/                  |             |
|    approx_kl            | 0.009044629 |
|    clip_fraction        | 0.0234      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 721         |
|    n_updates            | 100         |
|    policy_gradient_loss | -0.0347     |
|    value_loss           | 1.45e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.7        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 12          |
|    time_elapsed         | 7696        |
|    total_timesteps      | 768         |
| train/                  |             |
|    approx_kl            | 0.017904818 |
|    clip_fraction        | 0.0938      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.2       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 406         |
|    n_updates            | 110         |
|    policy_gradient_loss | -0.0558     |
|    value_loss           | 819         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 56          |
|    ep_rew_mean          | -268        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 13          |
|    time_elapsed         | 8058        |
|    total_timesteps      | 832         |
| train/                  |             |
|    approx_kl            | 0.008132456 |
|    clip_fraction        | 0.0266      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 623         |
|    n_updates            | 120         |
|    policy_gradient_loss | -0.036      |
|    value_loss           | 1.26e+03    |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 57.7        |
|    ep_rew_mean          | -276        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 14          |
|    time_elapsed         | 8412        |
|    total_timesteps      | 896         |
| train/                  |             |
|    approx_kl            | 0.012231968 |
|    clip_fraction        | 0.0469      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 433         |
|    n_updates            | 130         |
|    policy_gradient_loss | -0.0469     |
|    value_loss           | 873         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 60.2        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 15          |
|    time_elapsed         | 8731        |
|    total_timesteps      | 960         |
| train/                  |             |
|    approx_kl            | 0.012407127 |
|    clip_fraction        | 0.0672      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10.1       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 342         |
|    n_updates            | 140         |
|    policy_gradient_loss | -0.0468     |
|    value_loss           | 691         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 134         |
|    mean_reward          | -116        |
| time/                   |             |
|    total_timesteps      | 1024        |
| train/                  |             |
|    approx_kl            | 0.009310201 |
|    clip_fraction        | 0.0281      |
|    clip_range           | 0.2         |
|    entropy_loss         | -10         |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 403         |
|    n_updates            | 150         |
|    policy_gradient_loss | -0.0399     |
|    value_loss           | 814         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 60.2     |
|    ep_rew_mean     | -287     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 11854    |
|    total_timesteps | 1024     |
---------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 60.2       |
|    ep_rew_mean          | -287       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 17         |
|    time_elapsed         | 12194      |
|    total_timesteps      | 1088       |
| train/                  |            |
|    approx_kl            | 0.00579891 |
|    clip_fraction        | 0.0141     |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.97      |
|    explained_variance   | 5.96e-08   |
|    learning_rate        | 0.0003     |
|    loss                 | 524        |
|    n_updates            | 160        |
|    policy_gradient_loss | -0.0321    |
|    value_loss           | 1.06e+03   |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 62.7        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 18          |
|    time_elapsed         | 12500       |
|    total_timesteps      | 1152        |
| train/                  |             |
|    approx_kl            | 0.014871908 |
|    clip_fraction        | 0.0578      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.95       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 384         |
|    n_updates            | 170         |
|    policy_gradient_loss | -0.0536     |
|    value_loss           | 774         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 62.7        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 19          |
|    time_elapsed         | 12839       |
|    total_timesteps      | 1216        |
| train/                  |             |
|    approx_kl            | 0.015464095 |
|    clip_fraction        | 0.0906      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.88       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 349         |
|    n_updates            | 180         |
|    policy_gradient_loss | -0.0506     |
|    value_loss           | 705         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 62.5        |
|    ep_rew_mean          | -287        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 20          |
|    time_elapsed         | 13202       |
|    total_timesteps      | 1280        |
| train/                  |             |
|    approx_kl            | 0.011157144 |
|    clip_fraction        | 0.0516      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.86       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 388         |
|    n_updates            | 190         |
|    policy_gradient_loss | -0.042      |
|    value_loss           | 783         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 65.1        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 21          |
|    time_elapsed         | 13502       |
|    total_timesteps      | 1344        |
| train/                  |             |
|    approx_kl            | 0.009286194 |
|    clip_fraction        | 0.0297      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.85       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 459         |
|    n_updates            | 200         |
|    policy_gradient_loss | -0.0376     |
|    value_loss           | 925         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 64.8        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 22          |
|    time_elapsed         | 13816       |
|    total_timesteps      | 1408        |
| train/                  |             |
|    approx_kl            | 0.022929654 |
|    clip_fraction        | 0.106       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.88       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 262         |
|    n_updates            | 210         |
|    policy_gradient_loss | -0.0562     |
|    value_loss           | 529         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 64.8        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 23          |
|    time_elapsed         | 14131       |
|    total_timesteps      | 1472        |
| train/                  |             |
|    approx_kl            | 0.014848035 |
|    clip_fraction        | 0.0719      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.91       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 322         |
|    n_updates            | 220         |
|    policy_gradient_loss | -0.0474     |
|    value_loss           | 650         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 134         |
|    mean_reward          | -296        |
| time/                   |             |
|    total_timesteps      | 1536        |
| train/                  |             |
|    approx_kl            | 0.010459919 |
|    clip_fraction        | 0.0359      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.88       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 371         |
|    n_updates            | 230         |
|    policy_gradient_loss | -0.038      |
|    value_loss           | 749         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 64.3     |
|    ep_rew_mean     | -285     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 17156    |
|    total_timesteps | 1536     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 64.3        |
|    ep_rew_mean          | -285        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 25          |
|    time_elapsed         | 17423       |
|    total_timesteps      | 1600        |
| train/                  |             |
|    approx_kl            | 0.011442014 |
|    clip_fraction        | 0.0437      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.85       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 199         |
|    n_updates            | 240         |
|    policy_gradient_loss | -0.0473     |
|    value_loss           | 403         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 64.3        |
|    ep_rew_mean          | -285        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 26          |
|    time_elapsed         | 17743       |
|    total_timesteps      | 1664        |
| train/                  |             |
|    approx_kl            | 0.022848142 |
|    clip_fraction        | 0.116       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.85       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 216         |
|    n_updates            | 250         |
|    policy_gradient_loss | -0.058      |
|    value_loss           | 437         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 68.5        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 27          |
|    time_elapsed         | 18046       |
|    total_timesteps      | 1728        |
| train/                  |             |
|    approx_kl            | 0.011826382 |
|    clip_fraction        | 0.0437      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.85       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 352         |
|    n_updates            | 260         |
|    policy_gradient_loss | -0.0379     |
|    value_loss           | 711         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 66.4        |
|    ep_rew_mean          | -282        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 28          |
|    time_elapsed         | 18353       |
|    total_timesteps      | 1792        |
| train/                  |             |
|    approx_kl            | 0.015094347 |
|    clip_fraction        | 0.0656      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.82       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 218         |
|    n_updates            | 270         |
|    policy_gradient_loss | -0.0465     |
|    value_loss           | 442         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 66.4        |
|    ep_rew_mean          | -282        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 29          |
|    time_elapsed         | 18695       |
|    total_timesteps      | 1856        |
| train/                  |             |
|    approx_kl            | 0.013838103 |
|    clip_fraction        | 0.0625      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.8        |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 251         |
|    n_updates            | 280         |
|    policy_gradient_loss | -0.0453     |
|    value_loss           | 508         |
-----------------------------------------
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 68.5         |
|    ep_rew_mean          | -287         |
| time/                   |              |
|    fps                  | 0            |
|    iterations           | 30           |
|    time_elapsed         | 18978        |
|    total_timesteps      | 1920         |
| train/                  |              |
|    approx_kl            | 0.0143117765 |
|    clip_fraction        | 0.0563       |
|    clip_range           | 0.2          |
|    entropy_loss         | -9.79        |
|    explained_variance   | 5.96e-08     |
|    learning_rate        | 0.0003       |
|    loss                 | 425          |
|    n_updates            | 290          |
|    policy_gradient_loss | -0.0481      |
|    value_loss           | 856          |
------------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 70.3        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 31          |
|    time_elapsed         | 19273       |
|    total_timesteps      | 1984        |
| train/                  |             |
|    approx_kl            | 0.018789183 |
|    clip_fraction        | 0.0906      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.76       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 230         |
|    n_updates            | 300         |
|    policy_gradient_loss | -0.0565     |
|    value_loss           | 465         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 134         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 2048        |
| train/                  |             |
|    approx_kl            | 0.011357879 |
|    clip_fraction        | 0.05        |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.72       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 299         |
|    n_updates            | 310         |
|    policy_gradient_loss | -0.0373     |
|    value_loss           | 604         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 70.3     |
|    ep_rew_mean     | -290     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 22247    |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 70.3        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 33          |
|    time_elapsed         | 22500       |
|    total_timesteps      | 2112        |
| train/                  |             |
|    approx_kl            | 0.011729382 |
|    clip_fraction        | 0.0437      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.7        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 239         |
|    n_updates            | 320         |
|    policy_gradient_loss | -0.0432     |
|    value_loss           | 485         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 72.1        |
|    ep_rew_mean          | -293        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 34          |
|    time_elapsed         | 22782       |
|    total_timesteps      | 2176        |
| train/                  |             |
|    approx_kl            | 0.011633984 |
|    clip_fraction        | 0.0437      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.7        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 240         |
|    n_updates            | 330         |
|    policy_gradient_loss | -0.0397     |
|    value_loss           | 486         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 73.6        |
|    ep_rew_mean          | -299        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 35          |
|    time_elapsed         | 23081       |
|    total_timesteps      | 2240        |
| train/                  |             |
|    approx_kl            | 0.013890113 |
|    clip_fraction        | 0.0563      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.72       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 266         |
|    n_updates            | 340         |
|    policy_gradient_loss | -0.041      |
|    value_loss           | 537         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 73.6        |
|    ep_rew_mean          | -299        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 36          |
|    time_elapsed         | 23348       |
|    total_timesteps      | 2304        |
| train/                  |             |
|    approx_kl            | 0.009619005 |
|    clip_fraction        | 0.0297      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.73       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 330         |
|    n_updates            | 350         |
|    policy_gradient_loss | -0.04       |
|    value_loss           | 666         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 73.6        |
|    ep_rew_mean          | -299        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 37          |
|    time_elapsed         | 23634       |
|    total_timesteps      | 2368        |
| train/                  |             |
|    approx_kl            | 0.016308736 |
|    clip_fraction        | 0.0734      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.7        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 226         |
|    n_updates            | 360         |
|    policy_gradient_loss | -0.051      |
|    value_loss           | 458         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 74.7        |
|    ep_rew_mean          | -294        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 38          |
|    time_elapsed         | 23876       |
|    total_timesteps      | 2432        |
| train/                  |             |
|    approx_kl            | 0.018900804 |
|    clip_fraction        | 0.0922      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.67       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 212         |
|    n_updates            | 370         |
|    policy_gradient_loss | -0.0531     |
|    value_loss           | 429         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 76.4        |
|    ep_rew_mean          | -297        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 39          |
|    time_elapsed         | 24126       |
|    total_timesteps      | 2496        |
| train/                  |             |
|    approx_kl            | 0.014857366 |
|    clip_fraction        | 0.0672      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.7        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 87.9        |
|    n_updates            | 380         |
|    policy_gradient_loss | -0.0536     |
|    value_loss           | 180         |
-----------------------------------------
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 400        |
|    mean_reward          | -182       |
| time/                   |            |
|    total_timesteps      | 2560       |
| train/                  |            |
|    approx_kl            | 0.01839764 |
|    clip_fraction        | 0.0766     |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.72      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 151        |
|    n_updates            | 390        |
|    policy_gradient_loss | -0.0541    |
|    value_loss           | 308        |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 76.4     |
|    ep_rew_mean     | -297     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 27209    |
|    total_timesteps | 2560     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 75.6        |
|    ep_rew_mean          | -292        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 41          |
|    time_elapsed         | 27480       |
|    total_timesteps      | 2624        |
| train/                  |             |
|    approx_kl            | 0.017120507 |
|    clip_fraction        | 0.0922      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.75       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 239         |
|    n_updates            | 400         |
|    policy_gradient_loss | -0.0484     |
|    value_loss           | 483         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 75.6        |
|    ep_rew_mean          | -292        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 42          |
|    time_elapsed         | 27765       |
|    total_timesteps      | 2688        |
| train/                  |             |
|    approx_kl            | 0.013009103 |
|    clip_fraction        | 0.0531      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.75       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 258         |
|    n_updates            | 410         |
|    policy_gradient_loss | -0.0444     |
|    value_loss           | 522         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 77.2        |
|    ep_rew_mean          | -294        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 43          |
|    time_elapsed         | 28016       |
|    total_timesteps      | 2752        |
| train/                  |             |
|    approx_kl            | 0.014437213 |
|    clip_fraction        | 0.0688      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.7        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 287         |
|    n_updates            | 420         |
|    policy_gradient_loss | -0.0453     |
|    value_loss           | 579         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 76.2        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 44          |
|    time_elapsed         | 28295       |
|    total_timesteps      | 2816        |
| train/                  |             |
|    approx_kl            | 0.010962786 |
|    clip_fraction        | 0.0312      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.65       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 232         |
|    n_updates            | 430         |
|    policy_gradient_loss | -0.0453     |
|    value_loss           | 470         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 76.2        |
|    ep_rew_mean          | -290        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 45          |
|    time_elapsed         | 28554       |
|    total_timesteps      | 2880        |
| train/                  |             |
|    approx_kl            | 0.012972923 |
|    clip_fraction        | 0.0563      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.63       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 204         |
|    n_updates            | 440         |
|    policy_gradient_loss | -0.0409     |
|    value_loss           | 414         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 77.9        |
|    ep_rew_mean          | -292        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 46          |
|    time_elapsed         | 28788       |
|    total_timesteps      | 2944        |
| train/                  |             |
|    approx_kl            | 0.010065842 |
|    clip_fraction        | 0.0344      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.63       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 237         |
|    n_updates            | 450         |
|    policy_gradient_loss | -0.0393     |
|    value_loss           | 480         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79          |
|    ep_rew_mean          | -292        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 47          |
|    time_elapsed         | 29051       |
|    total_timesteps      | 3008        |
| train/                  |             |
|    approx_kl            | 0.018416584 |
|    clip_fraction        | 0.0797      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.61       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 202         |
|    n_updates            | 460         |
|    policy_gradient_loss | -0.0509     |
|    value_loss           | 409         |
-----------------------------------------
----------------------------------------
| eval/                   |            |
|    mean_ep_length       | 400        |
|    mean_reward          | -241       |
| time/                   |            |
|    total_timesteps      | 3072       |
| train/                  |            |
|    approx_kl            | 0.02356517 |
|    clip_fraction        | 0.106      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.62      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 100        |
|    n_updates            | 470        |
|    policy_gradient_loss | -0.0597    |
|    value_loss           | 204        |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 79       |
|    ep_rew_mean     | -292     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 32151    |
|    total_timesteps | 3072     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79          |
|    ep_rew_mean          | -292        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 49          |
|    time_elapsed         | 32402       |
|    total_timesteps      | 3136        |
| train/                  |             |
|    approx_kl            | 0.013335225 |
|    clip_fraction        | 0.0656      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.6        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 211         |
|    n_updates            | 480         |
|    policy_gradient_loss | -0.04       |
|    value_loss           | 428         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 78          |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 50          |
|    time_elapsed         | 32680       |
|    total_timesteps      | 3200        |
| train/                  |             |
|    approx_kl            | 0.010242311 |
|    clip_fraction        | 0.0359      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.58       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 236         |
|    n_updates            | 490         |
|    policy_gradient_loss | -0.0428     |
|    value_loss           | 478         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 78          |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 51          |
|    time_elapsed         | 32947       |
|    total_timesteps      | 3264        |
| train/                  |             |
|    approx_kl            | 0.008929431 |
|    clip_fraction        | 0.0375      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.54       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 235         |
|    n_updates            | 500         |
|    policy_gradient_loss | -0.0382     |
|    value_loss           | 476         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 78          |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 52          |
|    time_elapsed         | 33201       |
|    total_timesteps      | 3328        |
| train/                  |             |
|    approx_kl            | 0.009904224 |
|    clip_fraction        | 0.0437      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.53       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 273         |
|    n_updates            | 510         |
|    policy_gradient_loss | -0.037      |
|    value_loss           | 551         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79.8        |
|    ep_rew_mean          | -289        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 53          |
|    time_elapsed         | 33452       |
|    total_timesteps      | 3392        |
| train/                  |             |
|    approx_kl            | 0.014024525 |
|    clip_fraction        | 0.0641      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.5        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 163         |
|    n_updates            | 520         |
|    policy_gradient_loss | -0.0461     |
|    value_loss           | 331         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79.8        |
|    ep_rew_mean          | -289        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 54          |
|    time_elapsed         | 33679       |
|    total_timesteps      | 3456        |
| train/                  |             |
|    approx_kl            | 0.022488032 |
|    clip_fraction        | 0.117       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.5        |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 174         |
|    n_updates            | 530         |
|    policy_gradient_loss | -0.054      |
|    value_loss           | 353         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.3        |
|    ep_rew_mean          | -291        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 55          |
|    time_elapsed         | 33936       |
|    total_timesteps      | 3520        |
| train/                  |             |
|    approx_kl            | 0.023632366 |
|    clip_fraction        | 0.123       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.5        |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 98.3        |
|    n_updates            | 540         |
|    policy_gradient_loss | -0.0528     |
|    value_loss           | 201         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -184        |
| time/                   |             |
|    total_timesteps      | 3584        |
| train/                  |             |
|    approx_kl            | 0.024252271 |
|    clip_fraction        | 0.108       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.54       |
|    explained_variance   | 5.96e-08    |
|    learning_rate        | 0.0003      |
|    loss                 | 147         |
|    n_updates            | 550         |
|    policy_gradient_loss | -0.057      |
|    value_loss           | 297         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 82.3     |
|    ep_rew_mean     | -291     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 37019    |
|    total_timesteps | 3584     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 81.1        |
|    ep_rew_mean          | -286        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 57          |
|    time_elapsed         | 37321       |
|    total_timesteps      | 3648        |
| train/                  |             |
|    approx_kl            | 0.009553503 |
|    clip_fraction        | 0.0266      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.62       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 257         |
|    n_updates            | 560         |
|    policy_gradient_loss | -0.0309     |
|    value_loss           | 520         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.2        |
|    ep_rew_mean          | -288        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 58          |
|    time_elapsed         | 37583       |
|    total_timesteps      | 3712        |
| train/                  |             |
|    approx_kl            | 0.013557377 |
|    clip_fraction        | 0.0594      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.63       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 218         |
|    n_updates            | 570         |
|    policy_gradient_loss | -0.0433     |
|    value_loss           | 441         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 80.8        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 59          |
|    time_elapsed         | 37856       |
|    total_timesteps      | 3776        |
| train/                  |             |
|    approx_kl            | 0.014883036 |
|    clip_fraction        | 0.0688      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.63       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 172         |
|    n_updates            | 580         |
|    policy_gradient_loss | -0.0477     |
|    value_loss           | 349         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 80.8        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 60          |
|    time_elapsed         | 38139       |
|    total_timesteps      | 3840        |
| train/                  |             |
|    approx_kl            | 0.016481625 |
|    clip_fraction        | 0.0828      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.66       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 251         |
|    n_updates            | 590         |
|    policy_gradient_loss | -0.0458     |
|    value_loss           | 507         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 80.8        |
|    ep_rew_mean          | -283        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 61          |
|    time_elapsed         | 38424       |
|    total_timesteps      | 3904        |
| train/                  |             |
|    approx_kl            | 0.014165791 |
|    clip_fraction        | 0.0688      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.66       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 359         |
|    n_updates            | 600         |
|    policy_gradient_loss | -0.0418     |
|    value_loss           | 725         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.6        |
|    ep_rew_mean          | -284        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 62          |
|    time_elapsed         | 38731       |
|    total_timesteps      | 3968        |
| train/                  |             |
|    approx_kl            | 0.009812835 |
|    clip_fraction        | 0.025       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.64       |
|    explained_variance   | 1.79e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 192         |
|    n_updates            | 610         |
|    policy_gradient_loss | -0.038      |
|    value_loss           | 388         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 80.8        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 63          |
|    time_elapsed         | 38968       |
|    total_timesteps      | 4032        |
| train/                  |             |
|    approx_kl            | 0.016678378 |
|    clip_fraction        | 0.0766      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.61       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 215         |
|    n_updates            | 620         |
|    policy_gradient_loss | -0.0533     |
|    value_loss           | 435         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -182        |
| time/                   |             |
|    total_timesteps      | 4096        |
| train/                  |             |
|    approx_kl            | 0.015809234 |
|    clip_fraction        | 0.0734      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.58       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 164         |
|    n_updates            | 630         |
|    policy_gradient_loss | -0.0458     |
|    value_loss           | 333         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 80.8     |
|    ep_rew_mean     | -277     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 42054    |
|    total_timesteps | 4096     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79.8        |
|    ep_rew_mean          | -274        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 65          |
|    time_elapsed         | 42296       |
|    total_timesteps      | 4160        |
| train/                  |             |
|    approx_kl            | 0.014833016 |
|    clip_fraction        | 0.0656      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.56       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 249         |
|    n_updates            | 640         |
|    policy_gradient_loss | -0.0434     |
|    value_loss           | 504         |
-----------------------------------------
----------------------------------------
| rollout/                |            |
|    ep_len_mean          | 79.8       |
|    ep_rew_mean          | -274       |
| time/                   |            |
|    fps                  | 0          |
|    iterations           | 66         |
|    time_elapsed         | 42571      |
|    total_timesteps      | 4224       |
| train/                  |            |
|    approx_kl            | 0.02362166 |
|    clip_fraction        | 0.142      |
|    clip_range           | 0.2        |
|    entropy_loss         | -9.49      |
|    explained_variance   | 0          |
|    learning_rate        | 0.0003     |
|    loss                 | 213        |
|    n_updates            | 650        |
|    policy_gradient_loss | -0.057     |
|    value_loss           | 432        |
----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79.8        |
|    ep_rew_mean          | -274        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 67          |
|    time_elapsed         | 42811       |
|    total_timesteps      | 4288        |
| train/                  |             |
|    approx_kl            | 0.011060392 |
|    clip_fraction        | 0.0453      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.48       |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 235         |
|    n_updates            | 660         |
|    policy_gradient_loss | -0.0359     |
|    value_loss           | 476         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79.8        |
|    ep_rew_mean          | -274        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 68          |
|    time_elapsed         | 43072       |
|    total_timesteps      | 4352        |
| train/                  |             |
|    approx_kl            | 0.010925052 |
|    clip_fraction        | 0.0453      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.53       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 160         |
|    n_updates            | 670         |
|    policy_gradient_loss | -0.0439     |
|    value_loss           | 326         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 79.8        |
|    ep_rew_mean          | -274        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 69          |
|    time_elapsed         | 43336       |
|    total_timesteps      | 4416        |
| train/                  |             |
|    approx_kl            | 0.008436381 |
|    clip_fraction        | 0.025       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.52       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 193         |
|    n_updates            | 680         |
|    policy_gradient_loss | -0.0355     |
|    value_loss           | 390         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 81          |
|    ep_rew_mean          | -275        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 70          |
|    time_elapsed         | 43598       |
|    total_timesteps      | 4480        |
| train/                  |             |
|    approx_kl            | 0.023582386 |
|    clip_fraction        | 0.117       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.43       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 201         |
|    n_updates            | 690         |
|    policy_gradient_loss | -0.0582     |
|    value_loss           | 407         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.9        |
|    ep_rew_mean          | -279        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 71          |
|    time_elapsed         | 43888       |
|    total_timesteps      | 4544        |
| train/                  |             |
|    approx_kl            | 0.021217067 |
|    clip_fraction        | 0.113       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.37       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 178         |
|    n_updates            | 700         |
|    policy_gradient_loss | -0.0608     |
|    value_loss           | 361         |
-----------------------------------------
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 400         |
|    mean_reward          | -219        |
| time/                   |             |
|    total_timesteps      | 4608        |
| train/                  |             |
|    approx_kl            | 0.013921447 |
|    clip_fraction        | 0.0625      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.4        |
|    explained_variance   | 1.19e-07    |
|    learning_rate        | 0.0003      |
|    loss                 | 240         |
|    n_updates            | 710         |
|    policy_gradient_loss | -0.0363     |
|    value_loss           | 485         |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 82.9     |
|    ep_rew_mean     | -279     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 46926    |
|    total_timesteps | 4608     |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.9        |
|    ep_rew_mean          | -279        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 73          |
|    time_elapsed         | 47176       |
|    total_timesteps      | 4672        |
| train/                  |             |
|    approx_kl            | 0.007340722 |
|    clip_fraction        | 0.025       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.41       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 231         |
|    n_updates            | 720         |
|    policy_gradient_loss | -0.0354     |
|    value_loss           | 467         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.9        |
|    ep_rew_mean          | -279        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 74          |
|    time_elapsed         | 47419       |
|    total_timesteps      | 4736        |
| train/                  |             |
|    approx_kl            | 0.019293485 |
|    clip_fraction        | 0.0938      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.36       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 189         |
|    n_updates            | 730         |
|    policy_gradient_loss | -0.0598     |
|    value_loss           | 382         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.9        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 75          |
|    time_elapsed         | 47622       |
|    total_timesteps      | 4800        |
| train/                  |             |
|    approx_kl            | 0.013158467 |
|    clip_fraction        | 0.0484      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.29       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 172         |
|    n_updates            | 740         |
|    policy_gradient_loss | -0.0455     |
|    value_loss           | 348         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.9        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 76          |
|    time_elapsed         | 47895       |
|    total_timesteps      | 4864        |
| train/                  |             |
|    approx_kl            | 0.028335191 |
|    clip_fraction        | 0.172       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.22       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 142         |
|    n_updates            | 750         |
|    policy_gradient_loss | -0.0618     |
|    value_loss           | 287         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 82.9        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 77          |
|    time_elapsed         | 48070       |
|    total_timesteps      | 4928        |
| train/                  |             |
|    approx_kl            | 0.017344508 |
|    clip_fraction        | 0.102       |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.18       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 221         |
|    n_updates            | 760         |
|    policy_gradient_loss | -0.0426     |
|    value_loss           | 448         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 83.6        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 78          |
|    time_elapsed         | 48273       |
|    total_timesteps      | 4992        |
| train/                  |             |
|    approx_kl            | 0.012441653 |
|    clip_fraction        | 0.0547      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.27       |
|    explained_variance   | 0           |
|    learning_rate        | 0.0003      |
|    loss                 | 126         |
|    n_updates            | 770         |
|    policy_gradient_loss | -0.0416     |
|    value_loss           | 257         |
-----------------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 83.6        |
|    ep_rew_mean          | -277        |
| time/                   |             |
|    fps                  | 0           |
|    iterations           | 79          |
|    time_elapsed         | 48461       |
|    total_timesteps      | 5056        |
| train/                  |             |
|    approx_kl            | 0.016930362 |
|    clip_fraction        | 0.0766      |
|    clip_range           | 0.2         |
|    entropy_loss         | -9.32       |
|    explained_variance   | -1.19e-07   |
|    learning_rate        | 0.0003      |
|    loss                 | 131         |
|    n_updates            | 780         |
|    policy_gradient_loss | -0.0454     |
|    value_loss           | 267         |
-----------------------------------------
```

#### logs_PPO_0.1_baseline/progress.csv

```
time/iterations,time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,time/fps,time/time_elapsed,train/explained_variance,train/approx_kl,train/clip_fraction,train/loss,train/policy_gradient_loss,train/n_updates,train/learning_rate,train/entropy_loss,train/clip_range,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,64,14.0,-80.931749,0,425,,,,,,,,,,,,
2,128,14.0,-80.931749,0,890,0.0016412138938903809,0.012552523,0.0359375,627.545166015625,-0.049119981937110424,10,0.0003,-10.354959869384766,0.2,1296.2395385742188,,
3,192,14.0,-80.931749,0,1373,2.384185791015625e-07,0.014237673,0.0703125,720.7591552734375,-0.05183186791837215,20,0.0003,-10.342641735076905,0.2,1459.1286865234374,,
4,256,35.333333333333336,-172.41766700000002,0,1769,0.0,0.0065382356,0.00625,692.1764526367188,-0.03284749910235405,30,0.0003,-10.32564582824707,0.2,1399.5523681640625,,
5,320,45.5,-225.57673125000002,0,2167,0.0,0.011088308,0.046875,485.9396667480469,-0.049723210744559765,40,0.0003,-10.31291742324829,0.2,983.0759704589843,,
6,384,53.0,-262.80863220000003,0,2605,0.0,0.008641969,0.034375,541.3024291992188,-0.03516102079302073,50,0.0003,-10.300519561767578,0.2,1093.7272094726563,,
7,448,47.666666666666664,-234.4880201666667,0,2989,0.0,0.0047762934,0.003125,1056.9268798828125,-0.02961264718323946,60,0.0003,-10.27993564605713,0.2,2127.7888671875,,
,512,,,,,1.1920928955078125e-07,0.0077922577,0.0171875,523.6319580078125,-0.03825399521738291,70,0.0003,-10.270306396484376,0.2,1056.9500244140625,-183.63072720000002,134.0
8,512,47.666666666666664,-234.4880201666667,0,6155,,,,,,,,,,,,
9,576,54.333333333333336,-265.7745653333334,0,6551,0.0,0.00980701,0.0328125,579.60107421875,-0.04359602201730013,80,0.0003,-10.262164306640624,0.2,1168.9504150390626,,
10,640,54.333333333333336,-265.7745653333334,0,6964,-1.1920928955078125e-07,0.013457211,0.0609375,358.830322265625,-0.05360590592026711,90,0.0003,-10.23098373413086,0.2,725.2011535644531,,
11,704,57.7,-283.08462310000004,0,7320,-1.1920928955078125e-07,0.009044629,0.0234375,721.016357421875,-0.034691289998590946,100,0.0003,-10.195424556732178,0.2,1451.815625,,
12,768,57.7,-283.08462310000004,0,7696,0.0,0.017904818,0.09375,405.51214599609375,-0.055812977626919745,110,0.0003,-10.15788402557373,0.2,818.6698852539063,,
13,832,56.0,-268.0608108333334,0,8058,0.0,0.008132456,0.0265625,623.1353149414062,-0.03603706546127796,120,0.0003,-10.130534648895264,0.2,1255.1608520507812,,
14,896,57.69230769230769,-276.0923417692308,0,8412,0.0,0.012231968,0.046875,432.9107360839844,-0.04693916048854589,130,0.0003,-10.103424263000488,0.2,872.8664916992187,,
15,960,60.214285714285715,-286.5479905714286,0,8731,0.0,0.012407127,0.0671875,342.35650634765625,-0.04678118880838156,140,0.0003,-10.054755973815919,0.2,691.4053649902344,,
,1024,,,,,-1.1920928955078125e-07,0.009310201,0.028125,403.4467468261719,-0.03989046439528465,150,0.0003,-10.01145076751709,0.2,814.0053100585938,-116.35083119999999,134.0
16,1024,60.214285714285715,-286.5479905714286,0,11854,,,,,,,,,,,,
17,1088,60.214285714285715,-286.5479905714286,0,12194,5.960464477539063e-08,0.00579891,0.0140625,523.6986083984375,-0.03207609206438065,160,0.0003,-9.970413970947266,0.2,1055.2931396484375,,
18,1152,62.6875,-289.8401415625,0,12500,1.1920928955078125e-07,0.014871908,0.0578125,383.7554016113281,-0.05364432241767645,170,0.0003,-9.947218418121338,0.2,774.2820556640625,,
19,1216,62.6875,-289.8401415625,0,12839,0.0,0.015464095,0.090625,349.4842529296875,-0.05063926968723535,180,0.0003,-9.881262397766113,0.2,705.2978820800781,,
20,1280,62.5,-287.19690949999995,0,13202,-1.1920928955078125e-07,0.011157144,0.0515625,388.0343017578125,-0.0419872235506773,190,0.0003,-9.858598136901856,0.2,782.8974304199219,,
21,1344,65.05263157894737,-292.5053974210526,0,13502,0.0,0.009286194,0.0296875,459.1729431152344,-0.03755147494375706,200,0.0003,-9.853038883209228,0.2,925.1312255859375,,
22,1408,64.85,-289.53231105,0,13816,1.1920928955078125e-07,0.022929654,0.10625,261.6103515625,-0.056200099736452104,210,0.0003,-9.883120441436768,0.2,529.1935302734375,,
23,1472,64.85,-289.53231105,0,14131,0.0,0.014848035,0.071875,322.2337341308594,-0.04740738719701767,220,0.0003,-9.90728816986084,0.2,650.2090393066406,,
,1536,,,,,-1.1920928955078125e-07,0.010459919,0.0359375,371.0997009277344,-0.037965253740549085,230,0.0003,-9.876534175872802,0.2,748.9197021484375,-295.75913380000003,134.0
24,1536,64.28571428571429,-284.50313038095237,0,17156,,,,,,,,,,,,
25,1600,64.28571428571429,-284.50313038095237,0,17423,0.0,0.011442014,0.04375,198.69155883789062,-0.0473469378426671,240,0.0003,-9.852202987670898,0.2,402.68150329589844,,
26,1664,64.28571428571429,-284.50313038095237,0,17743,5.960464477539063e-08,0.022848142,0.115625,215.81007385253906,-0.057962853088974954,250,0.0003,-9.84992332458496,0.2,437.2499633789063,,
27,1728,68.52173913043478,-292.60837991304345,0,18046,0.0,0.011826382,0.04375,352.4669189453125,-0.0378749880939722,260,0.0003,-9.847576999664307,0.2,711.4200744628906,,
28,1792,66.375,-282.45552375000005,0,18353,0.0,0.015094347,0.065625,218.2051239013672,-0.046487811207771304,270,0.0003,-9.822467041015624,0.2,441.5942810058594,,
29,1856,66.375,-282.45552375000005,0,18695,5.960464477539063e-08,0.013838103,0.0625,251.1167449951172,-0.04526746291667223,280,0.0003,-9.803206062316894,0.2,507.6225250244141,,
30,1920,68.48,-286.62655956000003,0,18978,5.960464477539063e-08,0.0143117765,0.05625,424.50433349609375,-0.04809938240796328,290,0.0003,-9.790146541595458,0.2,855.8163391113281,,
31,1984,70.3076923076923,-289.7298111153846,0,19273,-1.1920928955078125e-07,0.018789183,0.090625,229.5792694091797,-0.05648132134228945,300,0.0003,-9.755115604400634,0.2,464.70754089355466,,
,2048,,,,,1.7881393432617188e-07,0.011357879,0.05,299.0532531738281,-0.03734900634735823,310,0.0003,-9.721208667755127,0.2,604.0715942382812,-184.08404099999998,134.0
32,2048,70.3076923076923,-289.7298111153846,0,22247,,,,,,,,,,,,
33,2112,70.3076923076923,-289.7298111153846,0,22500,0.0,0.011729382,0.04375,239.49827575683594,-0.04323175270110369,320,0.0003,-9.702918243408202,0.2,484.5974945068359,,
34,2176,72.14814814814815,-293.42895603703704,0,22782,0.0,0.011633984,0.04375,240.29759216308594,-0.039655090868473054,330,0.0003,-9.699145030975341,0.2,486.3175048828125,,
35,2240,73.64285714285714,-298.76583625000006,0,23081,0.0,0.013890113,0.05625,265.81109619140625,-0.04098499715328217,340,0.0003,-9.715316486358642,0.2,537.3714904785156,,
36,2304,73.64285714285714,-298.76583625000006,0,23348,5.960464477539063e-08,0.009619005,0.0296875,329.878662109375,-0.04002106282860041,350,0.0003,-9.734827423095703,0.2,665.6359130859375,,
37,2368,73.64285714285714,-298.76583625000006,0,23634,0.0,0.016308736,0.0734375,226.21347045898438,-0.050961440429091454,360,0.0003,-9.703580188751221,0.2,457.7948394775391,,
38,2432,74.73333333333333,-294.4706103333334,0,23876,0.0,0.018900804,0.0921875,211.8809356689453,-0.053064579237252475,370,0.0003,-9.671643161773682,0.2,429.45204162597656,,
39,2496,76.41935483870968,-297.36134045161293,0,24126,0.0,0.014857366,0.0671875,87.85997009277344,-0.05359007231891155,380,0.0003,-9.700585842132568,0.2,179.51960144042968,,
,2560,,,,,0.0,0.01839764,0.0765625,151.42047119140625,-0.054116389527916905,390,0.0003,-9.72072992324829,0.2,307.62598571777346,-181.9741884,400.0
40,2560,76.41935483870968,-297.36134045161293,0,27209,,,,,,,,,,,,
41,2624,75.5625,-292.0571683125,0,27480,1.1920928955078125e-07,0.017120507,0.0921875,238.86746215820312,-0.0483551986515522,400,0.0003,-9.746419429779053,0.2,483.47291564941406,,
42,2688,75.5625,-292.0571683125,0,27765,0.0,0.013009103,0.053125,258.1036682128906,-0.04442016826942563,410,0.0003,-9.753700733184814,0.2,521.5411193847656,,
43,2752,77.21212121212122,-294.2280076060606,0,28016,0.0,0.014437213,0.06875,286.7052307128906,-0.04526260457932949,420,0.0003,-9.701829433441162,0.2,579.261962890625,,
44,2816,76.20588235294117,-290.0700525294117,0,28295,0.0,0.010962786,0.03125,232.38595581054688,-0.04531225245445967,430,0.0003,-9.651302814483643,0.2,470.16957397460936,,
45,2880,76.20588235294117,-290.0700525294117,0,28554,0.0,0.012972923,0.05625,204.42559814453125,-0.04094082731753588,440,0.0003,-9.63396520614624,0.2,413.86280212402346,,
46,2944,77.91428571428571,-292.49970399999995,0,28788,1.1920928955078125e-07,0.010065842,0.034375,237.30364990234375,-0.039290782250463965,450,0.0003,-9.63214807510376,0.2,480.1730133056641,,
47,3008,78.97297297297297,-292.0514795945945,0,29051,-1.1920928955078125e-07,0.018416584,0.0796875,202.07630920410156,-0.050875288993120195,460,0.0003,-9.613435745239258,0.2,409.3677734375,,
,3072,,,,,0.0,0.02356517,0.10625,100.22802734375,-0.0596517969854176,470,0.0003,-9.622442436218261,0.2,204.3959747314453,-240.89289279999997,400.0
48,3072,78.97297297297297,-292.0514795945945,0,32151,,,,,,,,,,,,
49,3136,78.97297297297297,-292.0514795945945,0,32402,0.0,0.013335225,0.065625,211.34173583984375,-0.03995423000305891,480,0.0003,-9.601731777191162,0.2,428.05499572753905,,
50,3200,78.02631578947368,-287.8246884473684,0,32680,0.0,0.010242311,0.0359375,236.45291137695312,-0.042827412113547327,490,0.0003,-9.575149822235108,0.2,478.4297821044922,,
51,3264,78.02631578947368,-287.8246884473684,0,32947,0.0,0.008929431,0.0375,235.30093383789062,-0.03816067613661289,500,0.0003,-9.543386745452882,0.2,475.5574188232422,,
52,3328,78.02631578947368,-287.8246884473684,0,33201,1.1920928955078125e-07,0.009904224,0.04375,272.7679138183594,-0.03701415415853262,510,0.0003,-9.532285118103028,0.2,551.316845703125,,
53,3392,79.76923076923077,-288.979836,0,33452,0.0,0.014024525,0.0640625,162.90386962890625,-0.046116124466061593,520,0.0003,-9.5028639793396,0.2,330.7669921875,,
54,3456,79.76923076923077,-288.979836,0,33679,-1.1920928955078125e-07,0.022488032,0.1171875,174.1151123046875,-0.054020803049206735,530,0.0003,-9.496733951568604,0.2,352.9275665283203,,
55,3520,82.26829268292683,-290.87657246341456,0,33936,0.0,0.023632366,0.1234375,98.324462890625,-0.05282818470150232,540,0.0003,-9.49842357635498,0.2,200.93573913574218,,
,3584,,,,,5.960464477539063e-08,0.024252271,0.1078125,146.5943603515625,-0.05698062665760517,550,0.0003,-9.539766120910645,0.2,297.2725830078125,-183.75596479999996,400.0
56,3584,82.26829268292683,-290.87657246341456,0,37019,,,,,,,,,,,,
57,3648,81.11904761904762,-286.2475444047618,0,37321,0.0,0.009553503,0.0265625,256.99127197265625,-0.030883180722594262,560,0.0003,-9.61582155227661,0.2,519.7464599609375,,
58,3712,82.16279069767442,-288.13915951162784,0,37583,1.1920928955078125e-07,0.013557377,0.059375,217.73890686035156,-0.04329333771020174,570,0.0003,-9.628997802734375,0.2,440.71997985839846,,
59,3776,80.8409090909091,-283.0197024545454,0,37856,0.0,0.014883036,0.06875,172.21463012695312,-0.04766480531543493,580,0.0003,-9.63076400756836,0.2,349.3306427001953,,
60,3840,80.8409090909091,-283.0197024545454,0,38139,-1.1920928955078125e-07,0.016481625,0.0828125,250.89639282226562,-0.04579468984156847,590,0.0003,-9.662379169464112,0.2,506.906591796875,,
61,3904,80.8409090909091,-283.0197024545454,0,38424,-1.1920928955078125e-07,0.014165791,0.06875,359.2216491699219,-0.04183527352288365,600,0.0003,-9.66119031906128,0.2,724.7249694824219,,
62,3968,82.55319148936171,-284.057677787234,0,38731,1.7881393432617188e-07,0.009812835,0.025,191.64540100097656,-0.03796328976750374,610,0.0003,-9.637038898468017,0.2,388.4436401367187,,
63,4032,80.75510204081633,-277.4518136530612,0,38968,0.0,0.016678378,0.0765625,215.11187744140625,-0.05329837650060654,620,0.0003,-9.605067729949951,0.2,434.8207611083984,,
,4096,,,,,0.0,0.015809234,0.0734375,164.18045043945312,-0.045840376801788804,630,0.0003,-9.58061933517456,0.2,332.68121032714845,-182.49524079999998,400.0
64,4096,80.75510204081633,-277.4518136530612,0,42054,,,,,,,,,,,,
65,4160,79.76,-273.5265809,0,42296,-1.1920928955078125e-07,0.014833016,0.065625,249.2624053955078,-0.043377965316176416,640,0.0003,-9.555182838439942,0.2,503.9543853759766,,
66,4224,79.76,-273.5265809,0,42571,0.0,0.02362166,0.1421875,213.32345581054688,-0.05696405321359634,650,0.0003,-9.486999988555908,0.2,431.82036743164065,,
67,4288,79.76,-273.5265809,0,42811,1.1920928955078125e-07,0.011060392,0.0453125,235.0462646484375,-0.03587795495986938,660,0.0003,-9.481980323791504,0.2,475.7071105957031,,
68,4352,79.76,-273.5265809,0,43072,0.0,0.010925052,0.0453125,160.4119110107422,-0.0438777968287468,670,0.0003,-9.530216693878174,0.2,325.6804718017578,,
69,4416,79.76,-273.5265809,0,43336,0.0,0.008436381,0.025,192.6345672607422,-0.035475480183959004,680,0.0003,-9.51582612991333,0.2,390.3856262207031,,
70,4480,81.01923076923077,-274.7677898269231,0,43598,0.0,0.023582386,0.1171875,201.0261688232422,-0.058246466889977457,690,0.0003,-9.434142589569092,0.2,407.30310974121096,,
71,4544,82.88888888888889,-278.59938050000005,0,43888,0.0,0.021217067,0.1125,178.4319610595703,-0.06076903752982617,700,0.0003,-9.371031284332275,0.2,361.1342346191406,,
,4608,,,,,1.1920928955078125e-07,0.013921447,0.0625,239.96661376953125,-0.03626634888350964,710,0.0003,-9.401423358917237,0.2,484.79140319824216,-219.0963018,400.0
72,4608,82.88888888888889,-278.59938050000005,0,46926,,,,,,,,,,,,
73,4672,82.88888888888889,-278.59938050000005,0,47176,-1.1920928955078125e-07,0.007340722,0.025,231.023681640625,-0.03536526933312416,720,0.0003,-9.407366371154785,0.2,467.4460784912109,,
74,4736,82.88888888888889,-278.59938050000005,0,47419,0.0,0.019293485,0.09375,188.55319213867188,-0.059770349971950056,730,0.0003,-9.364792728424073,0.2,382.04433288574216,,
75,4800,82.85454545454546,-277.37433072727276,0,47622,0.0,0.013158467,0.0484375,171.68760681152344,-0.045491930656135084,740,0.0003,-9.294667053222657,0.2,348.34722290039065,,
76,4864,82.85454545454546,-277.37433072727276,0,47895,0.0,0.028335191,0.171875,141.75186157226562,-0.06182376872748137,750,0.0003,-9.215941619873046,0.2,287.2522796630859,,
77,4928,82.85454545454546,-277.37433072727276,0,48070,0.0,0.017344508,0.1015625,221.32028198242188,-0.04264892227947712,760,0.0003,-9.181961822509766,0.2,447.84568481445314,,
78,4992,83.55357142857143,-277.49831805357144,0,48273,0.0,0.012441653,0.0546875,126.00041198730469,-0.04158874116837978,770,0.0003,-9.2678053855896,0.2,256.67154693603516,,
79,5056,83.55357142857143,-277.49831805357144,0,48461,-1.1920928955078125e-07,0.016930362,0.0765625,131.1275177001953,-0.04543228503316641,780,0.0003,-9.319472980499267,0.2,266.54953918457034,,
```

#### logs_TRPO_0.001_baseline/log.txt

```
Logging to ./logs_TRPO_0.001_baseline
----------------------------
| time/              |     |
|    fps             | 0   |
|    iterations      | 1   |
|    time_elapsed    | 500 |
|    total_timesteps | 64  |
----------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 32       |
|    ep_rew_mean            | -131     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 2        |
|    time_elapsed           | 971      |
|    total_timesteps        | 128      |
| train/                    |          |
|    explained_variance     | -0.0162  |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000311 |
|    learning_rate          | 0.0003   |
|    n_updates              | 1        |
|    policy_objective       | 0.0743   |
|    value_loss             | 1.45e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 32       |
|    ep_rew_mean            | -131     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 3        |
|    time_elapsed           | 1479     |
|    total_timesteps        | 192      |
| train/                    |          |
|    explained_variance     | 0.000136 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000351 |
|    learning_rate          | 0.0003   |
|    n_updates              | 2        |
|    policy_objective       | 0.0752   |
|    value_loss             | 1.75e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 32       |
|    ep_rew_mean            | -131     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 4        |
|    time_elapsed           | 1985     |
|    total_timesteps        | 256      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000397 |
|    learning_rate          | 0.0003   |
|    n_updates              | 3        |
|    policy_objective       | 0.0686   |
|    value_loss             | 2.05e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 62.5     |
|    ep_rew_mean            | -367     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 5        |
|    time_elapsed           | 2436     |
|    total_timesteps        | 320      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000419 |
|    learning_rate          | 0.0003   |
|    n_updates              | 4        |
|    policy_objective       | 0.0686   |
|    value_loss             | 2.45e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 62.5     |
|    ep_rew_mean            | -367     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 6        |
|    time_elapsed           | 2885     |
|    total_timesteps        | 384      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000391 |
|    learning_rate          | 0.0003   |
|    n_updates              | 5        |
|    policy_objective       | 0.0736   |
|    value_loss             | 1.19e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 55.3      |
|    ep_rew_mean            | -322      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 7         |
|    time_elapsed           | 3368      |
|    total_timesteps        | 448       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000377  |
|    learning_rate          | 0.0003    |
|    n_updates              | 6         |
|    policy_objective       | 0.074     |
|    value_loss             | 1.6e+03   |
-----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 395       |
|    mean_reward            | -187      |
| time/                     |           |
|    total_timesteps        | 512       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000399  |
|    learning_rate          | 0.0003    |
|    n_updates              | 7         |
|    policy_objective       | 0.0717    |
|    value_loss             | 856       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 47.2     |
|    ep_rew_mean     | -267     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6690     |
|    total_timesteps | 512      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.5     |
|    ep_rew_mean            | -248     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 9        |
|    time_elapsed           | 7144     |
|    total_timesteps        | 576      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000396 |
|    learning_rate          | 0.0003   |
|    n_updates              | 8        |
|    policy_objective       | 0.079    |
|    value_loss             | 745      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 42.6     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 10       |
|    time_elapsed           | 7585     |
|    total_timesteps        | 640      |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000374 |
|    learning_rate          | 0.0003   |
|    n_updates              | 9        |
|    policy_objective       | 0.083    |
|    value_loss             | 1.01e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.1     |
|    ep_rew_mean            | -237     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 11       |
|    time_elapsed           | 8053     |
|    total_timesteps        | 704      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000416 |
|    learning_rate          | 0.0003   |
|    n_updates              | 10       |
|    policy_objective       | 0.0846   |
|    value_loss             | 1.1e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.1     |
|    ep_rew_mean            | -237     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 12       |
|    time_elapsed           | 8542     |
|    total_timesteps        | 768      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00041  |
|    learning_rate          | 0.0003   |
|    n_updates              | 11       |
|    policy_objective       | 0.0849   |
|    value_loss             | 1.06e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 42.5     |
|    ep_rew_mean            | -228     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 13       |
|    time_elapsed           | 8989     |
|    total_timesteps        | 832      |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000453 |
|    learning_rate          | 0.0003   |
|    n_updates              | 12       |
|    policy_objective       | 0.0793   |
|    value_loss             | 2.47e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 42.5      |
|    ep_rew_mean            | -228      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 14        |
|    time_elapsed           | 9473      |
|    total_timesteps        | 896       |
| train/                    |           |
|    explained_variance     | -2.38e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000473  |
|    learning_rate          | 0.0003    |
|    n_updates              | 13        |
|    policy_objective       | 0.0903    |
|    value_loss             | 1.21e+03  |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 41.6      |
|    ep_rew_mean            | -225      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 15        |
|    time_elapsed           | 9898      |
|    total_timesteps        | 960       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000393  |
|    learning_rate          | 0.0003    |
|    n_updates              | 14        |
|    policy_objective       | 0.0726    |
|    value_loss             | 1.5e+03   |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 155      |
|    mean_reward            | -188     |
| time/                     |          |
|    total_timesteps        | 1024     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000378 |
|    learning_rate          | 0.0003   |
|    n_updates              | 15       |
|    policy_objective       | 0.0757   |
|    value_loss             | 1.58e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 43.5     |
|    ep_rew_mean     | -235     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 13132    |
|    total_timesteps | 1024     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 45.1     |
|    ep_rew_mean            | -246     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 17       |
|    time_elapsed           | 13605    |
|    total_timesteps        | 1088     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000415 |
|    learning_rate          | 0.0003   |
|    n_updates              | 16       |
|    policy_objective       | 0.0769   |
|    value_loss             | 1.59e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 46.5     |
|    ep_rew_mean            | -255     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 18       |
|    time_elapsed           | 14080    |
|    total_timesteps        | 1152     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000471 |
|    learning_rate          | 0.0003   |
|    n_updates              | 17       |
|    policy_objective       | 0.0786   |
|    value_loss             | 2.11e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 46.5      |
|    ep_rew_mean            | -255      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 19        |
|    time_elapsed           | 14592     |
|    total_timesteps        | 1216      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000395  |
|    learning_rate          | 0.0003    |
|    n_updates              | 18        |
|    policy_objective       | 0.0798    |
|    value_loss             | 1.71e+03  |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 46.8      |
|    ep_rew_mean            | -256      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 20        |
|    time_elapsed           | 15096     |
|    total_timesteps        | 1280      |
| train/                    |           |
|    explained_variance     | -2.38e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000439  |
|    learning_rate          | 0.0003    |
|    n_updates              | 19        |
|    policy_objective       | 0.0878    |
|    value_loss             | 1.36e+03  |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 45.8      |
|    ep_rew_mean            | -249      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 21        |
|    time_elapsed           | 15585     |
|    total_timesteps        | 1344      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000433  |
|    learning_rate          | 0.0003    |
|    n_updates              | 20        |
|    policy_objective       | 0.0779    |
|    value_loss             | 2.19e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 46.6     |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 22       |
|    time_elapsed           | 16037    |
|    total_timesteps        | 1408     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000406 |
|    learning_rate          | 0.0003   |
|    n_updates              | 21       |
|    policy_objective       | 0.0822   |
|    value_loss             | 1.55e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 47.1     |
|    ep_rew_mean            | -261     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 23       |
|    time_elapsed           | 16468    |
|    total_timesteps        | 1472     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000414 |
|    learning_rate          | 0.0003   |
|    n_updates              | 22       |
|    policy_objective       | 0.0813   |
|    value_loss             | 1.57e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 399      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 1536     |
| train/                    |          |
|    explained_variance     | 1.79e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000484 |
|    learning_rate          | 0.0003   |
|    n_updates              | 23       |
|    policy_objective       | 0.0849   |
|    value_loss             | 1.34e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 47.7     |
|    ep_rew_mean     | -265     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 19771    |
|    total_timesteps | 1536     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 47.7     |
|    ep_rew_mean            | -265     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 25       |
|    time_elapsed           | 20252    |
|    total_timesteps        | 1600     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000408 |
|    learning_rate          | 0.0003   |
|    n_updates              | 24       |
|    policy_objective       | 0.0781   |
|    value_loss             | 1.84e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.6     |
|    ep_rew_mean            | -269     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 26       |
|    time_elapsed           | 20700    |
|    total_timesteps        | 1664     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000475 |
|    learning_rate          | 0.0003   |
|    n_updates              | 25       |
|    policy_objective       | 0.0805   |
|    value_loss             | 2.44e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 49.6     |
|    ep_rew_mean            | -277     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 27       |
|    time_elapsed           | 21191    |
|    total_timesteps        | 1728     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000483 |
|    learning_rate          | 0.0003   |
|    n_updates              | 26       |
|    policy_objective       | 0.08     |
|    value_loss             | 1.4e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.4     |
|    ep_rew_mean            | -283     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 28       |
|    time_elapsed           | 21659    |
|    total_timesteps        | 1792     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00046  |
|    learning_rate          | 0.0003   |
|    n_updates              | 27       |
|    policy_objective       | 0.083    |
|    value_loss             | 1.39e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 50.4      |
|    ep_rew_mean            | -283      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 29        |
|    time_elapsed           | 22114     |
|    total_timesteps        | 1856      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000463  |
|    learning_rate          | 0.0003    |
|    n_updates              | 28        |
|    policy_objective       | 0.0799    |
|    value_loss             | 2.06e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.3     |
|    ep_rew_mean            | -279     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 30       |
|    time_elapsed           | 22594    |
|    total_timesteps        | 1920     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000481 |
|    learning_rate          | 0.0003   |
|    n_updates              | 29       |
|    policy_objective       | 0.0814   |
|    value_loss             | 1.76e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.5     |
|    ep_rew_mean            | -281     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 31       |
|    time_elapsed           | 23050    |
|    total_timesteps        | 1984     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000463 |
|    learning_rate          | 0.0003   |
|    n_updates              | 30       |
|    policy_objective       | 0.0789   |
|    value_loss             | 828      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 2048     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000458 |
|    learning_rate          | 0.0003   |
|    n_updates              | 31       |
|    policy_objective       | 0.084    |
|    value_loss             | 1.35e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50.9     |
|    ep_rew_mean     | -282     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 26337    |
|    total_timesteps | 2048     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.9     |
|    ep_rew_mean            | -282     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 33       |
|    time_elapsed           | 26805    |
|    total_timesteps        | 2112     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000434 |
|    learning_rate          | 0.0003   |
|    n_updates              | 32       |
|    policy_objective       | 0.0778   |
|    value_loss             | 1.31e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 50.9      |
|    ep_rew_mean            | -282      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 34        |
|    time_elapsed           | 27252     |
|    total_timesteps        | 2176      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000436  |
|    learning_rate          | 0.0003    |
|    n_updates              | 33        |
|    policy_objective       | 0.0792    |
|    value_loss             | 1.51e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.5     |
|    ep_rew_mean            | -287     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 35       |
|    time_elapsed           | 27662    |
|    total_timesteps        | 2240     |
| train/                    |          |
|    explained_variance     | 1.79e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000516 |
|    learning_rate          | 0.0003   |
|    n_updates              | 34       |
|    policy_objective       | 0.0777   |
|    value_loss             | 1.63e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.2     |
|    ep_rew_mean            | -290     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 36       |
|    time_elapsed           | 28101    |
|    total_timesteps        | 2304     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000491 |
|    learning_rate          | 0.0003   |
|    n_updates              | 35       |
|    policy_objective       | 0.0787   |
|    value_loss             | 1.13e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.1     |
|    ep_rew_mean            | -288     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 37       |
|    time_elapsed           | 28532    |
|    total_timesteps        | 2368     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00048  |
|    learning_rate          | 0.0003   |
|    n_updates              | 36       |
|    policy_objective       | 0.0829   |
|    value_loss             | 1.2e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.2     |
|    ep_rew_mean            | -287     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 38       |
|    time_elapsed           | 29007    |
|    total_timesteps        | 2432     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000468 |
|    learning_rate          | 0.0003   |
|    n_updates              | 37       |
|    policy_objective       | 0.0807   |
|    value_loss             | 1.2e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.2     |
|    ep_rew_mean            | -287     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 39       |
|    time_elapsed           | 29502    |
|    total_timesteps        | 2496     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000485 |
|    learning_rate          | 0.0003   |
|    n_updates              | 38       |
|    policy_objective       | 0.0799   |
|    value_loss             | 1.56e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 201      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 2560     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00048  |
|    learning_rate          | 0.0003   |
|    n_updates              | 39       |
|    policy_objective       | 0.0779   |
|    value_loss             | 1.44e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 51.9     |
|    ep_rew_mean     | -285     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 32674    |
|    total_timesteps | 2560     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.3     |
|    ep_rew_mean            | -288     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 41       |
|    time_elapsed           | 33168    |
|    total_timesteps        | 2624     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000445 |
|    learning_rate          | 0.0003   |
|    n_updates              | 40       |
|    policy_objective       | 0.0798   |
|    value_loss             | 744      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.3     |
|    ep_rew_mean            | -288     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 42       |
|    time_elapsed           | 33637    |
|    total_timesteps        | 2688     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000541 |
|    learning_rate          | 0.0003   |
|    n_updates              | 41       |
|    policy_objective       | 0.0796   |
|    value_loss             | 2.03e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.4     |
|    ep_rew_mean            | -288     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 43       |
|    time_elapsed           | 34093    |
|    total_timesteps        | 2752     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000532 |
|    learning_rate          | 0.0003   |
|    n_updates              | 42       |
|    policy_objective       | 0.067    |
|    value_loss             | 1.79e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.5     |
|    ep_rew_mean            | -294     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 44       |
|    time_elapsed           | 34534    |
|    total_timesteps        | 2816     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000491 |
|    learning_rate          | 0.0003   |
|    n_updates              | 43       |
|    policy_objective       | 0.0876   |
|    value_loss             | 1.32e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 53.9      |
|    ep_rew_mean            | -297      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 45        |
|    time_elapsed           | 34994     |
|    total_timesteps        | 2880      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000536  |
|    learning_rate          | 0.0003    |
|    n_updates              | 44        |
|    policy_objective       | 0.0822    |
|    value_loss             | 1.47e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.9     |
|    ep_rew_mean            | -297     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 46       |
|    time_elapsed           | 35498    |
|    total_timesteps        | 2944     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000564 |
|    learning_rate          | 0.0003   |
|    n_updates              | 45       |
|    policy_objective       | 0.0805   |
|    value_loss             | 1.55e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.4     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 47       |
|    time_elapsed           | 36021    |
|    total_timesteps        | 3008     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000478 |
|    learning_rate          | 0.0003   |
|    n_updates              | 46       |
|    policy_objective       | 0.0785   |
|    value_loss             | 2.15e+03 |
----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 201       |
|    mean_reward            | -183      |
| time/                     |           |
|    total_timesteps        | 3072      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000459  |
|    learning_rate          | 0.0003    |
|    n_updates              | 47        |
|    policy_objective       | 0.0852    |
|    value_loss             | 2.12e+03  |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 54       |
|    ep_rew_mean     | -298     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 39247    |
|    total_timesteps | 3072     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.3     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 49       |
|    time_elapsed           | 39755    |
|    total_timesteps        | 3136     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000521 |
|    learning_rate          | 0.0003   |
|    n_updates              | 48       |
|    policy_objective       | 0.0827   |
|    value_loss             | 899      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.3     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 50       |
|    time_elapsed           | 40210    |
|    total_timesteps        | 3200     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000521 |
|    learning_rate          | 0.0003   |
|    n_updates              | 49       |
|    policy_objective       | 0.0803   |
|    value_loss             | 930      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.3     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 51       |
|    time_elapsed           | 40645    |
|    total_timesteps        | 3264     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000554 |
|    learning_rate          | 0.0003   |
|    n_updates              | 50       |
|    policy_objective       | 0.0827   |
|    value_loss             | 1.77e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.3     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 52       |
|    time_elapsed           | 41066    |
|    total_timesteps        | 3328     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000534 |
|    learning_rate          | 0.0003   |
|    n_updates              | 51       |
|    policy_objective       | 0.0713   |
|    value_loss             | 1.5e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.1     |
|    ep_rew_mean            | -299     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 53       |
|    time_elapsed           | 41490    |
|    total_timesteps        | 3392     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000551 |
|    learning_rate          | 0.0003   |
|    n_updates              | 52       |
|    policy_objective       | 0.0741   |
|    value_loss             | 1.56e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.9     |
|    ep_rew_mean            | -303     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 54       |
|    time_elapsed           | 41962    |
|    total_timesteps        | 3456     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00053  |
|    learning_rate          | 0.0003   |
|    n_updates              | 53       |
|    policy_objective       | 0.0786   |
|    value_loss             | 957      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.6      |
|    ep_rew_mean            | -301      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 55        |
|    time_elapsed           | 42429     |
|    total_timesteps        | 3520      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000484  |
|    learning_rate          | 0.0003    |
|    n_updates              | 54        |
|    policy_objective       | 0.0857    |
|    value_loss             | 1.22e+03  |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 205      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 3584     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00045  |
|    learning_rate          | 0.0003   |
|    n_updates              | 55       |
|    policy_objective       | 0.0838   |
|    value_loss             | 1.31e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 54.6     |
|    ep_rew_mean     | -301     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 45614    |
|    total_timesteps | 3584     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.4     |
|    ep_rew_mean            | -299     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 57       |
|    time_elapsed           | 46079    |
|    total_timesteps        | 3648     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000514 |
|    learning_rate          | 0.0003   |
|    n_updates              | 56       |
|    policy_objective       | 0.0742   |
|    value_loss             | 1.82e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.9     |
|    ep_rew_mean            | -303     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 58       |
|    time_elapsed           | 46574    |
|    total_timesteps        | 3712     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000457 |
|    learning_rate          | 0.0003   |
|    n_updates              | 57       |
|    policy_objective       | 0.0934   |
|    value_loss             | 1.69e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.4     |
|    ep_rew_mean            | -301     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 59       |
|    time_elapsed           | 47007    |
|    total_timesteps        | 3776     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000483 |
|    learning_rate          | 0.0003   |
|    n_updates              | 58       |
|    policy_objective       | 0.0821   |
|    value_loss             | 1.4e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.1     |
|    ep_rew_mean            | -298     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 60       |
|    time_elapsed           | 47484    |
|    total_timesteps        | 3840     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000463 |
|    learning_rate          | 0.0003   |
|    n_updates              | 59       |
|    policy_objective       | 0.0837   |
|    value_loss             | 927      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.1     |
|    ep_rew_mean            | -298     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 61       |
|    time_elapsed           | 47904    |
|    total_timesteps        | 3904     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000483 |
|    learning_rate          | 0.0003   |
|    n_updates              | 60       |
|    policy_objective       | 0.0834   |
|    value_loss             | 1.31e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.1      |
|    ep_rew_mean            | -298      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 62        |
|    time_elapsed           | 48358     |
|    total_timesteps        | 3968      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000609  |
|    learning_rate          | 0.0003    |
|    n_updates              | 61        |
|    policy_objective       | 0.0799    |
|    value_loss             | 1.82e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.4     |
|    ep_rew_mean            | -300     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 63       |
|    time_elapsed           | 48850    |
|    total_timesteps        | 4032     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000498 |
|    learning_rate          | 0.0003   |
|    n_updates              | 62       |
|    policy_objective       | 0.0798   |
|    value_loss             | 1.48e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 4096     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000473 |
|    learning_rate          | 0.0003   |
|    n_updates              | 63       |
|    policy_objective       | 0.0843   |
|    value_loss             | 1.15e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 53.9     |
|    ep_rew_mean     | -297     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 52125    |
|    total_timesteps | 4096     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.3     |
|    ep_rew_mean            | -298     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 65       |
|    time_elapsed           | 52596    |
|    total_timesteps        | 4160     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000485 |
|    learning_rate          | 0.0003   |
|    n_updates              | 64       |
|    policy_objective       | 0.0838   |
|    value_loss             | 986      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.7     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 66       |
|    time_elapsed           | 53059    |
|    total_timesteps        | 4224     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000455 |
|    learning_rate          | 0.0003   |
|    n_updates              | 65       |
|    policy_objective       | 0.0856   |
|    value_loss             | 761      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 53.7      |
|    ep_rew_mean            | -295      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 67        |
|    time_elapsed           | 53537     |
|    total_timesteps        | 4288      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000462  |
|    learning_rate          | 0.0003    |
|    n_updates              | 66        |
|    policy_objective       | 0.0863    |
|    value_loss             | 1.64e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.7     |
|    ep_rew_mean            | -295     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 68       |
|    time_elapsed           | 53951    |
|    total_timesteps        | 4352     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000489 |
|    learning_rate          | 0.0003   |
|    n_updates              | 67       |
|    policy_objective       | 0.0771   |
|    value_loss             | 1.91e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.1      |
|    ep_rew_mean            | -297      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 69        |
|    time_elapsed           | 54400     |
|    total_timesteps        | 4416      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000511  |
|    learning_rate          | 0.0003    |
|    n_updates              | 68        |
|    policy_objective       | 0.0801    |
|    value_loss             | 1.24e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.6     |
|    ep_rew_mean            | -300     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 70       |
|    time_elapsed           | 54859    |
|    total_timesteps        | 4480     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000498 |
|    learning_rate          | 0.0003   |
|    n_updates              | 69       |
|    policy_objective       | 0.0784   |
|    value_loss             | 1.14e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.6     |
|    ep_rew_mean            | -300     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 71       |
|    time_elapsed           | 55262    |
|    total_timesteps        | 4544     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000499 |
|    learning_rate          | 0.0003   |
|    n_updates              | 70       |
|    policy_objective       | 0.0879   |
|    value_loss             | 1.33e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 4608     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00048  |
|    learning_rate          | 0.0003   |
|    n_updates              | 71       |
|    policy_objective       | 0.0827   |
|    value_loss             | 1.17e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 54.1     |
|    ep_rew_mean     | -296     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 58506    |
|    total_timesteps | 4608     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.1     |
|    ep_rew_mean            | -296     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 73       |
|    time_elapsed           | 58999    |
|    total_timesteps        | 4672     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00049  |
|    learning_rate          | 0.0003   |
|    n_updates              | 72       |
|    policy_objective       | 0.0921   |
|    value_loss             | 848      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 53.7      |
|    ep_rew_mean            | -293      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 74        |
|    time_elapsed           | 59425     |
|    total_timesteps        | 4736      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000502  |
|    learning_rate          | 0.0003    |
|    n_updates              | 73        |
|    policy_objective       | 0.0851    |
|    value_loss             | 1.78e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.7     |
|    ep_rew_mean            | -293     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 75       |
|    time_elapsed           | 59865    |
|    total_timesteps        | 4800     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000453 |
|    learning_rate          | 0.0003   |
|    n_updates              | 74       |
|    policy_objective       | 0.0828   |
|    value_loss             | 645      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.4     |
|    ep_rew_mean            | -291     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 76       |
|    time_elapsed           | 60341    |
|    total_timesteps        | 4864     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0005   |
|    learning_rate          | 0.0003   |
|    n_updates              | 75       |
|    policy_objective       | 0.0919   |
|    value_loss             | 1.06e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 53.5      |
|    ep_rew_mean            | -291      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 77        |
|    time_elapsed           | 60771     |
|    total_timesteps        | 4928      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.000477  |
|    learning_rate          | 0.0003    |
|    n_updates              | 76        |
|    policy_objective       | 0.0907    |
|    value_loss             | 1.51e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.2     |
|    ep_rew_mean            | -290     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 78       |
|    time_elapsed           | 61147    |
|    total_timesteps        | 4992     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000509 |
|    learning_rate          | 0.0003   |
|    n_updates              | 77       |
|    policy_objective       | 0.0833   |
|    value_loss             | 1.07e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.5     |
|    ep_rew_mean            | -290     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 79       |
|    time_elapsed           | 61572    |
|    total_timesteps        | 5056     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.000433 |
|    learning_rate          | 0.0003   |
|    n_updates              | 78       |
|    policy_objective       | 0.0851   |
|    value_loss             | 1.16e+03 |
----------------------------------------
```

#### logs_TRPO_0.001_baseline/progress.csv

```
time/iterations,time/fps,time/total_timesteps,time/time_elapsed,rollout/ep_len_mean,train/is_line_search_success,train/explained_variance,train/n_updates,train/policy_objective,train/learning_rate,rollout/ep_rew_mean,train/kl_divergence_loss,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,0,64,500,,,,,,,,,,,
2,0,128,971,32.0,1.0,-0.016164541244506836,1,0.07427617162466049,0.0003,-131.291802,0.00031124416273087263,1448.3978149414063,,
3,0,192,1479,32.0,1.0,0.00013631582260131836,2,0.075151726603508,0.0003,-131.291802,0.00035110459430143237,1746.3820556640626,,
4,0,256,1985,32.0,1.0,0.0,3,0.06861859560012817,0.0003,-131.291802,0.0003965584037359804,2050.6980224609374,,
5,0,320,2436,62.5,1.0,0.0,4,0.06864295899868011,0.0003,-366.76165025,0.000418736832216382,2451.103515625,,
6,0,384,2885,62.5,1.0,0.0,5,0.07364393770694733,0.0003,-366.76165025,0.00039149686926975846,1185.5936279296875,,
7,0,448,3368,55.285714285714285,1.0,-1.1920928955078125e-07,6,0.07398258149623871,0.0003,-321.50065442857147,0.00037668077857233584,1601.7582397460938,,
,,512,,,1.0,-1.1920928955078125e-07,7,0.07168854773044586,0.0003,,0.0003994848229922354,855.8146423339844,-187.0237034,395.2
8,0,512,6690,47.22222222222222,,,,,,-267.2777071111111,,,,
9,0,576,7144,44.54545454545455,1.0,0.0,8,0.07902621477842331,0.0003,-247.99098672727277,0.00039574509719386697,744.6696350097657,,
10,0,640,7585,42.583333333333336,1.0,1.1920928955078125e-07,9,0.08302938938140869,0.0003,-233.72886958333336,0.0003736341022886336,1014.5242858886719,,
11,0,704,8053,44.07142857142857,1.0,0.0,10,0.08459264039993286,0.0003,-236.95644364285718,0.00041573692578822374,1096.0192749023438,,
12,0,768,8542,44.07142857142857,1.0,0.0,11,0.08493775129318237,0.0003,-236.95644364285718,0.0004103904066141695,1056.6981689453125,,
13,0,832,8989,42.5,1.0,5.960464477539063e-08,12,0.07934343814849854,0.0003,-228.03718016666667,0.00045279046753421426,2472.9794921875,,
14,0,896,9473,42.5,1.0,-2.384185791015625e-07,13,0.09028398245573044,0.0003,-228.03718016666667,0.00047257234109565616,1205.864404296875,,
15,0,960,9898,41.63157894736842,1.0,-1.1920928955078125e-07,14,0.07259120047092438,0.0003,-224.73717905263157,0.0003928618389181793,1503.54892578125,,
,,1024,,,1.0,0.0,15,0.0757291316986084,0.0003,,0.00037841530865989625,1583.6629638671875,-187.6252014,155.4
16,0,1024,13132,43.55,,,,,,-235.4140624,,,,
17,0,1088,13605,45.095238095238095,1.0,0.0,16,0.0769042819738388,0.0003,-246.22261000000003,0.0004146536812186241,1590.0814086914063,,
18,0,1152,14080,46.5,1.0,5.960464477539063e-08,17,0.07861895114183426,0.0003,-255.22263213636367,0.0004712150548584759,2107.902685546875,,
19,0,1216,14592,46.541666666666664,1.0,-1.1920928955078125e-07,18,0.07975687086582184,0.0003,-255.13746754166664,0.00039536593249067664,1708.5034423828124,,
20,0,1280,15096,46.8,1.0,-2.384185791015625e-07,19,0.08780943602323532,0.0003,-255.92383315999996,0.0004391579714138061,1363.4650390625,,
21,0,1344,15585,45.84615384615385,1.0,-1.1920928955078125e-07,20,0.07791236788034439,0.0003,-249.4553727692307,0.0004327538190409541,2187.915380859375,,
22,0,1408,16037,46.55555555555556,1.0,0.0,21,0.0821680948138237,0.0003,-257.1860483333333,0.0004059856873936951,1550.5422485351562,,
23,0,1472,16468,47.142857142857146,1.0,0.0,22,0.08134406805038452,0.0003,-261.0953384285714,0.0004144098493270576,1572.9318237304688,,
,,1536,,,1.0,1.7881393432617188e-07,23,0.084872767329216,0.0003,,0.0004844818322453648,1344.2047973632812,-183.70658680000003,398.8
24,0,1536,19771,47.7,,,,,,-264.9018014666666,,,,
25,0,1600,20252,47.7,1.0,0.0,24,0.07811588048934937,0.0003,-264.9018014666666,0.00040823849849402905,1842.6277709960937,,
26,0,1664,20700,48.58064516129032,1.0,0.0,25,0.08051176369190216,0.0003,-268.7946062580645,0.00047473679296672344,2443.3126220703125,,
27,0,1728,21191,49.60606060606061,1.0,0.0,26,0.0799601823091507,0.0003,-277.0878369696969,0.0004828079545404762,1403.4793823242187,,
28,0,1792,21659,50.38235294117647,1.0,0.0,27,0.08302906155586243,0.0003,-282.62868461764697,0.0004599433741532266,1388.7986938476563,,
29,0,1856,22114,50.38235294117647,1.0,-1.1920928955078125e-07,28,0.07987359166145325,0.0003,-282.62868461764697,0.00046290713362395763,2057.0543701171873,,
30,0,1920,22594,50.27777777777778,1.0,1.1920928955078125e-07,29,0.08142919093370438,0.0003,-278.8239807777777,0.0004807429213542491,1756.5073120117188,,
31,0,1984,23050,50.54054054054054,1.0,0.0,30,0.07888181507587433,0.0003,-280.7039427837837,0.00046321595436893404,827.7189514160157,,
,,2048,,,1.0,0.0,31,0.08398383110761642,0.0003,,0.00045806047273799777,1353.8101928710937,-183.21625640000002,400.0
32,0,2048,26337,50.92307692307692,,,,,,-282.458469051282,,,,
33,0,2112,26805,50.92307692307692,1.0,0.0,32,0.0778491348028183,0.0003,-282.458469051282,0.00043420371366664767,1313.52890625,,
34,0,2176,27252,50.92307692307692,1.0,-1.1920928955078125e-07,33,0.07924766838550568,0.0003,-282.458469051282,0.0004359258455224335,1505.1684692382812,,
35,0,2240,27662,51.55,1.0,1.7881393432617188e-07,34,0.07774608582258224,0.0003,-286.54860195000003,0.0005160616710782051,1628.9861694335937,,
36,0,2304,28101,52.166666666666664,1.0,0.0,35,0.07873044162988663,0.0003,-289.65975626190476,0.0004911072319373488,1127.4858032226562,,
37,0,2368,28532,52.09090909090909,1.0,1.1920928955078125e-07,36,0.08289843797683716,0.0003,-288.09614706818184,0.0004798591253347695,1200.29677734375,,
38,0,2432,29007,52.15555555555556,1.0,0.0,37,0.0806785300374031,0.0003,-287.34335851111115,0.0004683761508204043,1203.4230224609375,,
39,0,2496,29502,52.15555555555556,1.0,0.0,38,0.07987746596336365,0.0003,-287.34335851111115,0.00048537936527282,1555.5928955078125,,
,,2560,,,1.0,0.0,39,0.07790610194206238,0.0003,,0.0004803321207873523,1439.141748046875,-182.9138096,201.0
40,0,2560,32674,51.851063829787236,,,,,,-285.2176366170213,,,,
41,0,2624,33168,52.333333333333336,1.0,0.0,40,0.07981064915657043,0.0003,-287.9074455416667,0.0004448597610462457,744.1654296875,,
42,0,2688,33637,52.333333333333336,1.0,0.0,41,0.07959496974945068,0.0003,-287.9074455416667,0.0005409771110862494,2029.2415771484375,,
43,0,2752,34093,52.42857142857143,1.0,0.0,42,0.06697386503219604,0.0003,-288.2806142244898,0.0005315784364938736,1786.20634765625,,
44,0,2816,34534,53.509803921568626,1.0,5.960464477539063e-08,43,0.08758886158466339,0.0003,-294.15882125490197,0.0004906410467810929,1319.9898315429687,,
45,0,2880,34994,53.86538461538461,1.0,-1.1920928955078125e-07,44,0.08216432482004166,0.0003,-297.01967938461536,0.0005359019269235432,1467.3247314453124,,
46,0,2944,35498,53.86538461538461,1.0,5.960464477539063e-08,45,0.08045458793640137,0.0003,-297.01967938461536,0.0005639978917315602,1546.0219970703124,,
47,0,3008,36021,53.41509433962264,1.0,5.960464477539063e-08,46,0.07848306000232697,0.0003,-294.5422116415094,0.0004779747105203569,2146.728466796875,,
,,3072,,,1.0,-1.1920928955078125e-07,47,0.08517001569271088,0.0003,,0.00045942614087834954,2121.0770751953123,-183.4147948,201.0
48,0,3072,39247,53.96363636363636,,,,,,-298.39494450909086,,,,
49,0,3136,39755,53.275862068965516,1.0,0.0,48,0.08270436525344849,0.0003,-294.77440020689653,0.0005207619396969676,899.4836975097656,,
50,0,3200,40210,53.275862068965516,1.0,0.0,49,0.08025109767913818,0.0003,-294.77440020689653,0.0005205863271839917,929.7569274902344,,
51,0,3264,40645,53.275862068965516,1.0,5.960464477539063e-08,50,0.0826570987701416,0.0003,-294.77440020689653,0.0005543496226891875,1771.5116455078125,,
52,0,3328,41066,53.275862068965516,1.0,1.1920928955078125e-07,51,0.07129935920238495,0.0003,-294.77440020689653,0.0005342514486983418,1498.377880859375,,
53,0,3392,41490,54.083333333333336,1.0,0.0,52,0.07413551956415176,0.0003,-299.1316324833333,0.0005510657792910933,1561.351953125,,
54,0,3456,41962,54.903225806451616,1.0,5.960464477539063e-08,53,0.07860517501831055,0.0003,-302.7323184032258,0.000529800949152559,956.80419921875,,
55,0,3520,42429,54.58730158730159,1.0,-1.1920928955078125e-07,54,0.08568655699491501,0.0003,-301.1945364603174,0.00048367719864472747,1222.5070922851562,,
,,3584,,,1.0,5.960464477539063e-08,55,0.08379145711660385,0.0003,,0.0004498836351558566,1311.54892578125,-183.30150440000003,205.0
56,0,3584,45614,54.58730158730159,,,,,,-301.1945364603174,,,,
57,0,3648,46079,54.40625,1.0,1.1920928955078125e-07,56,0.0741700530052185,0.0003,-299.48149221875,0.0005136348772794008,1823.2013305664063,,
58,0,3712,46574,54.90909090909091,1.0,1.1920928955078125e-07,57,0.09335725009441376,0.0003,-303.458522030303,0.00045717397006228566,1690.0437255859374,,
59,0,3776,47007,54.44776119402985,1.0,0.0,58,0.08209254592657089,0.0003,-301.0266961492537,0.00048325335956178606,1395.0945556640625,,
60,0,3840,47484,54.08571428571429,1.0,1.1920928955078125e-07,59,0.08372488617897034,0.0003,-298.2215758142857,0.0004628427268471569,926.5442138671875,,
61,0,3904,47904,54.08571428571429,1.0,0.0,60,0.08344286680221558,0.0003,-298.2215758142857,0.0004826087970286608,1309.71416015625,,
62,0,3968,48358,54.08571428571429,1.0,-1.1920928955078125e-07,61,0.0799182876944542,0.0003,-298.2215758142857,0.0006093553965911269,1817.6067993164063,,
63,0,4032,48850,54.40277777777778,1.0,0.0,62,0.07975870370864868,0.0003,-300.03868276388886,0.0004978665383532643,1476.5268920898438,,
,,4096,,,1.0,0.0,63,0.08428939431905746,0.0003,,0.0004731995868496597,1154.373291015625,-182.84752199999997,400.0
64,0,4096,52125,53.9041095890411,,,,,,-296.933224109589,,,,
65,0,4160,52596,54.30263157894737,1.0,0.0,64,0.08375242352485657,0.0003,-298.3939765,0.00048543186858296394,985.5170043945312,,
66,0,4224,53059,53.688311688311686,1.0,0.0,65,0.08563316613435745,0.0003,-294.77112233766235,0.00045533356023952365,760.9236694335938,,
67,0,4288,53537,53.688311688311686,1.0,-1.1920928955078125e-07,66,0.08630955964326859,0.0003,-294.77112233766235,0.00046239251969382167,1644.2636596679688,,
68,0,4352,53951,53.688311688311686,1.0,0.0,67,0.07712352275848389,0.0003,-294.77112233766235,0.000488602090626955,1908.2941284179688,,
69,0,4416,54400,54.12658227848101,1.0,-1.1920928955078125e-07,68,0.08010859042406082,0.0003,-296.59237584810126,0.0005110871279612184,1235.5807861328126,,
70,0,4480,54859,54.617283950617285,1.0,5.960464477539063e-08,69,0.07843472808599472,0.0003,-300.0140244320987,0.0004975614137947559,1137.2821899414062,,
71,0,4544,55262,54.617283950617285,1.0,5.960464477539063e-08,70,0.08786509931087494,0.0003,-300.0140244320987,0.0004991743480786681,1330.2697021484375,,
,,4608,,,1.0,0.0,71,0.08272616565227509,0.0003,,0.0004799137241207063,1170.1659912109376,-183.07438380000002,400.0
72,0,4608,58506,54.13253012048193,,,,,,-296.2967192168675,,,,
73,0,4672,58999,54.13253012048193,1.0,0.0,72,0.09213798493146896,0.0003,-296.2967192168675,0.0004901762586086988,848.0029968261719,,
74,0,4736,59425,53.747126436781606,1.0,-1.1920928955078125e-07,73,0.08506886661052704,0.0003,-293.444942954023,0.0005017683142796159,1775.690625,,
75,0,4800,59865,53.69318181818182,1.0,0.0,74,0.08281835913658142,0.0003,-293.00542309090906,0.0004529263824224472,644.887548828125,,
76,0,4864,60341,53.41573033707865,1.0,0.0,75,0.09193868935108185,0.0003,-291.3334481123595,0.0005003145197406411,1062.9482788085938,,
77,0,4928,60771,53.46666666666667,1.0,-1.1920928955078125e-07,76,0.09069505333900452,0.0003,-291.2875530444444,0.0004771211533807218,1512.2759765625,,
78,0,4992,61147,53.24175824175824,1.0,0.0,77,0.08328163623809814,0.0003,-289.7365060989011,0.0005086169112473726,1067.5254760742187,,
79,0,5056,61572,53.483870967741936,1.0,0.0,78,0.08509628474712372,0.0003,-290.4684918064516,0.0004331041418481618,1161.5903930664062,,
```

#### logs_TRPO_0.01_baseline/log.txt

```
Logging to ./logs_TRPO_0.01_baseline
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 15       |
|    ep_rew_mean     | -77.5    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 485      |
|    total_timesteps | 64       |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 23.5     |
|    ep_rew_mean            | -109     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 2        |
|    time_elapsed           | 943      |
|    total_timesteps        | 128      |
| train/                    |          |
|    explained_variance     | -0.0249  |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00149  |
|    learning_rate          | 0.0003   |
|    n_updates              | 1        |
|    policy_objective       | 0.149    |
|    value_loss             | 1.09e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 18.3      |
|    ep_rew_mean            | -86.2     |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 3         |
|    time_elapsed           | 1414      |
|    total_timesteps        | 192       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00186   |
|    learning_rate          | 0.0003    |
|    n_updates              | 2         |
|    policy_objective       | 0.151     |
|    value_loss             | 1.68e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 18.3     |
|    ep_rew_mean            | -86.2    |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 4        |
|    time_elapsed           | 1884     |
|    total_timesteps        | 256      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00255  |
|    learning_rate          | 0.0003   |
|    n_updates              | 3        |
|    policy_objective       | 0.177    |
|    value_loss             | 1.24e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 42.6      |
|    ep_rew_mean            | -223      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 5         |
|    time_elapsed           | 2328      |
|    total_timesteps        | 320       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00275   |
|    learning_rate          | 0.0003    |
|    n_updates              | 4         |
|    policy_objective       | 0.182     |
|    value_loss             | 1.85e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.2     |
|    ep_rew_mean            | -250     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 6        |
|    time_elapsed           | 2813     |
|    total_timesteps        | 384      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00341  |
|    learning_rate          | 0.0003   |
|    n_updates              | 5        |
|    policy_objective       | 0.238    |
|    value_loss             | 1.4e+03  |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 46.5      |
|    ep_rew_mean            | -252      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 7         |
|    time_elapsed           | 3297      |
|    total_timesteps        | 448       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00316   |
|    learning_rate          | 0.0003    |
|    n_updates              | 6         |
|    policy_objective       | 0.189     |
|    value_loss             | 1.61e+03  |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 136      |
|    mean_reward            | -302     |
| time/                     |          |
|    total_timesteps        | 512      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00325  |
|    learning_rate          | 0.0003   |
|    n_updates              | 7        |
|    policy_objective       | 0.223    |
|    value_loss             | 1.57e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 46.5     |
|    ep_rew_mean     | -252     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6502     |
|    total_timesteps | 512      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 46.5     |
|    ep_rew_mean            | -252     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 9        |
|    time_elapsed           | 6983     |
|    total_timesteps        | 576      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00404  |
|    learning_rate          | 0.0003   |
|    n_updates              | 8        |
|    policy_objective       | 0.238    |
|    value_loss             | 1.81e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.2     |
|    ep_rew_mean            | -281     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 10       |
|    time_elapsed           | 7409     |
|    total_timesteps        | 640      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00367  |
|    learning_rate          | 0.0003   |
|    n_updates              | 9        |
|    policy_objective       | 0.219    |
|    value_loss             | 1.81e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.3     |
|    ep_rew_mean            | -264     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 11       |
|    time_elapsed           | 7914     |
|    total_timesteps        | 704      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0038   |
|    learning_rate          | 0.0003   |
|    n_updates              | 10       |
|    policy_objective       | 0.231    |
|    value_loss             | 1.42e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.5     |
|    ep_rew_mean            | -278     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 12       |
|    time_elapsed           | 8390     |
|    total_timesteps        | 768      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00349  |
|    learning_rate          | 0.0003   |
|    n_updates              | 11       |
|    policy_objective       | 0.225    |
|    value_loss             | 1.57e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.5     |
|    ep_rew_mean            | -267     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 13       |
|    time_elapsed           | 8807     |
|    total_timesteps        | 832      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00418  |
|    learning_rate          | 0.0003   |
|    n_updates              | 12       |
|    policy_objective       | 0.24     |
|    value_loss             | 1.74e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.5     |
|    ep_rew_mean            | -267     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 14       |
|    time_elapsed           | 9276     |
|    total_timesteps        | 896      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00444  |
|    learning_rate          | 0.0003   |
|    n_updates              | 13       |
|    policy_objective       | 0.28     |
|    value_loss             | 878      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.8     |
|    ep_rew_mean            | -274     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 15       |
|    time_elapsed           | 9696     |
|    total_timesteps        | 960      |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0039   |
|    learning_rate          | 0.0003   |
|    n_updates              | 14       |
|    policy_objective       | 0.238    |
|    value_loss             | 1.52e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 61       |
|    mean_reward            | -88      |
| time/                     |          |
|    total_timesteps        | 1024     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00449  |
|    learning_rate          | 0.0003   |
|    n_updates              | 15       |
|    policy_objective       | 0.282    |
|    value_loss             | 1.36e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50.8     |
|    ep_rew_mean     | -274     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 11366    |
|    total_timesteps | 1024     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.1     |
|    ep_rew_mean            | -282     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 17       |
|    time_elapsed           | 11839    |
|    total_timesteps        | 1088     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00432  |
|    learning_rate          | 0.0003   |
|    n_updates              | 16       |
|    policy_objective       | 0.267    |
|    value_loss             | 1.6e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.4     |
|    ep_rew_mean            | -291     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 18       |
|    time_elapsed           | 12279    |
|    total_timesteps        | 1152     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00495  |
|    learning_rate          | 0.0003   |
|    n_updates              | 17       |
|    policy_objective       | 0.258    |
|    value_loss             | 1.78e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.3     |
|    ep_rew_mean            | -284     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 19       |
|    time_elapsed           | 12717    |
|    total_timesteps        | 1216     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00402  |
|    learning_rate          | 0.0003   |
|    n_updates              | 18       |
|    policy_objective       | 0.266    |
|    value_loss             | 1.57e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 53.4     |
|    ep_rew_mean            | -289     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 20       |
|    time_elapsed           | 13164    |
|    total_timesteps        | 1280     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00427  |
|    learning_rate          | 0.0003   |
|    n_updates              | 19       |
|    policy_objective       | 0.3      |
|    value_loss             | 1.02e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 52.1     |
|    ep_rew_mean            | -281     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 21       |
|    time_elapsed           | 13640    |
|    total_timesteps        | 1344     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00412  |
|    learning_rate          | 0.0003   |
|    n_updates              | 20       |
|    policy_objective       | 0.276    |
|    value_loss             | 1.22e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 52.1      |
|    ep_rew_mean            | -281      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 22        |
|    time_elapsed           | 14123     |
|    total_timesteps        | 1408      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00441   |
|    learning_rate          | 0.0003    |
|    n_updates              | 21        |
|    policy_objective       | 0.292     |
|    value_loss             | 1.63e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.4     |
|    ep_rew_mean            | -294     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 23       |
|    time_elapsed           | 14661    |
|    total_timesteps        | 1472     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00445  |
|    learning_rate          | 0.0003   |
|    n_updates              | 22       |
|    policy_objective       | 0.26     |
|    value_loss             | 1.88e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 134      |
|    mean_reward            | -185     |
| time/                     |          |
|    total_timesteps        | 1536     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00466  |
|    learning_rate          | 0.0003   |
|    n_updates              | 23       |
|    policy_objective       | 0.259    |
|    value_loss             | 1.1e+03  |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 55       |
|    ep_rew_mean     | -296     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 17845    |
|    total_timesteps | 1536     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 55       |
|    ep_rew_mean            | -296     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 25       |
|    time_elapsed           | 18305    |
|    total_timesteps        | 1600     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00412  |
|    learning_rate          | 0.0003   |
|    n_updates              | 24       |
|    policy_objective       | 0.274    |
|    value_loss             | 1.5e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.1     |
|    ep_rew_mean            | -291     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 26       |
|    time_elapsed           | 18786    |
|    total_timesteps        | 1664     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00374  |
|    learning_rate          | 0.0003   |
|    n_updates              | 25       |
|    policy_objective       | 0.263    |
|    value_loss             | 1.91e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.8      |
|    ep_rew_mean            | -296      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 27        |
|    time_elapsed           | 19217     |
|    total_timesteps        | 1728      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00445   |
|    learning_rate          | 0.0003    |
|    n_updates              | 26        |
|    policy_objective       | 0.268     |
|    value_loss             | 1.51e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 55.4     |
|    ep_rew_mean            | -300     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 28       |
|    time_elapsed           | 19662    |
|    total_timesteps        | 1792     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00466  |
|    learning_rate          | 0.0003   |
|    n_updates              | 27       |
|    policy_objective       | 0.269    |
|    value_loss             | 1.09e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 56       |
|    ep_rew_mean            | -304     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 29       |
|    time_elapsed           | 20092    |
|    total_timesteps        | 1856     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0043   |
|    learning_rate          | 0.0003   |
|    n_updates              | 28       |
|    policy_objective       | 0.269    |
|    value_loss             | 1.85e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 56       |
|    ep_rew_mean            | -304     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 30       |
|    time_elapsed           | 20520    |
|    total_timesteps        | 1920     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00465  |
|    learning_rate          | 0.0003   |
|    n_updates              | 29       |
|    policy_objective       | 0.25     |
|    value_loss             | 1.65e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 56.7      |
|    ep_rew_mean            | -307      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 31        |
|    time_elapsed           | 20920     |
|    total_timesteps        | 1984      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00464   |
|    learning_rate          | 0.0003    |
|    n_updates              | 30        |
|    policy_objective       | 0.277     |
|    value_loss             | 1.1e+03   |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 135      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 2048     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00467  |
|    learning_rate          | 0.0003   |
|    n_updates              | 31       |
|    policy_objective       | 0.255    |
|    value_loss             | 726      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 56.7     |
|    ep_rew_mean     | -307     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 24023    |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.5      |
|    ep_rew_mean            | -314      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 33        |
|    time_elapsed           | 24383     |
|    total_timesteps        | 2112      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00467   |
|    learning_rate          | 0.0003    |
|    n_updates              | 32        |
|    policy_objective       | 0.283     |
|    value_loss             | 1.21e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 59       |
|    ep_rew_mean            | -315     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 34       |
|    time_elapsed           | 24769    |
|    total_timesteps        | 2176     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00437  |
|    learning_rate          | 0.0003   |
|    n_updates              | 33       |
|    policy_objective       | 0.26     |
|    value_loss             | 917      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 59       |
|    ep_rew_mean            | -315     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 35       |
|    time_elapsed           | 25248    |
|    total_timesteps        | 2240     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00498  |
|    learning_rate          | 0.0003   |
|    n_updates              | 34       |
|    policy_objective       | 0.272    |
|    value_loss             | 1.07e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.3      |
|    ep_rew_mean            | -310      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 36        |
|    time_elapsed           | 25657     |
|    total_timesteps        | 2304      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0044    |
|    learning_rate          | 0.0003    |
|    n_updates              | 35        |
|    policy_objective       | 0.282     |
|    value_loss             | 1.56e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 59.3     |
|    ep_rew_mean            | -311     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 37       |
|    time_elapsed           | 26063    |
|    total_timesteps        | 2368     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00484  |
|    learning_rate          | 0.0003   |
|    n_updates              | 36       |
|    policy_objective       | 0.311    |
|    value_loss             | 934      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.7      |
|    ep_rew_mean            | -313      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 38        |
|    time_elapsed           | 26544     |
|    total_timesteps        | 2432      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00442   |
|    learning_rate          | 0.0003    |
|    n_updates              | 37        |
|    policy_objective       | 0.273     |
|    value_loss             | 699       |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.7      |
|    ep_rew_mean            | -313      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 39        |
|    time_elapsed           | 27011     |
|    total_timesteps        | 2496      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0044    |
|    learning_rate          | 0.0003    |
|    n_updates              | 38        |
|    policy_objective       | 0.282     |
|    value_loss             | 1.51e+03  |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 398      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 2560     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0044   |
|    learning_rate          | 0.0003   |
|    n_updates              | 39       |
|    policy_objective       | 0.301    |
|    value_loss             | 1.73e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 60.1     |
|    ep_rew_mean     | -316     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 30195    |
|    total_timesteps | 2560     |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.3      |
|    ep_rew_mean            | -311      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 41        |
|    time_elapsed           | 30623     |
|    total_timesteps        | 2624      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00476   |
|    learning_rate          | 0.0003    |
|    n_updates              | 40        |
|    policy_objective       | 0.287     |
|    value_loss             | 959       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.7     |
|    ep_rew_mean            | -306     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 42       |
|    time_elapsed           | 31028    |
|    total_timesteps        | 2688     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00384  |
|    learning_rate          | 0.0003   |
|    n_updates              | 41       |
|    policy_objective       | 0.263    |
|    value_loss             | 801      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.8     |
|    ep_rew_mean            | -301     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 43       |
|    time_elapsed           | 31454    |
|    total_timesteps        | 2752     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00444  |
|    learning_rate          | 0.0003   |
|    n_updates              | 42       |
|    policy_objective       | 0.283    |
|    value_loss             | 864      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.6     |
|    ep_rew_mean            | -305     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 44       |
|    time_elapsed           | 31855    |
|    total_timesteps        | 2816     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00407  |
|    learning_rate          | 0.0003   |
|    n_updates              | 43       |
|    policy_objective       | 0.285    |
|    value_loss             | 1.16e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.1     |
|    ep_rew_mean            | -301     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 45       |
|    time_elapsed           | 32208    |
|    total_timesteps        | 2880     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00427  |
|    learning_rate          | 0.0003   |
|    n_updates              | 44       |
|    policy_objective       | 0.253    |
|    value_loss             | 989      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.1     |
|    ep_rew_mean            | -301     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 46       |
|    time_elapsed           | 32604    |
|    total_timesteps        | 2944     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00416  |
|    learning_rate          | 0.0003   |
|    n_updates              | 45       |
|    policy_objective       | 0.261    |
|    value_loss             | 598      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.1      |
|    ep_rew_mean            | -298      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 47        |
|    time_elapsed           | 32990     |
|    total_timesteps        | 3008      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00425   |
|    learning_rate          | 0.0003    |
|    n_updates              | 46        |
|    policy_objective       | 0.239     |
|    value_loss             | 962       |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 3072     |
| train/                    |          |
|    explained_variance     | 2.38e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00419  |
|    learning_rate          | 0.0003   |
|    n_updates              | 47       |
|    policy_objective       | 0.253    |
|    value_loss             | 503      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 58.4     |
|    ep_rew_mean     | -297     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 36183    |
|    total_timesteps | 3072     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58       |
|    ep_rew_mean            | -294     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 49       |
|    time_elapsed           | 36550    |
|    total_timesteps        | 3136     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0042   |
|    learning_rate          | 0.0003   |
|    n_updates              | 48       |
|    policy_objective       | 0.282    |
|    value_loss             | 673      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58       |
|    ep_rew_mean            | -294     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 50       |
|    time_elapsed           | 36947    |
|    total_timesteps        | 3200     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00422  |
|    learning_rate          | 0.0003   |
|    n_updates              | 49       |
|    policy_objective       | 0.289    |
|    value_loss             | 847      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58       |
|    ep_rew_mean            | -294     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 51       |
|    time_elapsed           | 37340    |
|    total_timesteps        | 3264     |
| train/                    |          |
|    explained_variance     | 1.79e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00447  |
|    learning_rate          | 0.0003   |
|    n_updates              | 50       |
|    policy_objective       | 0.282    |
|    value_loss             | 1.18e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.2      |
|    ep_rew_mean            | -297      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 52        |
|    time_elapsed           | 37762     |
|    total_timesteps        | 3328      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0048    |
|    learning_rate          | 0.0003    |
|    n_updates              | 51        |
|    policy_objective       | 0.282     |
|    value_loss             | 927       |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.2      |
|    ep_rew_mean            | -296      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 53        |
|    time_elapsed           | 38162     |
|    total_timesteps        | 3392      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0043    |
|    learning_rate          | 0.0003    |
|    n_updates              | 52        |
|    policy_objective       | 0.287     |
|    value_loss             | 1.05e+03  |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.8      |
|    ep_rew_mean            | -298      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 54        |
|    time_elapsed           | 38503     |
|    total_timesteps        | 3456      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00448   |
|    learning_rate          | 0.0003    |
|    n_updates              | 53        |
|    policy_objective       | 0.29      |
|    value_loss             | 857       |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.1      |
|    ep_rew_mean            | -294      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 55        |
|    time_elapsed           | 38905     |
|    total_timesteps        | 3520      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00437   |
|    learning_rate          | 0.0003    |
|    n_updates              | 54        |
|    policy_objective       | 0.294     |
|    value_loss             | 1.12e+03  |
-----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 400       |
|    mean_reward            | -183      |
| time/                     |           |
|    total_timesteps        | 3584      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00499   |
|    learning_rate          | 0.0003    |
|    n_updates              | 55        |
|    policy_objective       | 0.319     |
|    value_loss             | 981       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 58.6     |
|    ep_rew_mean     | -291     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 42059    |
|    total_timesteps | 3584     |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.7      |
|    ep_rew_mean            | -291      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 57        |
|    time_elapsed           | 42482     |
|    total_timesteps        | 3648      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00487   |
|    learning_rate          | 0.0003    |
|    n_updates              | 56        |
|    policy_objective       | 0.29      |
|    value_loss             | 828       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 59.2     |
|    ep_rew_mean            | -293     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 58       |
|    time_elapsed           | 42926    |
|    total_timesteps        | 3712     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00442  |
|    learning_rate          | 0.0003   |
|    n_updates              | 57       |
|    policy_objective       | 0.335    |
|    value_loss             | 793      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.9      |
|    ep_rew_mean            | -291      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 59        |
|    time_elapsed           | 43311     |
|    total_timesteps        | 3776      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00499   |
|    learning_rate          | 0.0003    |
|    n_updates              | 58        |
|    policy_objective       | 0.265     |
|    value_loss             | 1.15e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.9     |
|    ep_rew_mean            | -291     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 60       |
|    time_elapsed           | 43702    |
|    total_timesteps        | 3840     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00487  |
|    learning_rate          | 0.0003   |
|    n_updates              | 59       |
|    policy_objective       | 0.305    |
|    value_loss             | 440      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 59.3     |
|    ep_rew_mean            | -291     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 61       |
|    time_elapsed           | 44058    |
|    total_timesteps        | 3904     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00503  |
|    learning_rate          | 0.0003   |
|    n_updates              | 60       |
|    policy_objective       | 0.278    |
|    value_loss             | 981      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.4     |
|    ep_rew_mean            | -285     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 62       |
|    time_elapsed           | 44464    |
|    total_timesteps        | 3968     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00468  |
|    learning_rate          | 0.0003   |
|    n_updates              | 61       |
|    policy_objective       | 0.282    |
|    value_loss             | 617      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.7     |
|    ep_rew_mean            | -281     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 63       |
|    time_elapsed           | 44918    |
|    total_timesteps        | 4032     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00461  |
|    learning_rate          | 0.0003   |
|    n_updates              | 62       |
|    policy_objective       | 0.295    |
|    value_loss             | 584      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 4096     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00478  |
|    learning_rate          | 0.0003   |
|    n_updates              | 63       |
|    policy_objective       | 0.283    |
|    value_loss             | 1.14e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 58.1     |
|    ep_rew_mean     | -282     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 48104    |
|    total_timesteps | 4096     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.4     |
|    ep_rew_mean            | -279     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 65       |
|    time_elapsed           | 48503    |
|    total_timesteps        | 4160     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00446  |
|    learning_rate          | 0.0003   |
|    n_updates              | 64       |
|    policy_objective       | 0.282    |
|    value_loss             | 777      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.4     |
|    ep_rew_mean            | -279     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 66       |
|    time_elapsed           | 48935    |
|    total_timesteps        | 4224     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00457  |
|    learning_rate          | 0.0003   |
|    n_updates              | 65       |
|    policy_objective       | 0.301    |
|    value_loss             | 1.21e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.8     |
|    ep_rew_mean            | -280     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 67       |
|    time_elapsed           | 49359    |
|    total_timesteps        | 4288     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00495  |
|    learning_rate          | 0.0003   |
|    n_updates              | 66       |
|    policy_objective       | 0.328    |
|    value_loss             | 1.37e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.1     |
|    ep_rew_mean            | -281     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 68       |
|    time_elapsed           | 49758    |
|    total_timesteps        | 4352     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0049   |
|    learning_rate          | 0.0003   |
|    n_updates              | 67       |
|    policy_objective       | 0.287    |
|    value_loss             | 942      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.9     |
|    ep_rew_mean            | -280     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 69       |
|    time_elapsed           | 50143    |
|    total_timesteps        | 4416     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00477  |
|    learning_rate          | 0.0003   |
|    n_updates              | 68       |
|    policy_objective       | 0.283    |
|    value_loss             | 696      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.5     |
|    ep_rew_mean            | -277     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 70       |
|    time_elapsed           | 50535    |
|    total_timesteps        | 4480     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00484  |
|    learning_rate          | 0.0003   |
|    n_updates              | 69       |
|    policy_objective       | 0.283    |
|    value_loss             | 414      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57       |
|    ep_rew_mean            | -274     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 71       |
|    time_elapsed           | 50898    |
|    total_timesteps        | 4544     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00576  |
|    learning_rate          | 0.0003   |
|    n_updates              | 70       |
|    policy_objective       | 0.342    |
|    value_loss             | 1.16e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 4608     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00442  |
|    learning_rate          | 0.0003   |
|    n_updates              | 71       |
|    policy_objective       | 0.323    |
|    value_loss             | 664      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 57       |
|    ep_rew_mean     | -274     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 54082    |
|    total_timesteps | 4608     |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 57.4      |
|    ep_rew_mean            | -276      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 73        |
|    time_elapsed           | 54503     |
|    total_timesteps        | 4672      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00465   |
|    learning_rate          | 0.0003    |
|    n_updates              | 72        |
|    policy_objective       | 0.295     |
|    value_loss             | 1e+03     |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 57.6     |
|    ep_rew_mean            | -276     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 74       |
|    time_elapsed           | 54877    |
|    total_timesteps        | 4736     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00503  |
|    learning_rate          | 0.0003   |
|    n_updates              | 73       |
|    policy_objective       | 0.315    |
|    value_loss             | 1.4e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.3     |
|    ep_rew_mean            | -279     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 75       |
|    time_elapsed           | 55264    |
|    total_timesteps        | 4800     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00521  |
|    learning_rate          | 0.0003   |
|    n_updates              | 74       |
|    policy_objective       | 0.309    |
|    value_loss             | 712      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.3     |
|    ep_rew_mean            | -279     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 76       |
|    time_elapsed           | 55668    |
|    total_timesteps        | 4864     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00563  |
|    learning_rate          | 0.0003   |
|    n_updates              | 75       |
|    policy_objective       | 0.311    |
|    value_loss             | 718      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.3     |
|    ep_rew_mean            | -279     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 77       |
|    time_elapsed           | 56028    |
|    total_timesteps        | 4928     |
| train/                    |          |
|    explained_variance     | 1.79e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00515  |
|    learning_rate          | 0.0003   |
|    n_updates              | 76       |
|    policy_objective       | 0.285    |
|    value_loss             | 1.41e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.8     |
|    ep_rew_mean            | -280     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 78       |
|    time_elapsed           | 56435    |
|    total_timesteps        | 4992     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00513  |
|    learning_rate          | 0.0003   |
|    n_updates              | 77       |
|    policy_objective       | 0.288    |
|    value_loss             | 723      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.8      |
|    ep_rew_mean            | -280      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 79        |
|    time_elapsed           | 56852     |
|    total_timesteps        | 5056      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.00599   |
|    learning_rate          | 0.0003    |
|    n_updates              | 78        |
|    policy_objective       | 0.291     |
|    value_loss             | 866       |
-----------------------------------------
```

#### logs_TRPO_0.01_baseline/progress.csv

```
time/iterations,time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,time/fps,time/time_elapsed,train/is_line_search_success,train/explained_variance,train/n_updates,train/policy_objective,train/learning_rate,train/kl_divergence_loss,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,64,15.0,-77.544756,0,485,,,,,,,,,
2,128,23.5,-109.160916,0,943,1.0,-0.024878621101379395,1,0.1491893231868744,0.0003,0.0014881810639053583,1087.6498291015625,,
3,192,18.333333333333332,-86.191373,0,1414,1.0,-1.1920928955078125e-07,2,0.15139630436897278,0.0003,0.0018603672506287694,1677.0912841796876,,
4,256,18.333333333333332,-86.191373,0,1884,1.0,0.0,3,0.1765211969614029,0.0003,0.002552302088588476,1236.2372436523438,,
5,320,42.6,-223.3717538,0,2328,1.0,-1.1920928955078125e-07,4,0.1821383535861969,0.0003,0.002748404163867235,1845.3325805664062,,
6,384,48.166666666666664,-250.21910466666665,0,2813,1.0,0.0,5,0.23841190338134766,0.0003,0.003413879545405507,1397.5708740234375,,
7,448,46.5,-252.37932037500002,0,3297,1.0,-1.1920928955078125e-07,6,0.18917052447795868,0.0003,0.0031572282314300537,1609.4874755859375,,
,512,,,,,1.0,0.0,7,0.2234266698360443,0.0003,0.0032527900766581297,1567.87666015625,-302.19571600000006,136.4
8,512,46.5,-252.37932037500002,0,6502,,,,,,,,,
9,576,46.5,-252.37932037500002,0,6983,1.0,0.0,8,0.23795200884342194,0.0003,0.004043059889227152,1808.5174438476563,,
10,640,51.18181818181818,-281.0413626363636,0,7409,1.0,0.0,9,0.21866920590400696,0.0003,0.003665490308776498,1805.7278564453125,,
11,704,48.333333333333336,-263.8874195,0,7914,1.0,0.0,10,0.23099049925804138,0.0003,0.0037951418198645115,1424.3647583007812,,
12,768,50.46153846153846,-278.04787007692306,0,8390,1.0,0.0,11,0.22460925579071045,0.0003,0.003491814713925123,1568.6758911132813,,
13,832,48.53333333333333,-267.2462798,0,8807,1.0,0.0,12,0.240183487534523,0.0003,0.0041814823634922504,1740.3809448242187,,
14,896,48.53333333333333,-267.2462798,0,9276,1.0,0.0,13,0.27959194779396057,0.0003,0.004443839192390442,877.7748229980468,,
15,960,50.76470588235294,-274.31026117647053,0,9696,1.0,5.960464477539063e-08,14,0.23819884657859802,0.0003,0.0039032078348100185,1524.3915283203125,,
,1024,,,,,1.0,5.960464477539063e-08,15,0.2824093997478485,0.0003,0.004488675855100155,1355.0455932617188,-88.03385,61.0
16,1024,50.76470588235294,-274.31026117647053,0,11366,,,,,,,,,
17,1088,52.05555555555556,-281.80843916666663,0,11839,1.0,0.0,16,0.26678356528282166,0.0003,0.004323655739426613,1604.3095092773438,,
18,1152,53.4,-290.7960164,0,12279,1.0,0.0,17,0.25825417041778564,0.0003,0.004945710767060518,1781.3670776367187,,
19,1216,52.333333333333336,-284.282319,0,12717,1.0,5.960464477539063e-08,18,0.26612192392349243,0.0003,0.004022205248475075,1572.6874145507813,,
20,1280,53.36363636363637,-289.1350007272727,0,13164,1.0,0.0,19,0.30043506622314453,0.0003,0.004266971722245216,1023.240673828125,,
21,1344,52.130434782608695,-280.89687821739125,0,13640,1.0,0.0,20,0.27600282430648804,0.0003,0.004121406003832817,1221.8793090820313,,
22,1408,52.130434782608695,-280.89687821739125,0,14123,1.0,-1.1920928955078125e-07,21,0.29171860218048096,0.0003,0.004414173774421215,1625.9212036132812,,
23,1472,54.4,-293.54065812,0,14661,1.0,0.0,22,0.25970712304115295,0.0003,0.004451603628695011,1877.1793334960937,,
,1536,,,,,1.0,0.0,23,0.2589764893054962,0.0003,0.004658356308937073,1102.8596435546874,-184.5925772,134.2
24,1536,55.0,-295.9752632592593,0,17845,,,,,,,,,
25,1600,55.0,-295.9752632592593,0,18305,1.0,0.0,24,0.2742161154747009,0.0003,0.004117611795663834,1502.2704223632813,,
26,1664,54.142857142857146,-291.3918061071429,0,18786,1.0,5.960464477539063e-08,25,0.26274651288986206,0.0003,0.0037379853893071413,1906.7043579101562,,
27,1728,54.758620689655174,-296.31144924137936,0,19217,1.0,-1.1920928955078125e-07,26,0.26796016097068787,0.0003,0.004447916056960821,1513.6040649414062,,
28,1792,55.43333333333333,-300.0057339333334,0,19662,1.0,0.0,27,0.2685543894767761,0.0003,0.004657446406781673,1094.101025390625,,
29,1856,56.0,-304.3536378387097,0,20092,1.0,0.0,28,0.2694959044456482,0.0003,0.004296374507248402,1845.3303833007812,,
30,1920,56.0,-304.3536378387097,0,20520,1.0,5.960464477539063e-08,29,0.24979449808597565,0.0003,0.0046463096514344215,1653.4411376953126,,
31,1984,56.65625,-306.95123265625006,0,20920,1.0,-1.1920928955078125e-07,30,0.27653202414512634,0.0003,0.004641881678253412,1104.310009765625,,
,2048,,,,,1.0,0.0,31,0.25516170263290405,0.0003,0.004674299154430628,725.5089416503906,-183.32112479999998,134.8
32,2048,56.65625,-306.95123265625006,0,24023,,,,,,,,,
33,2112,58.470588235294116,-314.05613773529416,0,24383,1.0,-1.1920928955078125e-07,32,0.2828855514526367,0.0003,0.004668813664466143,1207.9959228515625,,
34,2176,59.02857142857143,-315.4019449714286,0,24769,1.0,0.0,33,0.2596757113933563,0.0003,0.004374557174742222,916.9000671386718,,
35,2240,59.02857142857143,-315.4019449714286,0,25248,1.0,1.1920928955078125e-07,34,0.2717728316783905,0.0003,0.00497589074075222,1065.369873046875,,
36,2304,58.30555555555556,-310.21263208333335,0,25657,1.0,-1.1920928955078125e-07,35,0.2815384268760681,0.0003,0.00440268125385046,1563.1095581054688,,
37,2368,59.3421052631579,-311.0143297894737,0,26063,1.0,5.960464477539063e-08,36,0.310863196849823,0.0003,0.004844851791858673,934.3379699707032,,
38,2432,59.717948717948715,-313.405300974359,0,26544,1.0,-1.1920928955078125e-07,37,0.2729046940803528,0.0003,0.00442031305283308,698.8287475585937,,
39,2496,59.717948717948715,-313.405300974359,0,27011,1.0,-1.1920928955078125e-07,38,0.28230714797973633,0.0003,0.004404543898999691,1514.7473510742188,,
,2560,,,,,1.0,0.0,39,0.3007575273513794,0.0003,0.004400081001222134,1734.5921752929687,-184.4062534,398.2
40,2560,60.1,-316.121062025,0,30195,,,,,,,,,
41,2624,59.285714285714285,-310.5465729761904,0,30623,1.0,-1.1920928955078125e-07,40,0.28716328740119934,0.0003,0.0047552455216646194,958.8767883300782,,
42,2688,58.72727272727273,-306.35781620454543,0,31028,1.0,0.0,41,0.26279568672180176,0.0003,0.0038405470550060272,801.0129577636719,,
43,2752,57.82222222222222,-301.42734079999997,0,31454,1.0,0.0,42,0.2831668257713318,0.0003,0.004442079458385706,864.4633544921875,,
44,2816,58.58695652173913,-304.786285673913,0,31855,1.0,0.0,43,0.2845398485660553,0.0003,0.004066579043865204,1158.6267333984374,,
45,2880,58.125,-301.0122782083333,0,32208,1.0,0.0,44,0.25274550914764404,0.0003,0.00427200086414814,989.2832824707032,,
46,2944,58.125,-301.0122782083333,0,32604,1.0,1.1920928955078125e-07,45,0.26127198338508606,0.0003,0.004162215162068605,597.7794616699218,,
47,3008,58.1,-297.56741036,0,32990,1.0,-1.1920928955078125e-07,46,0.23896214365959167,0.0003,0.0042485021986067295,962.4001342773438,,
,3072,,,,,1.0,2.384185791015625e-07,47,0.25284498929977417,0.0003,0.0041947029531002045,503.3673980712891,-183.3637296,400.0
48,3072,58.3921568627451,-297.48603239215686,0,36183,,,,,,,,,
49,3136,57.98076923076923,-293.84049775,0,36550,1.0,1.1920928955078125e-07,48,0.2817927896976471,0.0003,0.004204469732940197,672.8502807617188,,
50,3200,57.98076923076923,-293.84049775,0,36947,1.0,0.0,49,0.288970410823822,0.0003,0.004221714101731777,846.8042236328125,,
51,3264,57.98076923076923,-293.84049775,0,37340,1.0,1.7881393432617188e-07,50,0.28191012144088745,0.0003,0.004472858272492886,1181.62763671875,,
52,3328,59.2037037037037,-297.03826499999997,0,37762,1.0,-1.1920928955078125e-07,51,0.28208717703819275,0.0003,0.004796797409653664,926.5612182617188,,
53,3392,59.163636363636364,-296.30232265454543,0,38162,1.0,-1.1920928955078125e-07,52,0.2868298888206482,0.0003,0.004303612746298313,1048.6595825195313,,
54,3456,59.767857142857146,-297.9578726964286,0,38503,1.0,-1.1920928955078125e-07,53,0.2898334264755249,0.0003,0.004483974538743496,857.4072814941406,,
55,3520,59.14035087719298,-294.29691233333335,0,38905,1.0,-1.1920928955078125e-07,54,0.29412543773651123,0.0003,0.004372728057205677,1123.3149047851562,,
,3584,,,,,1.0,-1.1920928955078125e-07,55,0.31863048672676086,0.0003,0.004989565350115299,981.0925354003906,-182.7828822,400.0
56,3584,58.55172413793103,-291.11854012068966,0,42059,,,,,,,,,
57,3648,58.733333333333334,-290.6588747,0,42482,1.0,-1.1920928955078125e-07,56,0.290283739566803,0.0003,0.004872943740338087,828.4552856445313,,
58,3712,59.24590163934426,-292.556803442623,0,42926,1.0,0.0,57,0.3345826268196106,0.0003,0.004419112112373114,793.4882995605469,,
59,3776,58.903225806451616,-290.6108905161291,0,43311,1.0,-1.1920928955078125e-07,58,0.2645239233970642,0.0003,0.004986449144780636,1145.1065307617187,,
60,3840,58.903225806451616,-290.6108905161291,0,43702,1.0,1.1920928955078125e-07,59,0.30543023347854614,0.0003,0.004873097408562899,439.557958984375,,
61,3904,59.34375,-290.768308328125,0,44058,1.0,0.0,60,0.27786266803741455,0.0003,0.005026786122471094,981.49599609375,,
62,3968,58.43939393939394,-285.130648,0,44464,1.0,0.0,61,0.28223469853401184,0.0003,0.004681121092289686,616.8443115234375,,
63,4032,57.656716417910445,-281.38523802985077,0,44918,1.0,0.0,62,0.29525524377822876,0.0003,0.004614068195223808,584.4136291503906,,
,4096,,,,,1.0,0.0,63,0.2828950583934784,0.0003,0.004775091074407101,1143.827783203125,-183.66435959999995,399.8
64,4096,58.11594202898551,-282.2790850289855,0,48104,,,,,,,,,
65,4160,57.42857142857143,-278.71963765714287,0,48503,1.0,0.0,64,0.2817509174346924,0.0003,0.004462528973817825,777.0231201171875,,
66,4224,57.42857142857143,-278.71963765714287,0,48935,1.0,1.1920928955078125e-07,65,0.30081087350845337,0.0003,0.00456650136038661,1210.651611328125,,
67,4288,57.791666666666664,-280.18418719444446,0,49359,1.0,0.0,66,0.32819703221321106,0.0003,0.0049537415616214275,1368.290478515625,,
68,4352,58.10958904109589,-281.4897627945205,0,49758,1.0,0.0,67,0.28737762570381165,0.0003,0.004898452665656805,941.6061340332031,,
69,4416,57.89333333333333,-279.61012732,0,50143,1.0,0.0,68,0.2834154963493347,0.0003,0.004765857942402363,696.1071594238281,,
70,4480,57.473684210526315,-277.0014608157894,0,50535,1.0,0.0,69,0.28255948424339294,0.0003,0.0048399316146969795,414.1732421875,,
71,4544,57.0,-274.42219179220774,0,50898,1.0,0.0,70,0.3418430984020233,0.0003,0.0057565802708268166,1164.7032470703125,,
,4608,,,,,1.0,1.1920928955078125e-07,71,0.32272204756736755,0.0003,0.004423768725246191,664.4916198730468,-183.8698022,400.0
72,4608,57.0,-274.42219179220774,0,54082,,,,,,,,,
73,4672,57.42307692307692,-275.5366971282051,0,54503,1.0,-1.1920928955078125e-07,72,0.294921338558197,0.0003,0.004646770656108856,1002.7272583007813,,
74,4736,57.64556962025316,-276.11625598734173,0,54877,1.0,1.1920928955078125e-07,73,0.31530818343162537,0.0003,0.00503154331818223,1396.81337890625,,
75,4800,58.34567901234568,-278.50844806172836,0,55264,1.0,1.1920928955078125e-07,74,0.30919161438941956,0.0003,0.00521484250202775,711.7747192382812,,
76,4864,58.34567901234568,-278.50844806172836,0,55668,1.0,0.0,75,0.31144675612449646,0.0003,0.005630571860820055,718.2232788085937,,
77,4928,58.34567901234568,-278.50844806172836,0,56028,1.0,1.7881393432617188e-07,76,0.28453120589256287,0.0003,0.005148099735379219,1405.4882690429688,,
78,4992,58.76829268292683,-280.2213298170731,0,56435,1.0,1.1920928955078125e-07,77,0.287841260433197,0.0003,0.005125473253428936,722.7692932128906,,
79,5056,58.78313253012048,-279.7804758192771,0,56852,1.0,-1.1920928955078125e-07,78,0.2906595766544342,0.0003,0.005985231138765812,866.4919982910156,,
```

#### logs_TRPO_0.0_baseline/log.txt

```
Logging to ./logs_TRPO_0.0_baseline
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 14       |
|    ep_rew_mean     | -78.8    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 411      |
|    total_timesteps | 64       |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 20.5     |
|    ep_rew_mean            | -89.7    |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 2        |
|    time_elapsed           | 860      |
|    total_timesteps        | 128      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0.000696 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0355   |
|    learning_rate          | 0.0003   |
|    n_updates              | 1        |
|    policy_objective       | 0.245    |
|    value_loss             | 1.29e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 24.3     |
|    ep_rew_mean            | -127     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 3        |
|    time_elapsed           | 1262     |
|    total_timesteps        | 192      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.154    |
|    learning_rate          | 0.0003   |
|    n_updates              | 2        |
|    policy_objective       | 1.85     |
|    value_loss             | 1.44e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 21.8     |
|    ep_rew_mean            | -110     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 4        |
|    time_elapsed           | 1722     |
|    total_timesteps        | 256      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.151    |
|    learning_rate          | 0.0003   |
|    n_updates              | 3        |
|    policy_objective       | 1.53     |
|    value_loss             | 969      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 32.2     |
|    ep_rew_mean            | -179     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 5        |
|    time_elapsed           | 2148     |
|    total_timesteps        | 320      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.163    |
|    learning_rate          | 0.0003   |
|    n_updates              | 4        |
|    policy_objective       | 2.3      |
|    value_loss             | 1.34e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 42.7     |
|    ep_rew_mean            | -221     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 6        |
|    time_elapsed           | 2522     |
|    total_timesteps        | 384      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.166    |
|    learning_rate          | 0.0003   |
|    n_updates              | 5        |
|    policy_objective       | 2.26     |
|    value_loss             | 943      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 39.2     |
|    ep_rew_mean            | -197     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 7        |
|    time_elapsed           | 2937     |
|    total_timesteps        | 448      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.161    |
|    learning_rate          | 0.0003   |
|    n_updates              | 6        |
|    policy_objective       | 2.37     |
|    value_loss             | 1.03e+03 |
----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 196       |
|    mean_reward            | -247      |
| time/                     |           |
|    total_timesteps        | 512       |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.209     |
|    learning_rate          | 0.0003    |
|    n_updates              | 7         |
|    policy_objective       | 5.2       |
|    value_loss             | 1.15e+03  |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 39.2     |
|    ep_rew_mean     | -197     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6041     |
|    total_timesteps | 512      |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 39.2     |
|    ep_rew_mean            | -197     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 9        |
|    time_elapsed           | 6361     |
|    total_timesteps        | 576      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.22     |
|    learning_rate          | 0.0003   |
|    n_updates              | 8        |
|    policy_objective       | 2.13     |
|    value_loss             | 1.08e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 48.1      |
|    ep_rew_mean            | -216      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 10        |
|    time_elapsed           | 6659      |
|    total_timesteps        | 640       |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.269     |
|    learning_rate          | 0.0003    |
|    n_updates              | 9         |
|    policy_objective       | 1.8       |
|    value_loss             | 484       |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 52.1      |
|    ep_rew_mean            | -232      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 11        |
|    time_elapsed           | 6962      |
|    total_timesteps        | 704       |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -2.38e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.442     |
|    learning_rate          | 0.0003    |
|    n_updates              | 10        |
|    policy_objective       | 19.4      |
|    value_loss             | 323       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.9     |
|    ep_rew_mean            | -228     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 12       |
|    time_elapsed           | 7318     |
|    total_timesteps        | 768      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.239    |
|    learning_rate          | 0.0003   |
|    n_updates              | 11       |
|    policy_objective       | 2.32     |
|    value_loss             | 650      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 49.8     |
|    ep_rew_mean            | -216     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 13       |
|    time_elapsed           | 7645     |
|    total_timesteps        | 832      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.269    |
|    learning_rate          | 0.0003   |
|    n_updates              | 12       |
|    policy_objective       | 2.52     |
|    value_loss             | 638      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.9     |
|    ep_rew_mean            | -213     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 14       |
|    time_elapsed           | 7995     |
|    total_timesteps        | 896      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.391    |
|    learning_rate          | 0.0003   |
|    n_updates              | 13       |
|    policy_objective       | 6.26     |
|    value_loss             | 331      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.9     |
|    ep_rew_mean            | -213     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 15       |
|    time_elapsed           | 8307     |
|    total_timesteps        | 960      |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.202    |
|    learning_rate          | 0.0003   |
|    n_updates              | 14       |
|    policy_objective       | 2.03     |
|    value_loss             | 354      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 396      |
|    mean_reward            | -224     |
| time/                     |          |
|    total_timesteps        | 1024     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.365    |
|    learning_rate          | 0.0003   |
|    n_updates              | 15       |
|    policy_objective       | 2.91     |
|    value_loss             | 542      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 51.9     |
|    ep_rew_mean     | -213     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 11419    |
|    total_timesteps | 1024     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.9     |
|    ep_rew_mean            | -213     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 17       |
|    time_elapsed           | 11705    |
|    total_timesteps        | 1088     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.475    |
|    learning_rate          | 0.0003   |
|    n_updates              | 16       |
|    policy_objective       | 3        |
|    value_loss             | 337      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 51.9     |
|    ep_rew_mean            | -213     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 18       |
|    time_elapsed           | 11918    |
|    total_timesteps        | 1152     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.282    |
|    learning_rate          | 0.0003   |
|    n_updates              | 17       |
|    policy_objective       | 1.94     |
|    value_loss             | 413      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 55.7     |
|    ep_rew_mean            | -222     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 19       |
|    time_elapsed           | 12147    |
|    total_timesteps        | 1216     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.213    |
|    learning_rate          | 0.0003   |
|    n_updates              | 18       |
|    policy_objective       | 1.54     |
|    value_loss             | 340      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 55.7     |
|    ep_rew_mean            | -222     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 20       |
|    time_elapsed           | 12368    |
|    total_timesteps        | 1280     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.275    |
|    learning_rate          | 0.0003   |
|    n_updates              | 19       |
|    policy_objective       | 1.79     |
|    value_loss             | 160      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.3      |
|    ep_rew_mean            | -223      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 21        |
|    time_elapsed           | 12578     |
|    total_timesteps        | 1344      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.308     |
|    learning_rate          | 0.0003    |
|    n_updates              | 20        |
|    policy_objective       | 1.86      |
|    value_loss             | 232       |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 58.3      |
|    ep_rew_mean            | -223      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 22        |
|    time_elapsed           | 12779     |
|    total_timesteps        | 1408      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.486     |
|    learning_rate          | 0.0003    |
|    n_updates              | 21        |
|    policy_objective       | 20.4      |
|    value_loss             | 138       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 63.4     |
|    ep_rew_mean            | -230     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 23       |
|    time_elapsed           | 12978    |
|    total_timesteps        | 1472     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.334    |
|    learning_rate          | 0.0003   |
|    n_updates              | 22       |
|    policy_objective       | 3.02     |
|    value_loss             | 229      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 386      |
|    mean_reward            | -259     |
| time/                     |          |
|    total_timesteps        | 1536     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.287    |
|    learning_rate          | 0.0003   |
|    n_updates              | 23       |
|    policy_objective       | 2.11     |
|    value_loss             | 168      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 68       |
|    ep_rew_mean     | -236     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 16015    |
|    total_timesteps | 1536     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 68       |
|    ep_rew_mean            | -236     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 25       |
|    time_elapsed           | 16239    |
|    total_timesteps        | 1600     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.317    |
|    learning_rate          | 0.0003   |
|    n_updates              | 24       |
|    policy_objective       | 1.7      |
|    value_loss             | 207      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 68       |
|    ep_rew_mean            | -236     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 26       |
|    time_elapsed           | 16456    |
|    total_timesteps        | 1664     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 1.79e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.309    |
|    learning_rate          | 0.0003   |
|    n_updates              | 25       |
|    policy_objective       | 1.42     |
|    value_loss             | 324      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 68       |
|    ep_rew_mean            | -236     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 27       |
|    time_elapsed           | 16736    |
|    total_timesteps        | 1728     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.299    |
|    learning_rate          | 0.0003   |
|    n_updates              | 26       |
|    policy_objective       | 1.17     |
|    value_loss             | 194      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 68        |
|    ep_rew_mean            | -236      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 28        |
|    time_elapsed           | 17016     |
|    total_timesteps        | 1792      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.391     |
|    learning_rate          | 0.0003    |
|    n_updates              | 27        |
|    policy_objective       | 3.21      |
|    value_loss             | 446       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 72.1     |
|    ep_rew_mean            | -241     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 29       |
|    time_elapsed           | 17196    |
|    total_timesteps        | 1856     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.411    |
|    learning_rate          | 0.0003   |
|    n_updates              | 28       |
|    policy_objective       | 1.87     |
|    value_loss             | 286      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 72.1      |
|    ep_rew_mean            | -241      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 30        |
|    time_elapsed           | 17388     |
|    total_timesteps        | 1920      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.286     |
|    learning_rate          | 0.0003    |
|    n_updates              | 29        |
|    policy_objective       | 1.85      |
|    value_loss             | 145       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 72.1     |
|    ep_rew_mean            | -241     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 31       |
|    time_elapsed           | 17634    |
|    total_timesteps        | 1984     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.299    |
|    learning_rate          | 0.0003   |
|    n_updates              | 30       |
|    policy_objective       | 1.03     |
|    value_loss             | 209      |
----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 400       |
|    mean_reward            | -421      |
| time/                     |           |
|    total_timesteps        | 2048      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -2.38e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.408     |
|    learning_rate          | 0.0003    |
|    n_updates              | 31        |
|    policy_objective       | 1.03      |
|    value_loss             | 382       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 80.1     |
|    ep_rew_mean     | -248     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 20689    |
|    total_timesteps | 2048     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 80.1     |
|    ep_rew_mean            | -248     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 33       |
|    time_elapsed           | 20884    |
|    total_timesteps        | 2112     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.374    |
|    learning_rate          | 0.0003   |
|    n_updates              | 32       |
|    policy_objective       | 1.74     |
|    value_loss             | 131      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 83.4      |
|    ep_rew_mean            | -250      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 34        |
|    time_elapsed           | 21034     |
|    total_timesteps        | 2176      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.25      |
|    learning_rate          | 0.0003    |
|    n_updates              | 33        |
|    policy_objective       | 1.25      |
|    value_loss             | 188       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 83.4     |
|    ep_rew_mean            | -250     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 35       |
|    time_elapsed           | 21172    |
|    total_timesteps        | 2240     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.279    |
|    learning_rate          | 0.0003   |
|    n_updates              | 34       |
|    policy_objective       | 2.08     |
|    value_loss             | 84       |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 83.4     |
|    ep_rew_mean            | -250     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 36       |
|    time_elapsed           | 21323    |
|    total_timesteps        | 2304     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.25     |
|    learning_rate          | 0.0003   |
|    n_updates              | 35       |
|    policy_objective       | 1.57     |
|    value_loss             | 72.4     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 83.4     |
|    ep_rew_mean            | -250     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 37       |
|    time_elapsed           | 21453    |
|    total_timesteps        | 2368     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.286    |
|    learning_rate          | 0.0003   |
|    n_updates              | 36       |
|    policy_objective       | 0.884    |
|    value_loss             | 111      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 83.4     |
|    ep_rew_mean            | -250     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 38       |
|    time_elapsed           | 21612    |
|    total_timesteps        | 2432     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.499    |
|    learning_rate          | 0.0003   |
|    n_updates              | 37       |
|    policy_objective       | 7.06     |
|    value_loss             | 95.2     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 83.4     |
|    ep_rew_mean            | -250     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 39       |
|    time_elapsed           | 21717    |
|    total_timesteps        | 2496     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.329    |
|    learning_rate          | 0.0003   |
|    n_updates              | 38       |
|    policy_objective       | 4.15     |
|    value_loss             | 162      |
----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 400       |
|    mean_reward            | -308      |
| time/                     |           |
|    total_timesteps        | 2560      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.374     |
|    learning_rate          | 0.0003    |
|    n_updates              | 39        |
|    policy_objective       | 1.81      |
|    value_loss             | 48.5      |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 83.4     |
|    ep_rew_mean     | -250     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 24678    |
|    total_timesteps | 2560     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 87.9     |
|    ep_rew_mean            | -253     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 41       |
|    time_elapsed           | 24807    |
|    total_timesteps        | 2624     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.245    |
|    learning_rate          | 0.0003   |
|    n_updates              | 40       |
|    policy_objective       | 0.652    |
|    value_loss             | 124      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 87.9     |
|    ep_rew_mean            | -253     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 42       |
|    time_elapsed           | 24966    |
|    total_timesteps        | 2688     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.471    |
|    learning_rate          | 0.0003   |
|    n_updates              | 41       |
|    policy_objective       | 2.78     |
|    value_loss             | 74.4     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 91.2     |
|    ep_rew_mean            | -252     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 43       |
|    time_elapsed           | 25100    |
|    total_timesteps        | 2752     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.249    |
|    learning_rate          | 0.0003   |
|    n_updates              | 42       |
|    policy_objective       | 0.577    |
|    value_loss             | 68.3     |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 91.2      |
|    ep_rew_mean            | -252      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 44        |
|    time_elapsed           | 25229     |
|    total_timesteps        | 2816      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.391     |
|    learning_rate          | 0.0003    |
|    n_updates              | 43        |
|    policy_objective       | 63.6      |
|    value_loss             | 52.9      |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 91.2     |
|    ep_rew_mean            | -252     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 45       |
|    time_elapsed           | 25367    |
|    total_timesteps        | 2880     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.255    |
|    learning_rate          | 0.0003   |
|    n_updates              | 44       |
|    policy_objective       | 0.752    |
|    value_loss             | 61.7     |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 91.2      |
|    ep_rew_mean            | -252      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 46        |
|    time_elapsed           | 25528     |
|    total_timesteps        | 2944      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.482     |
|    learning_rate          | 0.0003    |
|    n_updates              | 45        |
|    policy_objective       | 3.93      |
|    value_loss             | 72.1      |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 96.7     |
|    ep_rew_mean            | -255     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 47       |
|    time_elapsed           | 25669    |
|    total_timesteps        | 3008     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.291    |
|    learning_rate          | 0.0003   |
|    n_updates              | 46       |
|    policy_objective       | 1.86     |
|    value_loss             | 75       |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -248     |
| time/                     |          |
|    total_timesteps        | 3072     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.451    |
|    learning_rate          | 0.0003   |
|    n_updates              | 47       |
|    policy_objective       | 2.65     |
|    value_loss             | 75.4     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 101      |
|    ep_rew_mean     | -255     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 28635    |
|    total_timesteps | 3072     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 101       |
|    ep_rew_mean            | -255      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 49        |
|    time_elapsed           | 28793     |
|    total_timesteps        | 3136      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.307     |
|    learning_rate          | 0.0003    |
|    n_updates              | 48        |
|    policy_objective       | 5.33      |
|    value_loss             | 69.3      |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 101       |
|    ep_rew_mean            | -255      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 50        |
|    time_elapsed           | 28964     |
|    total_timesteps        | 3200      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.239     |
|    learning_rate          | 0.0003    |
|    n_updates              | 49        |
|    policy_objective       | 0.867     |
|    value_loss             | 131       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 101      |
|    ep_rew_mean            | -255     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 51       |
|    time_elapsed           | 29094    |
|    total_timesteps        | 3264     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.285    |
|    learning_rate          | 0.0003   |
|    n_updates              | 50       |
|    policy_objective       | 0.653    |
|    value_loss             | 126      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 101       |
|    ep_rew_mean            | -255      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 52        |
|    time_elapsed           | 29205     |
|    total_timesteps        | 3328      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.331     |
|    learning_rate          | 0.0003    |
|    n_updates              | 51        |
|    policy_objective       | 0.531     |
|    value_loss             | 65.3      |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 101      |
|    ep_rew_mean            | -255     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 53       |
|    time_elapsed           | 29330    |
|    total_timesteps        | 3392     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.368    |
|    learning_rate          | 0.0003   |
|    n_updates              | 52       |
|    policy_objective       | 1.03     |
|    value_loss             | 42.1     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 101      |
|    ep_rew_mean            | -255     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 54       |
|    time_elapsed           | 29442    |
|    total_timesteps        | 3456     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.287    |
|    learning_rate          | 0.0003   |
|    n_updates              | 53       |
|    policy_objective       | 0.783    |
|    value_loss             | 44.5     |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 101       |
|    ep_rew_mean            | -255      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 55        |
|    time_elapsed           | 29565     |
|    total_timesteps        | 3520      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.236     |
|    learning_rate          | 0.0003    |
|    n_updates              | 54        |
|    policy_objective       | 0.874     |
|    value_loss             | 37.9      |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -182     |
| time/                     |          |
|    total_timesteps        | 3584     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.398    |
|    learning_rate          | 0.0003   |
|    n_updates              | 55       |
|    policy_objective       | 1.19     |
|    value_loss             | 52.8     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 101      |
|    ep_rew_mean     | -255     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 32455    |
|    total_timesteps | 3584     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 101      |
|    ep_rew_mean            | -255     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 57       |
|    time_elapsed           | 32568    |
|    total_timesteps        | 3648     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.268    |
|    learning_rate          | 0.0003   |
|    n_updates              | 56       |
|    policy_objective       | 0.348    |
|    value_loss             | 33.9     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 107      |
|    ep_rew_mean            | -256     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 58       |
|    time_elapsed           | 32658    |
|    total_timesteps        | 3712     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.265    |
|    learning_rate          | 0.0003   |
|    n_updates              | 57       |
|    policy_objective       | 0.428    |
|    value_loss             | 28.4     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 107      |
|    ep_rew_mean            | -256     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 59       |
|    time_elapsed           | 32745    |
|    total_timesteps        | 3776     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.223    |
|    learning_rate          | 0.0003   |
|    n_updates              | 58       |
|    policy_objective       | 0.511    |
|    value_loss             | 28.4     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 107      |
|    ep_rew_mean            | -256     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 60       |
|    time_elapsed           | 32840    |
|    total_timesteps        | 3840     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.29     |
|    learning_rate          | 0.0003   |
|    n_updates              | 59       |
|    policy_objective       | 14.2     |
|    value_loss             | 16.9     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 113      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 61       |
|    time_elapsed           | 32940    |
|    total_timesteps        | 3904     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.454    |
|    learning_rate          | 0.0003   |
|    n_updates              | 60       |
|    policy_objective       | 0.125    |
|    value_loss             | 18.5     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 113      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 62       |
|    time_elapsed           | 33034    |
|    total_timesteps        | 3968     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.422    |
|    learning_rate          | 0.0003   |
|    n_updates              | 61       |
|    policy_objective       | 1.14     |
|    value_loss             | 25.9     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 113      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 63       |
|    time_elapsed           | 33136    |
|    total_timesteps        | 4032     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.243    |
|    learning_rate          | 0.0003   |
|    n_updates              | 62       |
|    policy_objective       | 0.372    |
|    value_loss             | 18.3     |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -186     |
| time/                     |          |
|    total_timesteps        | 4096     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.304    |
|    learning_rate          | 0.0003   |
|    n_updates              | 63       |
|    policy_objective       | 0.908    |
|    value_loss             | 20.5     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 113      |
|    ep_rew_mean     | -257     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 36044    |
|    total_timesteps | 4096     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 113      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 65       |
|    time_elapsed           | 36176    |
|    total_timesteps        | 4160     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.378    |
|    learning_rate          | 0.0003   |
|    n_updates              | 64       |
|    policy_objective       | 0.427    |
|    value_loss             | 17.3     |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 119       |
|    ep_rew_mean            | -257      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 66        |
|    time_elapsed           | 36301     |
|    total_timesteps        | 4224      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.236     |
|    learning_rate          | 0.0003    |
|    n_updates              | 65        |
|    policy_objective       | 0.979     |
|    value_loss             | 38.3      |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 67       |
|    time_elapsed           | 36401    |
|    total_timesteps        | 4288     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.237    |
|    learning_rate          | 0.0003   |
|    n_updates              | 66       |
|    policy_objective       | 0.519    |
|    value_loss             | 36.7     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 68       |
|    time_elapsed           | 36504    |
|    total_timesteps        | 4352     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.384    |
|    learning_rate          | 0.0003   |
|    n_updates              | 67       |
|    policy_objective       | 0.612    |
|    value_loss             | 27       |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 69       |
|    time_elapsed           | 36625    |
|    total_timesteps        | 4416     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.1      |
|    learning_rate          | 0.0003   |
|    n_updates              | 68       |
|    policy_objective       | 0.333    |
|    value_loss             | 22.5     |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 125       |
|    ep_rew_mean            | -257      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 70        |
|    time_elapsed           | 36735     |
|    total_timesteps        | 4480      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.166     |
|    learning_rate          | 0.0003    |
|    n_updates              | 69        |
|    policy_objective       | 0.29      |
|    value_loss             | 28        |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 125       |
|    ep_rew_mean            | -257      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 71        |
|    time_elapsed           | 36834     |
|    total_timesteps        | 4544      |
| train/                    |           |
|    adaptive_beta          | 0.5       |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.324     |
|    learning_rate          | 0.0003    |
|    n_updates              | 70        |
|    policy_objective       | 0.29      |
|    value_loss             | 22.2      |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 4608     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.261    |
|    learning_rate          | 0.0003   |
|    n_updates              | 71       |
|    policy_objective       | 0.484    |
|    value_loss             | 15.3     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 125      |
|    ep_rew_mean     | -257     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 39740    |
|    total_timesteps | 4608     |
| train/             |          |
|    adaptive_beta   | 0.5      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 73       |
|    time_elapsed           | 39838    |
|    total_timesteps        | 4672     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.383    |
|    learning_rate          | 0.0003   |
|    n_updates              | 72       |
|    policy_objective       | 0.341    |
|    value_loss             | 13.5     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 74       |
|    time_elapsed           | 39940    |
|    total_timesteps        | 4736     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.287    |
|    learning_rate          | 0.0003   |
|    n_updates              | 73       |
|    policy_objective       | 0.291    |
|    value_loss             | 14.8     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 75       |
|    time_elapsed           | 40036    |
|    total_timesteps        | 4800     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.079    |
|    learning_rate          | 0.0003   |
|    n_updates              | 74       |
|    policy_objective       | 0.096    |
|    value_loss             | 16.3     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 76       |
|    time_elapsed           | 40127    |
|    total_timesteps        | 4864     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.211    |
|    learning_rate          | 0.0003   |
|    n_updates              | 75       |
|    policy_objective       | 0.247    |
|    value_loss             | 12.4     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 77       |
|    time_elapsed           | 40218    |
|    total_timesteps        | 4928     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.478    |
|    learning_rate          | 0.0003   |
|    n_updates              | 76       |
|    policy_objective       | 0.419    |
|    value_loss             | 11.1     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 78       |
|    time_elapsed           | 40308    |
|    total_timesteps        | 4992     |
| train/                    |          |
|    adaptive_beta          | 0.5      |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.335    |
|    learning_rate          | 0.0003   |
|    n_updates              | 77       |
|    policy_objective       | 0.189    |
|    value_loss             | 11       |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 125      |
|    ep_rew_mean            | -257     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 79       |
|    time_elapsed           | 40403    |
|    total_timesteps        | 5056     |
| train/                    |          |
|    adaptive_beta          | 0.475    |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.334    |
|    learning_rate          | 0.0003   |
|    n_updates              | 78       |
|    policy_objective       | 0.117    |
|    value_loss             | 11       |
----------------------------------------
```

#### logs_TRPO_0.0_baseline/progress.csv

```
time/iterations,train/adaptive_beta,time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,time/fps,time/time_elapsed,train/is_line_search_success,train/explained_variance,train/n_updates,train/policy_objective,train/learning_rate,train/kl_divergence_loss,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,0.5,64,14.0,-78.7798,0,411,,,,,,,,,
2,0.5,128,20.5,-89.654096,0,860,1.0,0.0006957054138183594,1,0.24477379024028778,0.0003,0.035498976707458496,1293.2154296875,,
3,0.5,192,24.333333333333332,-126.60591133333332,0,1262,1.0,0.0,2,1.849414587020874,0.0003,0.1535874605178833,1444.34814453125,,
4,0.5,256,21.75,-110.42305925,0,1722,1.0,0.0,3,1.5286444425582886,0.0003,0.15079380571842194,969.3696411132812,,
5,0.5,320,32.2,-179.143625,0,2148,1.0,0.0,4,2.2974798679351807,0.0003,0.16256506741046906,1339.3044799804688,,
6,0.5,384,42.714285714285715,-221.286107,0,2522,1.0,0.0,5,2.258720874786377,0.0003,0.16599521040916443,943.1713256835938,,
7,0.5,448,39.25,-196.75962,0,2937,1.0,5.960464477539063e-08,6,2.3695261478424072,0.0003,0.16091668605804443,1025.0928344726562,,
,0.5,512,,,,,1.0,-1.1920928955078125e-07,7,5.204194068908691,0.0003,0.2091255635023117,1145.4092895507813,-246.98299260000005,195.8
8,0.5,512,39.25,-196.75962,0,6041,,,,,,,,,
9,0.5,576,39.25,-196.75962,0,6361,1.0,0.0,8,2.125321865081787,0.0003,0.21959537267684937,1076.0040283203125,,
10,0.5,640,48.1,-215.55371730000002,0,6659,1.0,-1.1920928955078125e-07,9,1.7968025207519531,0.0003,0.2686240077018738,483.81427612304685,,
11,0.5,704,52.09090909090909,-232.0294420909091,0,6962,1.0,-2.384185791015625e-07,10,19.378665924072266,0.0003,0.4420636296272278,322.9839294433594,,
12,0.5,768,51.92307692307692,-227.8325006153846,0,7318,1.0,0.0,11,2.3232357501983643,0.0003,0.23871488869190216,650.4200561523437,,
13,0.5,832,49.785714285714285,-215.96995707142855,0,7645,1.0,0.0,12,2.5218236446380615,0.0003,0.2687276601791382,638.2035888671875,,
14,0.5,896,51.875,-213.40619175,0,7995,1.0,5.960464477539063e-08,13,6.25534725189209,0.0003,0.3910481631755829,331.29859924316406,,
15,0.5,960,51.875,-213.40619175,0,8307,1.0,5.960464477539063e-08,14,2.0263710021972656,0.0003,0.2021431177854538,354.09836730957034,,
,0.5,1024,,,,,1.0,0.0,15,2.9087471961975098,0.0003,0.3654627501964569,542.1618408203125,-224.37688980000001,396.2
16,0.5,1024,51.875,-213.40619175,0,11419,,,,,,,,,
17,0.5,1088,51.875,-213.40619175,0,11705,1.0,1.1920928955078125e-07,16,2.9953229427337646,0.0003,0.47547054290771484,337.11424865722654,,
18,0.5,1152,51.875,-213.40619175,0,11918,1.0,0.0,17,1.9392261505126953,0.0003,0.2821427583694458,412.59605407714844,,
19,0.5,1216,55.705882352941174,-221.85325076470588,0,12147,1.0,0.0,18,1.5426325798034668,0.0003,0.21325691044330597,339.70554809570314,,
20,0.5,1280,55.705882352941174,-221.85325076470588,0,12368,1.0,0.0,19,1.7947720289230347,0.0003,0.2747221291065216,159.5761505126953,,
21,0.5,1344,58.27777777777778,-222.6912076111111,0,12578,1.0,-1.1920928955078125e-07,20,1.8577055931091309,0.0003,0.30794471502304077,232.3834442138672,,
22,0.5,1408,58.27777777777778,-222.6912076111111,0,12779,1.0,-1.1920928955078125e-07,21,20.411575317382812,0.0003,0.48557013273239136,138.46969451904297,,
23,0.5,1472,63.36842105263158,-229.60846005263159,0,12978,1.0,0.0,22,3.0160064697265625,0.0003,0.3344595432281494,229.19335479736327,,
,0.5,1536,,,,,1.0,0.0,23,2.1104953289031982,0.0003,0.28699690103530884,168.08270568847655,-258.8051572,386.0
24,0.5,1536,67.95,-235.57237185,0,16015,,,,,,,,,
25,0.5,1600,67.95,-235.57237185,0,16239,1.0,0.0,24,1.6982699632644653,0.0003,0.3168380558490753,207.3227981567383,,
26,0.5,1664,67.95,-235.57237185,0,16456,1.0,1.7881393432617188e-07,25,1.4216601848602295,0.0003,0.30861368775367737,323.88990783691406,,
27,0.5,1728,67.95,-235.57237185,0,16736,1.0,0.0,26,1.1658217906951904,0.0003,0.298720121383667,193.6035583496094,,
28,0.5,1792,67.95,-235.57237185,0,17016,1.0,-1.1920928955078125e-07,27,3.206035614013672,0.0003,0.3910781741142273,446.22999572753906,,
29,0.5,1856,72.14285714285714,-241.04192614285716,0,17196,1.0,5.960464477539063e-08,28,1.8651503324508667,0.0003,0.4105904698371887,285.5618560791016,,
30,0.5,1920,72.14285714285714,-241.04192614285716,0,17388,1.0,-1.1920928955078125e-07,29,1.8475923538208008,0.0003,0.2861670255661011,145.08311157226564,,
31,0.5,1984,72.14285714285714,-241.04192614285716,0,17634,1.0,0.0,30,1.0328909158706665,0.0003,0.29934680461883545,208.8363494873047,,
,0.5,2048,,,,,1.0,-2.384185791015625e-07,31,1.0250279903411865,0.0003,0.40815091133117676,381.6971923828125,-421.0851692,400.0
32,0.5,2048,80.1304347826087,-247.82102473913042,0,20689,,,,,,,,,
33,0.5,2112,80.1304347826087,-247.82102473913042,0,20884,1.0,0.0,32,1.7380058765411377,0.0003,0.3741450309753418,130.72793273925782,,
34,0.5,2176,83.41666666666667,-250.37061041666666,0,21034,1.0,-1.1920928955078125e-07,33,1.246248722076416,0.0003,0.24987433850765228,187.72982940673828,,
35,0.5,2240,83.41666666666667,-250.37061041666666,0,21172,1.0,0.0,34,2.079833507537842,0.0003,0.2791439890861511,83.9620376586914,,
36,0.5,2304,83.41666666666667,-250.37061041666666,0,21323,1.0,1.1920928955078125e-07,35,1.5736507177352905,0.0003,0.2495899647474289,72.41566772460938,,
37,0.5,2368,83.41666666666667,-250.37061041666666,0,21453,1.0,0.0,36,0.8838262557983398,0.0003,0.2856942415237427,111.38203659057618,,
38,0.5,2432,83.41666666666667,-250.37061041666666,0,21612,1.0,0.0,37,7.063626289367676,0.0003,0.4991901218891144,95.24980239868164,,
39,0.5,2496,83.41666666666667,-250.37061041666666,0,21717,1.0,0.0,38,4.154143810272217,0.0003,0.3293285071849823,162.1160400390625,,
,0.5,2560,,,,,1.0,-1.1920928955078125e-07,39,1.8054136037826538,0.0003,0.37413525581359863,48.51988983154297,-307.6199126,400.0
40,0.5,2560,83.41666666666667,-250.37061041666666,0,24678,,,,,,,,,
41,0.5,2624,87.92,-253.05178984000003,0,24807,1.0,5.960464477539063e-08,40,0.6524631381034851,0.0003,0.24490126967430115,124.46904144287109,,
42,0.5,2688,87.92,-253.05178984000003,0,24966,1.0,0.0,41,2.782616138458252,0.0003,0.47086215019226074,74.441357421875,,
43,0.5,2752,91.23076923076923,-251.934057,0,25100,1.0,1.1920928955078125e-07,42,0.5772613286972046,0.0003,0.2485140860080719,68.25503311157226,,
44,0.5,2816,91.23076923076923,-251.934057,0,25229,1.0,-1.1920928955078125e-07,43,63.560733795166016,0.0003,0.3908517360687256,52.85265045166015,,
45,0.5,2880,91.23076923076923,-251.934057,0,25367,1.0,0.0,44,0.7520185708999634,0.0003,0.25524941086769104,61.682029724121094,,
46,0.5,2944,91.23076923076923,-251.934057,0,25528,1.0,-1.1920928955078125e-07,45,3.927596092224121,0.0003,0.4819995164871216,72.13735122680664,,
47,0.5,3008,96.66666666666667,-254.80641670370372,0,25669,1.0,0.0,46,1.8621196746826172,0.0003,0.29099005460739136,75.01411972045898,,
,0.5,3072,,,,,1.0,0.0,47,2.646557331085205,0.0003,0.45121753215789795,75.35675506591797,-248.08594279999997,400.0
48,0.5,3072,100.92857142857143,-254.80836314285716,0,28635,,,,,,,,,
49,0.5,3136,100.92857142857143,-254.80836314285716,0,28793,1.0,-1.1920928955078125e-07,48,5.332037925720215,0.0003,0.3066028952598572,69.29304428100586,,
50,0.5,3200,100.92857142857143,-254.80836314285716,0,28964,1.0,-1.1920928955078125e-07,49,0.8673243522644043,0.0003,0.23883545398712158,131.27445220947266,,
51,0.5,3264,100.92857142857143,-254.80836314285716,0,29094,1.0,0.0,50,0.6531758904457092,0.0003,0.28498727083206177,125.92226943969726,,
52,0.5,3328,100.92857142857143,-254.80836314285716,0,29205,1.0,-1.1920928955078125e-07,51,0.5314767956733704,0.0003,0.3309982120990753,65.2776107788086,,
53,0.5,3392,100.92857142857143,-254.80836314285716,0,29330,1.0,0.0,52,1.034723162651062,0.0003,0.36847877502441406,42.073309326171874,,
54,0.5,3456,100.92857142857143,-254.80836314285716,0,29442,1.0,1.1920928955078125e-07,53,0.7825139760971069,0.0003,0.28700315952301025,44.52385368347168,,
55,0.5,3520,100.92857142857143,-254.80836314285716,0,29565,1.0,-1.1920928955078125e-07,54,0.8739372491836548,0.0003,0.23572327196598053,37.91524314880371,,
,0.5,3584,,,,,1.0,5.960464477539063e-08,55,1.1945066452026367,0.0003,0.39760956168174744,52.78259506225586,-182.35718,400.0
56,0.5,3584,100.92857142857143,-254.80836314285716,0,32455,,,,,,,,,
57,0.5,3648,100.92857142857143,-254.80836314285716,0,32568,1.0,0.0,56,0.3484569787979126,0.0003,0.26750051975250244,33.85471687316895,,
58,0.5,3712,106.93103448275862,-256.05884686206895,0,32658,1.0,0.0,57,0.4281923770904541,0.0003,0.2653506398200989,28.362645530700682,,
59,0.5,3776,106.93103448275862,-256.05884686206895,0,32745,1.0,0.0,58,0.5110276341438293,0.0003,0.2232457399368286,28.377955436706543,,
60,0.5,3840,106.93103448275862,-256.05884686206895,0,32840,1.0,0.0,59,14.20184326171875,0.0003,0.28968560695648193,16.931870651245116,,
61,0.5,3904,113.1,-257.2356525,0,32940,1.0,0.0,60,0.12477415800094604,0.0003,0.45379069447517395,18.54473533630371,,
62,0.5,3968,113.1,-257.2356525,0,33034,1.0,0.0,61,1.1448760032653809,0.0003,0.4218311309814453,25.941247177124023,,
63,0.5,4032,113.1,-257.2356525,0,33136,1.0,0.0,62,0.3716823160648346,0.0003,0.24322697520256042,18.280438232421876,,
,0.5,4096,,,,,1.0,0.0,63,0.9075140953063965,0.0003,0.3036412000656128,20.537005615234374,-185.8658566,400.0
64,0.5,4096,113.1,-257.2356525,0,36044,,,,,,,,,
65,0.5,4160,113.1,-257.2356525,0,36176,1.0,0.0,64,0.4265362322330475,0.0003,0.3783677816390991,17.260285568237304,,
66,0.5,4224,119.38709677419355,-257.2160111935484,0,36301,1.0,-1.1920928955078125e-07,65,0.9793795943260193,0.0003,0.23604975640773773,38.32394905090332,,
67,0.5,4288,125.34375,-256.97266778125004,0,36401,1.0,0.0,66,0.5189065933227539,0.0003,0.23741793632507324,36.69276809692383,,
68,0.5,4352,125.34375,-256.97266778125004,0,36504,1.0,0.0,67,0.6117376685142517,0.0003,0.38427603244781494,27.006244087219237,,
69,0.5,4416,125.34375,-256.97266778125004,0,36625,1.0,0.0,68,0.3326306641101837,0.0003,0.10042104125022888,22.456539726257326,,
70,0.5,4480,125.34375,-256.97266778125004,0,36735,1.0,-1.1920928955078125e-07,69,0.2899148166179657,0.0003,0.16642829775810242,27.999152374267577,,
71,0.5,4544,125.34375,-256.97266778125004,0,36834,1.0,-1.1920928955078125e-07,70,0.2899656295776367,0.0003,0.32388466596603394,22.1860631942749,,
,0.5,4608,,,,,1.0,0.0,71,0.484147310256958,0.0003,0.26133185625076294,15.259608745574951,-184.0077526,400.0
72,0.5,4608,125.34375,-256.97266778125004,0,39740,,,,,,,,,
73,0.5,4672,125.34375,-256.97266778125004,0,39838,1.0,0.0,72,0.34098824858665466,0.0003,0.3827637732028961,13.478408050537109,,
74,0.5,4736,125.34375,-256.97266778125004,0,39940,1.0,0.0,73,0.29105573892593384,0.0003,0.2874658703804016,14.802531433105468,,
75,0.5,4800,125.34375,-256.97266778125004,0,40036,1.0,0.0,74,0.09598854184150696,0.0003,0.07895992696285248,16.334272956848146,,
76,0.5,4864,125.34375,-256.97266778125004,0,40127,1.0,0.0,75,0.24711784720420837,0.0003,0.21056991815567017,12.372817897796631,,
77,0.5,4928,125.34375,-256.97266778125004,0,40218,1.0,0.0,76,0.4188193082809448,0.0003,0.47792476415634155,11.104378700256348,,
78,0.5,4992,125.34375,-256.97266778125004,0,40308,1.0,0.0,77,0.18927378952503204,0.0003,0.3347945809364319,11.024467086791992,,
79,0.475,5056,125.34375,-256.97266778125004,0,40403,1.0,0.0,78,0.11733293533325195,0.0003,0.3337523937225342,11.018231773376465,,
```

#### logs_TRPO_0.1_baseline/log.txt

```
Logging to ./logs_TRPO_0.1_baseline
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 15       |
|    ep_rew_mean     | -79.3    |
| time/              |          |
|    fps             | 0        |
|    iterations      | 1        |
|    time_elapsed    | 500      |
|    total_timesteps | 64       |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 23       |
|    ep_rew_mean            | -109     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 2        |
|    time_elapsed           | 979      |
|    total_timesteps        | 128      |
| train/                    |          |
|    explained_variance     | -0.0279  |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.00448  |
|    learning_rate          | 0.0003   |
|    n_updates              | 1        |
|    policy_objective       | 0.197    |
|    value_loss             | 1.14e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 18.3     |
|    ep_rew_mean            | -90.6    |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 3        |
|    time_elapsed           | 1506     |
|    total_timesteps        | 192      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0231   |
|    learning_rate          | 0.0003   |
|    n_updates              | 2        |
|    policy_objective       | 0.392    |
|    value_loss             | 1.91e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 18.3     |
|    ep_rew_mean            | -90.6    |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 4        |
|    time_elapsed           | 1968     |
|    total_timesteps        | 256      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0467   |
|    learning_rate          | 0.0003   |
|    n_updates              | 3        |
|    policy_objective       | 0.683    |
|    value_loss             | 2.03e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 35.8      |
|    ep_rew_mean            | -212      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 5         |
|    time_elapsed           | 2440      |
|    total_timesteps        | 320       |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0407    |
|    learning_rate          | 0.0003    |
|    n_updates              | 4         |
|    policy_objective       | 0.625     |
|    value_loss             | 2e+03     |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 40.9     |
|    ep_rew_mean            | -222     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 6        |
|    time_elapsed           | 2938     |
|    total_timesteps        | 384      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0324   |
|    learning_rate          | 0.0003   |
|    n_updates              | 5        |
|    policy_objective       | 0.684    |
|    value_loss             | 1.2e+03  |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 41.1     |
|    ep_rew_mean            | -233     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 7        |
|    time_elapsed           | 3452     |
|    total_timesteps        | 448      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0382   |
|    learning_rate          | 0.0003   |
|    n_updates              | 6        |
|    policy_objective       | 0.724    |
|    value_loss             | 1.29e+03 |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 197      |
|    mean_reward            | -292     |
| time/                     |          |
|    total_timesteps        | 512      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0421   |
|    learning_rate          | 0.0003   |
|    n_updates              | 7        |
|    policy_objective       | 0.941    |
|    value_loss             | 1.81e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 41.1     |
|    ep_rew_mean     | -233     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 8        |
|    time_elapsed    | 6695     |
|    total_timesteps | 512      |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 40.5     |
|    ep_rew_mean            | -226     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 9        |
|    time_elapsed           | 7151     |
|    total_timesteps        | 576      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0856   |
|    learning_rate          | 0.0003   |
|    n_updates              | 8        |
|    policy_objective       | 1.53     |
|    value_loss             | 1.74e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 46.2     |
|    ep_rew_mean            | -259     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 10       |
|    time_elapsed           | 7606     |
|    total_timesteps        | 640      |
| train/                    |          |
|    explained_variance     | 1.79e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0427   |
|    learning_rate          | 0.0003   |
|    n_updates              | 9        |
|    policy_objective       | 0.865    |
|    value_loss             | 1.32e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.7     |
|    ep_rew_mean            | -249     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 11       |
|    time_elapsed           | 8047     |
|    total_timesteps        | 704      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0617   |
|    learning_rate          | 0.0003   |
|    n_updates              | 10       |
|    policy_objective       | 0.88     |
|    value_loss             | 1.05e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.5     |
|    ep_rew_mean            | -246     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 12       |
|    time_elapsed           | 8476     |
|    total_timesteps        | 768      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0578   |
|    learning_rate          | 0.0003   |
|    n_updates              | 11       |
|    policy_objective       | 1.1      |
|    value_loss             | 992      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 45.2     |
|    ep_rew_mean            | -245     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 13       |
|    time_elapsed           | 8935     |
|    total_timesteps        | 832      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0547   |
|    learning_rate          | 0.0003   |
|    n_updates              | 12       |
|    policy_objective       | 0.824    |
|    value_loss             | 1.27e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 45.5     |
|    ep_rew_mean            | -245     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 14       |
|    time_elapsed           | 9308     |
|    total_timesteps        | 896      |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0521   |
|    learning_rate          | 0.0003   |
|    n_updates              | 13       |
|    policy_objective       | 0.859    |
|    value_loss             | 1.23e+03 |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.1     |
|    ep_rew_mean            | -236     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 15       |
|    time_elapsed           | 9714     |
|    total_timesteps        | 960      |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0559   |
|    learning_rate          | 0.0003   |
|    n_updates              | 14       |
|    policy_objective       | 0.92     |
|    value_loss             | 708      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 1024     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0624   |
|    learning_rate          | 0.0003   |
|    n_updates              | 15       |
|    policy_objective       | 1.6      |
|    value_loss             | 1.21e+03 |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 45.2     |
|    ep_rew_mean     | -241     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 16       |
|    time_elapsed    | 12944    |
|    total_timesteps | 1024     |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 47        |
|    ep_rew_mean            | -245      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 17        |
|    time_elapsed           | 13397     |
|    total_timesteps        | 1088      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0474    |
|    learning_rate          | 0.0003    |
|    n_updates              | 16        |
|    policy_objective       | 0.883     |
|    value_loss             | 958       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 47       |
|    ep_rew_mean            | -245     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 18       |
|    time_elapsed           | 13790    |
|    total_timesteps        | 1152     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0557   |
|    learning_rate          | 0.0003   |
|    n_updates              | 17       |
|    policy_objective       | 0.696    |
|    value_loss             | 1.02e+03 |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 46.3      |
|    ep_rew_mean            | -236      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 19        |
|    time_elapsed           | 14217     |
|    total_timesteps        | 1216      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0517    |
|    learning_rate          | 0.0003    |
|    n_updates              | 18        |
|    policy_objective       | 0.968     |
|    value_loss             | 1.01e+03  |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.2     |
|    ep_rew_mean            | -223     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 20       |
|    time_elapsed           | 14603    |
|    total_timesteps        | 1280     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0508   |
|    learning_rate          | 0.0003   |
|    n_updates              | 19       |
|    policy_objective       | 1        |
|    value_loss             | 383      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.2     |
|    ep_rew_mean            | -223     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 21       |
|    time_elapsed           | 14972    |
|    total_timesteps        | 1344     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0496   |
|    learning_rate          | 0.0003   |
|    n_updates              | 20       |
|    policy_objective       | 0.768    |
|    value_loss             | 932      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 44.2     |
|    ep_rew_mean            | -223     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 22       |
|    time_elapsed           | 15330    |
|    total_timesteps        | 1408     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0559   |
|    learning_rate          | 0.0003   |
|    n_updates              | 21       |
|    policy_objective       | 0.755    |
|    value_loss             | 925      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 46.5     |
|    ep_rew_mean            | -230     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 23       |
|    time_elapsed           | 15670    |
|    total_timesteps        | 1472     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0495   |
|    learning_rate          | 0.0003   |
|    n_updates              | 22       |
|    policy_objective       | 0.839    |
|    value_loss             | 698      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 391      |
|    mean_reward            | -188     |
| time/                     |          |
|    total_timesteps        | 1536     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.052    |
|    learning_rate          | 0.0003   |
|    n_updates              | 23       |
|    policy_objective       | 0.798    |
|    value_loss             | 569      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 46.5     |
|    ep_rew_mean     | -230     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 24       |
|    time_elapsed    | 18834    |
|    total_timesteps | 1536     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 47       |
|    ep_rew_mean            | -230     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 25       |
|    time_elapsed           | 19186    |
|    total_timesteps        | 1600     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0557   |
|    learning_rate          | 0.0003   |
|    n_updates              | 24       |
|    policy_objective       | 1.15     |
|    value_loss             | 878      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.5     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 26       |
|    time_elapsed           | 19515    |
|    total_timesteps        | 1664     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0545   |
|    learning_rate          | 0.0003   |
|    n_updates              | 25       |
|    policy_objective       | 1.01     |
|    value_loss             | 748      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.5     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 27       |
|    time_elapsed           | 19890    |
|    total_timesteps        | 1728     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0506   |
|    learning_rate          | 0.0003   |
|    n_updates              | 26       |
|    policy_objective       | 0.673    |
|    value_loss             | 557      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 48.5     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 28       |
|    time_elapsed           | 20223    |
|    total_timesteps        | 1792     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0495   |
|    learning_rate          | 0.0003   |
|    n_updates              | 27       |
|    policy_objective       | 0.679    |
|    value_loss             | 965      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.1     |
|    ep_rew_mean            | -240     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 29       |
|    time_elapsed           | 20576    |
|    total_timesteps        | 1856     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0604   |
|    learning_rate          | 0.0003   |
|    n_updates              | 28       |
|    policy_objective       | 0.779    |
|    value_loss             | 1e+03    |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.5     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 30       |
|    time_elapsed           | 20913    |
|    total_timesteps        | 1920     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0824   |
|    learning_rate          | 0.0003   |
|    n_updates              | 29       |
|    policy_objective       | 0.826    |
|    value_loss             | 685      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 50.5     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 31       |
|    time_elapsed           | 21236    |
|    total_timesteps        | 1984     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0532   |
|    learning_rate          | 0.0003   |
|    n_updates              | 30       |
|    policy_objective       | 1.12     |
|    value_loss             | 473      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 394      |
|    mean_reward            | -184     |
| time/                     |          |
|    total_timesteps        | 2048     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0612   |
|    learning_rate          | 0.0003   |
|    n_updates              | 31       |
|    policy_objective       | 0.771    |
|    value_loss             | 590      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 53.9     |
|    ep_rew_mean     | -251     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 32       |
|    time_elapsed    | 24381    |
|    total_timesteps | 2048     |
---------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.2      |
|    ep_rew_mean            | -249      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 33        |
|    time_elapsed           | 24687     |
|    total_timesteps        | 2112      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0565    |
|    learning_rate          | 0.0003    |
|    n_updates              | 32        |
|    policy_objective       | 0.693     |
|    value_loss             | 508       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.2     |
|    ep_rew_mean            | -249     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 34       |
|    time_elapsed           | 25022    |
|    total_timesteps        | 2176     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0639   |
|    learning_rate          | 0.0003   |
|    n_updates              | 33       |
|    policy_objective       | 1.03     |
|    value_loss             | 428      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.2      |
|    ep_rew_mean            | -249      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 35        |
|    time_elapsed           | 25333     |
|    total_timesteps        | 2240      |
| train/                    |           |
|    explained_variance     | -2.38e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0682    |
|    learning_rate          | 0.0003    |
|    n_updates              | 34        |
|    policy_objective       | 0.868     |
|    value_loss             | 696       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.2     |
|    ep_rew_mean            | -244     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 36       |
|    time_elapsed           | 25652    |
|    total_timesteps        | 2304     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0814   |
|    learning_rate          | 0.0003   |
|    n_updates              | 35       |
|    policy_objective       | 1        |
|    value_loss             | 468      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.6     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 37       |
|    time_elapsed           | 25950    |
|    total_timesteps        | 2368     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0719   |
|    learning_rate          | 0.0003   |
|    n_updates              | 36       |
|    policy_objective       | 0.676    |
|    value_loss             | 311      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 54.6      |
|    ep_rew_mean            | -242      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 38        |
|    time_elapsed           | 26264     |
|    total_timesteps        | 2432      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0793    |
|    learning_rate          | 0.0003    |
|    n_updates              | 37        |
|    policy_objective       | 1.01      |
|    value_loss             | 256       |
-----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 55.2      |
|    ep_rew_mean            | -241      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 39        |
|    time_elapsed           | 26529     |
|    total_timesteps        | 2496      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0756    |
|    learning_rate          | 0.0003    |
|    n_updates              | 38        |
|    policy_objective       | 0.85      |
|    value_loss             | 690       |
-----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 396      |
|    mean_reward            | -205     |
| time/                     |          |
|    total_timesteps        | 2560     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0595   |
|    learning_rate          | 0.0003   |
|    n_updates              | 39       |
|    policy_objective       | 0.794    |
|    value_loss             | 192      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 55.2     |
|    ep_rew_mean     | -238     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 40       |
|    time_elapsed    | 29613    |
|    total_timesteps | 2560     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.8     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 41       |
|    time_elapsed           | 29897    |
|    total_timesteps        | 2624     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0645   |
|    learning_rate          | 0.0003   |
|    n_updates              | 40       |
|    policy_objective       | 1        |
|    value_loss             | 292      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 54.8     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 42       |
|    time_elapsed           | 30139    |
|    total_timesteps        | 2688     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.073    |
|    learning_rate          | 0.0003   |
|    n_updates              | 41       |
|    policy_objective       | 0.905    |
|    value_loss             | 334      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 56.2     |
|    ep_rew_mean            | -237     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 43       |
|    time_elapsed           | 30395    |
|    total_timesteps        | 2752     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0628   |
|    learning_rate          | 0.0003   |
|    n_updates              | 42       |
|    policy_objective       | 0.856    |
|    value_loss             | 399      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 56.3     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 44       |
|    time_elapsed           | 30676    |
|    total_timesteps        | 2816     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0739   |
|    learning_rate          | 0.0003   |
|    n_updates              | 43       |
|    policy_objective       | 0.521    |
|    value_loss             | 332      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 56.3     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 45       |
|    time_elapsed           | 30958    |
|    total_timesteps        | 2880     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0669   |
|    learning_rate          | 0.0003   |
|    n_updates              | 44       |
|    policy_objective       | 0.776    |
|    value_loss             | 525      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 56.3     |
|    ep_rew_mean            | -234     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 46       |
|    time_elapsed           | 31179    |
|    total_timesteps        | 2944     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0787   |
|    learning_rate          | 0.0003   |
|    n_updates              | 45       |
|    policy_objective       | 0.773    |
|    value_loss             | 376      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 56.3      |
|    ep_rew_mean            | -234      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 47        |
|    time_elapsed           | 31384     |
|    total_timesteps        | 3008      |
| train/                    |           |
|    explained_variance     | -2.38e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0614    |
|    learning_rate          | 0.0003    |
|    n_updates              | 46        |
|    policy_objective       | 0.68      |
|    value_loss             | 273       |
-----------------------------------------
-----------------------------------------
| eval/                     |           |
|    mean_ep_length         | 400       |
|    mean_reward            | -183      |
| time/                     |           |
|    total_timesteps        | 3072      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0786    |
|    learning_rate          | 0.0003    |
|    n_updates              | 47        |
|    policy_objective       | 0.606     |
|    value_loss             | 135       |
-----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 58.2     |
|    ep_rew_mean     | -235     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 48       |
|    time_elapsed    | 34405    |
|    total_timesteps | 3072     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 58.2     |
|    ep_rew_mean            | -235     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 49       |
|    time_elapsed           | 34671    |
|    total_timesteps        | 3136     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0694   |
|    learning_rate          | 0.0003   |
|    n_updates              | 48       |
|    policy_objective       | 1.25     |
|    value_loss             | 243      |
----------------------------------------
-----------------------------------------
| rollout/                  |           |
|    ep_len_mean            | 59.9      |
|    ep_rew_mean            | -237      |
| time/                     |           |
|    fps                    | 0         |
|    iterations             | 50        |
|    time_elapsed           | 34899     |
|    total_timesteps        | 3200      |
| train/                    |           |
|    explained_variance     | -1.19e-07 |
|    is_line_search_success | 1         |
|    kl_divergence_loss     | 0.0596    |
|    learning_rate          | 0.0003    |
|    n_updates              | 49        |
|    policy_objective       | 0.643     |
|    value_loss             | 225       |
-----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 59.9     |
|    ep_rew_mean            | -237     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 51       |
|    time_elapsed           | 35154    |
|    total_timesteps        | 3264     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0656   |
|    learning_rate          | 0.0003   |
|    n_updates              | 50       |
|    policy_objective       | 0.972    |
|    value_loss             | 217      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 61.3     |
|    ep_rew_mean            | -238     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 52       |
|    time_elapsed           | 35380    |
|    total_timesteps        | 3328     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0983   |
|    learning_rate          | 0.0003   |
|    n_updates              | 51       |
|    policy_objective       | 0.703    |
|    value_loss             | 285      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 61.3     |
|    ep_rew_mean            | -238     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 53       |
|    time_elapsed           | 35602    |
|    total_timesteps        | 3392     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0776   |
|    learning_rate          | 0.0003   |
|    n_updates              | 52       |
|    policy_objective       | 1.07     |
|    value_loss             | 148      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 61.3     |
|    ep_rew_mean            | -238     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 54       |
|    time_elapsed           | 35812    |
|    total_timesteps        | 3456     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.062    |
|    learning_rate          | 0.0003   |
|    n_updates              | 53       |
|    policy_objective       | 0.595    |
|    value_loss             | 133      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 61.3     |
|    ep_rew_mean            | -238     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 55       |
|    time_elapsed           | 36030    |
|    total_timesteps        | 3520     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0662   |
|    learning_rate          | 0.0003   |
|    n_updates              | 54       |
|    policy_objective       | 0.565    |
|    value_loss             | 133      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 3584     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0588   |
|    learning_rate          | 0.0003   |
|    n_updates              | 55       |
|    policy_objective       | 0.533    |
|    value_loss             | 173      |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 62.8     |
|    ep_rew_mean     | -240     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 56       |
|    time_elapsed    | 39052    |
|    total_timesteps | 3584     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 62.8     |
|    ep_rew_mean            | -240     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 57       |
|    time_elapsed           | 39303    |
|    total_timesteps        | 3648     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0664   |
|    learning_rate          | 0.0003   |
|    n_updates              | 56       |
|    policy_objective       | 0.619    |
|    value_loss             | 76.1     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 64.6     |
|    ep_rew_mean            | -240     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 58       |
|    time_elapsed           | 39484    |
|    total_timesteps        | 3712     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0586   |
|    learning_rate          | 0.0003   |
|    n_updates              | 57       |
|    policy_objective       | 0.555    |
|    value_loss             | 262      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 64.6     |
|    ep_rew_mean            | -240     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 59       |
|    time_elapsed           | 39661    |
|    total_timesteps        | 3776     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0824   |
|    learning_rate          | 0.0003   |
|    n_updates              | 58       |
|    policy_objective       | 1.74     |
|    value_loss             | 87.9     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 66.4     |
|    ep_rew_mean            | -241     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 60       |
|    time_elapsed           | 39872    |
|    total_timesteps        | 3840     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0521   |
|    learning_rate          | 0.0003   |
|    n_updates              | 59       |
|    policy_objective       | 0.63     |
|    value_loss             | 108      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 66.4     |
|    ep_rew_mean            | -241     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 61       |
|    time_elapsed           | 40059    |
|    total_timesteps        | 3904     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0561   |
|    learning_rate          | 0.0003   |
|    n_updates              | 60       |
|    policy_objective       | 0.563    |
|    value_loss             | 189      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 66.4     |
|    ep_rew_mean            | -241     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 62       |
|    time_elapsed           | 40213    |
|    total_timesteps        | 3968     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0578   |
|    learning_rate          | 0.0003   |
|    n_updates              | 61       |
|    policy_objective       | 0.602    |
|    value_loss             | 150      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 66.4     |
|    ep_rew_mean            | -241     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 63       |
|    time_elapsed           | 40370    |
|    total_timesteps        | 4032     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0752   |
|    learning_rate          | 0.0003   |
|    n_updates              | 62       |
|    policy_objective       | 1.02     |
|    value_loss             | 71.5     |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -190     |
| time/                     |          |
|    total_timesteps        | 4096     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0682   |
|    learning_rate          | 0.0003   |
|    n_updates              | 63       |
|    policy_objective       | 0.687    |
|    value_loss             | 88.4     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 68.8     |
|    ep_rew_mean     | -242     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 64       |
|    time_elapsed    | 43338    |
|    total_timesteps | 4096     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 68.8     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 65       |
|    time_elapsed           | 43524    |
|    total_timesteps        | 4160     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0626   |
|    learning_rate          | 0.0003   |
|    n_updates              | 64       |
|    policy_objective       | 0.742    |
|    value_loss             | 110      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 68.8     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 66       |
|    time_elapsed           | 43670    |
|    total_timesteps        | 4224     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0909   |
|    learning_rate          | 0.0003   |
|    n_updates              | 65       |
|    policy_objective       | 0.801    |
|    value_loss             | 111      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 70.7     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 67       |
|    time_elapsed           | 43846    |
|    total_timesteps        | 4288     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0595   |
|    learning_rate          | 0.0003   |
|    n_updates              | 66       |
|    policy_objective       | 0.859    |
|    value_loss             | 67       |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 70.7     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 68       |
|    time_elapsed           | 44025    |
|    total_timesteps        | 4352     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0573   |
|    learning_rate          | 0.0003   |
|    n_updates              | 67       |
|    policy_objective       | 0.882    |
|    value_loss             | 146      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 70.7     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 69       |
|    time_elapsed           | 44174    |
|    total_timesteps        | 4416     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0625   |
|    learning_rate          | 0.0003   |
|    n_updates              | 68       |
|    policy_objective       | 0.743    |
|    value_loss             | 118      |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 70.7     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 70       |
|    time_elapsed           | 44333    |
|    total_timesteps        | 4480     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0464   |
|    learning_rate          | 0.0003   |
|    n_updates              | 69       |
|    policy_objective       | 0.43     |
|    value_loss             | 92       |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 70.7     |
|    ep_rew_mean            | -242     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 71       |
|    time_elapsed           | 44447    |
|    total_timesteps        | 4544     |
| train/                    |          |
|    explained_variance     | 1.19e-07 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0412   |
|    learning_rate          | 0.0003   |
|    n_updates              | 70       |
|    policy_objective       | 0.452    |
|    value_loss             | 116      |
----------------------------------------
----------------------------------------
| eval/                     |          |
|    mean_ep_length         | 400      |
|    mean_reward            | -183     |
| time/                     |          |
|    total_timesteps        | 4608     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0795   |
|    learning_rate          | 0.0003   |
|    n_updates              | 71       |
|    policy_objective       | 0.424    |
|    value_loss             | 49.4     |
----------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 73.5     |
|    ep_rew_mean     | -243     |
| time/              |          |
|    fps             | 0        |
|    iterations      | 72       |
|    time_elapsed    | 47342    |
|    total_timesteps | 4608     |
---------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 73.5     |
|    ep_rew_mean            | -243     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 73       |
|    time_elapsed           | 47466    |
|    total_timesteps        | 4672     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0765   |
|    learning_rate          | 0.0003   |
|    n_updates              | 72       |
|    policy_objective       | 0.877    |
|    value_loss             | 67.3     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 76       |
|    ep_rew_mean            | -243     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 74       |
|    time_elapsed           | 47592    |
|    total_timesteps        | 4736     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0713   |
|    learning_rate          | 0.0003   |
|    n_updates              | 73       |
|    policy_objective       | 0.652    |
|    value_loss             | 57.7     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 76       |
|    ep_rew_mean            | -243     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 75       |
|    time_elapsed           | 47737    |
|    total_timesteps        | 4800     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0739   |
|    learning_rate          | 0.0003   |
|    n_updates              | 74       |
|    policy_objective       | 1.56     |
|    value_loss             | 63.8     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 76       |
|    ep_rew_mean            | -243     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 76       |
|    time_elapsed           | 47867    |
|    total_timesteps        | 4864     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0795   |
|    learning_rate          | 0.0003   |
|    n_updates              | 75       |
|    policy_objective       | 1.1      |
|    value_loss             | 83.7     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 76       |
|    ep_rew_mean            | -243     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 77       |
|    time_elapsed           | 47983    |
|    total_timesteps        | 4928     |
| train/                    |          |
|    explained_variance     | 5.96e-08 |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0714   |
|    learning_rate          | 0.0003   |
|    n_updates              | 76       |
|    policy_objective       | 0.496    |
|    value_loss             | 57.4     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 76       |
|    ep_rew_mean            | -243     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 78       |
|    time_elapsed           | 48117    |
|    total_timesteps        | 4992     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0646   |
|    learning_rate          | 0.0003   |
|    n_updates              | 77       |
|    policy_objective       | 0.55     |
|    value_loss             | 51.5     |
----------------------------------------
----------------------------------------
| rollout/                  |          |
|    ep_len_mean            | 78.8     |
|    ep_rew_mean            | -244     |
| time/                     |          |
|    fps                    | 0        |
|    iterations             | 79       |
|    time_elapsed           | 48250    |
|    total_timesteps        | 5056     |
| train/                    |          |
|    explained_variance     | 0        |
|    is_line_search_success | 1        |
|    kl_divergence_loss     | 0.0602   |
|    learning_rate          | 0.0003   |
|    n_updates              | 78       |
|    policy_objective       | 0.686    |
|    value_loss             | 91.6     |
----------------------------------------
```

#### logs_TRPO_0.1_baseline/progress.csv

```
time/iterations,time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,time/fps,time/time_elapsed,train/is_line_search_success,train/explained_variance,train/n_updates,train/policy_objective,train/learning_rate,train/kl_divergence_loss,train/value_loss,eval/mean_reward,eval/mean_ep_length
1,64,15.0,-79.324616,0,500,,,,,,,,,
2,128,23.0,-108.61050750000001,0,979,1.0,-0.027903437614440918,1,0.19661575555801392,0.0003,0.004484458826482296,1137.0498168945312,,
3,192,18.333333333333332,-90.61675100000001,0,1506,1.0,0.0,2,0.39175108075141907,0.0003,0.02306540310382843,1908.49462890625,,
4,256,18.333333333333332,-90.61675100000001,0,1968,1.0,0.0,3,0.6831451058387756,0.0003,0.04668460041284561,2028.1684204101562,,
5,320,35.8,-211.9690216,0,2440,1.0,-1.1920928955078125e-07,4,0.6253384351730347,0.0003,0.04066692292690277,2004.8252075195312,,
6,384,40.857142857142854,-222.43890971428573,0,2938,1.0,0.0,5,0.68387770652771,0.0003,0.032356567680835724,1202.1092163085937,,
7,448,41.111111111111114,-233.07890733333332,0,3452,1.0,0.0,6,0.7242639660835266,0.0003,0.03820443153381348,1290.6505615234375,,
,512,,,,,1.0,0.0,7,0.9414438605308533,0.0003,0.04211948439478874,1806.611181640625,-292.0063798,197.2
8,512,41.111111111111114,-233.07890733333332,0,6695,,,,,,,,,
9,576,40.5,-225.60581910000002,0,7151,1.0,0.0,8,1.5304269790649414,0.0003,0.08556190878152847,1739.12802734375,,
10,640,46.166666666666664,-259.23023083333334,0,7606,1.0,1.7881393432617188e-07,9,0.864545464515686,0.0003,0.042714398354291916,1321.0530883789063,,
11,704,44.714285714285715,-249.10133078571425,0,8047,1.0,0.0,10,0.8796893358230591,0.0003,0.061670929193496704,1052.2167236328125,,
12,768,44.53333333333333,-245.60890593333332,0,8476,1.0,0.0,11,1.096269965171814,0.0003,0.057773906737565994,991.6382568359375,,
13,832,45.1875,-244.6482664375,0,8935,1.0,0.0,12,0.8242993354797363,0.0003,0.05465082824230194,1267.4689086914063,,
14,896,45.529411764705884,-244.66172541176473,0,9308,1.0,1.1920928955078125e-07,13,0.859078049659729,0.0003,0.052126459777355194,1225.0009155273438,,
15,960,44.111111111111114,-236.36139733333337,0,9714,1.0,0.0,14,0.9195904731750488,0.0003,0.05588745325803757,708.3138671875,,
,1024,,,,,1.0,1.1920928955078125e-07,15,1.601607322692871,0.0003,0.06244153156876564,1205.0764892578125,-183.631291,400.0
16,1024,45.2,-241.26244415000002,0,12944,,,,,,,,,
17,1088,47.04545454545455,-244.63806990909094,0,13397,1.0,-1.1920928955078125e-07,16,0.8828146457672119,0.0003,0.04743971675634384,957.7093994140625,,
18,1152,47.04545454545455,-244.63806990909094,0,13790,1.0,0.0,17,0.6955784559249878,0.0003,0.055682770907878876,1020.0104614257813,,
19,1216,46.32,-236.1343776,0,14217,1.0,-1.1920928955078125e-07,18,0.9680713415145874,0.0003,0.051672372967004776,1013.5282409667968,,
20,1280,44.22222222222222,-223.11401292592595,0,14603,1.0,0.0,19,1.0032017230987549,0.0003,0.05082634091377258,383.1236328125,,
21,1344,44.22222222222222,-223.11401292592595,0,14972,1.0,0.0,20,0.7684864401817322,0.0003,0.04961778596043587,931.8091918945313,,
22,1408,44.22222222222222,-223.11401292592595,0,15330,1.0,0.0,21,0.7554737329483032,0.0003,0.05587449297308922,924.734423828125,,
23,1472,46.51724137931034,-230.44027596551726,0,15670,1.0,1.1920928955078125e-07,22,0.838546097278595,0.0003,0.04953068494796753,697.9846862792969,,
,1536,,,,,1.0,5.960464477539063e-08,23,0.7977536916732788,0.0003,0.05204373225569725,569.0057434082031,-187.993961,390.8
24,1536,46.51724137931034,-230.44027596551726,0,18834,,,,,,,,,
25,1600,46.96774193548387,-229.7239831935484,0,19186,1.0,0.0,24,1.1527364253997803,0.0003,0.05571771785616875,877.5208862304687,,
26,1664,48.53125,-234.35415053125,0,19515,1.0,0.0,25,1.010246992111206,0.0003,0.05445433780550957,748.18046875,,
27,1728,48.53125,-234.35415053125,0,19890,1.0,0.0,26,0.6730772256851196,0.0003,0.050613563507795334,557.2301940917969,,
28,1792,48.53125,-234.35415053125,0,20223,1.0,0.0,27,0.6792588233947754,0.0003,0.04949645698070526,964.7550659179688,,
29,1856,50.09090909090909,-240.0146957878788,0,20576,1.0,0.0,28,0.7791696786880493,0.0003,0.06040749326348305,1000.4255126953125,,
30,1920,50.529411764705884,-241.62605802941175,0,20913,1.0,0.0,29,0.8263498544692993,0.0003,0.08240830898284912,685.1993103027344,,
31,1984,50.529411764705884,-241.62605802941175,0,21236,1.0,0.0,30,1.1187825202941895,0.0003,0.05318080261349678,472.9828826904297,,
,2048,,,,,1.0,1.1920928955078125e-07,31,0.7712170481681824,0.0003,0.06123460829257965,590.4312866210937,-184.47016019999998,394.2
32,2048,53.888888888888886,-250.68359716666666,0,24381,,,,,,,,,
33,2112,54.21621621621622,-248.76558864864865,0,24687,1.0,-1.1920928955078125e-07,32,0.6930130124092102,0.0003,0.05649002641439438,508.35393676757815,,
34,2176,54.21621621621622,-248.76558864864865,0,25022,1.0,0.0,33,1.0250016450881958,0.0003,0.06391331553459167,427.64414367675784,,
35,2240,54.21621621621622,-248.76558864864865,0,25333,1.0,-2.384185791015625e-07,34,0.8678762912750244,0.0003,0.06819809973239899,696.34599609375,,
36,2304,54.17948717948718,-243.62346471794874,0,25652,1.0,5.960464477539063e-08,35,1.0024210214614868,0.0003,0.08140278607606888,467.7045440673828,,
37,2368,54.58536585365854,-242.3378350731707,0,25950,1.0,0.0,36,0.6755391955375671,0.0003,0.07190363109111786,311.1546325683594,,
38,2432,54.58536585365854,-242.3378350731707,0,26264,1.0,-1.1920928955078125e-07,37,1.007573127746582,0.0003,0.07925796508789062,256.1240859985352,,
39,2496,55.23255813953488,-240.5300519767442,0,26529,1.0,-1.1920928955078125e-07,38,0.8502604961395264,0.0003,0.07564985752105713,690.2914245605468,,
,2560,,,,,1.0,5.960464477539063e-08,39,0.7935724258422852,0.0003,0.059513919055461884,191.90554504394532,-204.99256839999998,396.0
40,2560,55.25,-237.72140677272728,0,29613,,,,,,,,,
41,2624,54.84444444444444,-234.4003083111111,0,29897,1.0,0.0,40,1.002885341644287,0.0003,0.06450024247169495,292.3947265625,,
42,2688,54.84444444444444,-234.4003083111111,0,30139,1.0,0.0,41,0.9048497676849365,0.0003,0.07298602163791656,334.2795471191406,,
43,2752,56.19565217391305,-236.88777552173914,0,30395,1.0,0.0,42,0.8563683032989502,0.0003,0.06277994811534882,399.37743225097654,,
44,2816,56.255319148936174,-234.3412315531915,0,30676,1.0,0.0,43,0.5214835405349731,0.0003,0.07391564548015594,332.2722564697266,,
45,2880,56.255319148936174,-234.3412315531915,0,30958,1.0,5.960464477539063e-08,44,0.77569580078125,0.0003,0.06694954633712769,524.6182983398437,,
46,2944,56.255319148936174,-234.3412315531915,0,31179,1.0,0.0,45,0.7732858061790466,0.0003,0.07865829765796661,376.33460693359376,,
47,3008,56.255319148936174,-234.3412315531915,0,31384,1.0,-2.384185791015625e-07,46,0.6798321008682251,0.0003,0.06135036051273346,272.72449951171876,,
,3072,,,,,1.0,-1.1920928955078125e-07,47,0.6060301065444946,0.0003,0.07855372130870819,134.87289428710938,-182.50913459999998,400.0
48,3072,58.204081632653065,-235.28053910204082,0,34405,,,,,,,,,
49,3136,58.204081632653065,-235.28053910204082,0,34671,1.0,5.960464477539063e-08,48,1.2535146474838257,0.0003,0.06936758011579514,243.40359344482422,,
50,3200,59.88,-237.07792254,0,34899,1.0,-1.1920928955078125e-07,49,0.6433145999908447,0.0003,0.05956052243709564,224.68515014648438,,
51,3264,59.88,-237.07792254,0,35154,1.0,0.0,50,0.971835732460022,0.0003,0.06558512151241302,216.60579833984374,,
52,3328,61.294117647058826,-238.43362078431372,0,35380,1.0,0.0,51,0.7026162147521973,0.0003,0.09831047803163528,284.933154296875,,
53,3392,61.294117647058826,-238.43362078431372,0,35602,1.0,0.0,52,1.0656967163085938,0.0003,0.07763180881738663,147.89527587890626,,
54,3456,61.294117647058826,-238.43362078431372,0,35812,1.0,5.960464477539063e-08,53,0.5953652858734131,0.0003,0.06201433390378952,133.07013397216798,,
55,3520,61.294117647058826,-238.43362078431372,0,36030,1.0,0.0,54,0.5651680827140808,0.0003,0.06624295562505722,132.78118896484375,,
,3584,,,,,1.0,1.1920928955078125e-07,55,0.5328514575958252,0.0003,0.05882257595658302,172.89970855712892,-183.1240148,400.0
56,3584,62.76923076923077,-239.58214996153845,0,39052,,,,,,,,,
57,3648,62.76923076923077,-239.58214996153845,0,39303,1.0,0.0,56,0.6194781064987183,0.0003,0.06644098460674286,76.12984313964844,,
58,3712,64.60377358490567,-240.36926441509434,0,39484,1.0,1.1920928955078125e-07,57,0.5546965599060059,0.0003,0.05859503895044327,262.1407073974609,,
59,3776,64.60377358490567,-240.36926441509434,0,39661,1.0,0.0,58,1.7379709482192993,0.0003,0.08240017294883728,87.91199264526367,,
60,3840,66.4074074074074,-241.26293564814816,0,39872,1.0,5.960464477539063e-08,59,0.6303677558898926,0.0003,0.05212140828371048,107.69836120605468,,
61,3904,66.4074074074074,-241.26293564814816,0,40059,1.0,0.0,60,0.5626556873321533,0.0003,0.056093670427799225,189.03103637695312,,
62,3968,66.4074074074074,-241.26293564814816,0,40213,1.0,0.0,61,0.6021094918251038,0.0003,0.05775240808725357,150.46033325195313,,
63,4032,66.4074074074074,-241.26293564814816,0,40370,1.0,0.0,62,1.0156031847000122,0.0003,0.07515427470207214,71.49255905151367,,
,4096,,,,,1.0,0.0,63,0.6874299049377441,0.0003,0.0682087242603302,88.40294570922852,-189.7547492,400.0
64,4096,68.78181818181818,-241.5224601818182,0,43338,,,,,,,,,
65,4160,68.78181818181818,-241.5224601818182,0,43524,1.0,0.0,64,0.7416543960571289,0.0003,0.06255760043859482,109.738037109375,,
66,4224,68.78181818181818,-241.5224601818182,0,43670,1.0,0.0,65,0.8009423613548279,0.0003,0.09091097116470337,110.9291763305664,,
67,4288,70.71428571428571,-242.17544535714288,0,43846,1.0,5.960464477539063e-08,66,0.8590681552886963,0.0003,0.05952882766723633,66.97026138305664,,
68,4352,70.71428571428571,-242.17544535714288,0,44025,1.0,0.0,67,0.8817448019981384,0.0003,0.05725359171628952,145.93388824462892,,
69,4416,70.71428571428571,-242.17544535714288,0,44174,1.0,0.0,68,0.7427841424942017,0.0003,0.06253497302532196,118.01910705566407,,
70,4480,70.71428571428571,-242.17544535714288,0,44333,1.0,0.0,69,0.4298950135707855,0.0003,0.046379730105400085,91.97350540161133,,
71,4544,70.71428571428571,-242.17544535714288,0,44447,1.0,1.1920928955078125e-07,70,0.4522067904472351,0.0003,0.041232116520404816,115.50609893798828,,
,4608,,,,,1.0,0.0,71,0.42430752515792847,0.0003,0.07946598529815674,49.38346939086914,-182.9108916,400.0
72,4608,73.54385964912281,-242.6250955263158,0,47342,,,,,,,,,
73,4672,73.54385964912281,-242.6250955263158,0,47466,1.0,5.960464477539063e-08,72,0.8765700459480286,0.0003,0.0764569342136383,67.2722053527832,,
74,4736,76.01724137931035,-243.19732412068967,0,47592,1.0,0.0,73,0.6519477367401123,0.0003,0.07134272903203964,57.67027816772461,,
75,4800,76.01724137931035,-243.19732412068967,0,47737,1.0,0.0,74,1.5628324747085571,0.0003,0.07393705099821091,63.76750564575195,,
76,4864,76.01724137931035,-243.19732412068967,0,47867,1.0,0.0,75,1.0956175327301025,0.0003,0.07946903258562088,83.68355865478516,,
77,4928,76.01724137931035,-243.19732412068967,0,47983,1.0,5.960464477539063e-08,76,0.4961526393890381,0.0003,0.07136911153793335,57.365796661376955,,
78,4992,76.01724137931035,-243.19732412068967,0,48117,1.0,0.0,77,0.549684464931488,0.0003,0.06459289789199829,51.53182830810547,,
79,5056,78.83050847457628,-243.87820377966102,0,48250,1.0,0.0,78,0.6861369609832764,0.0003,0.06016753613948822,91.56399688720703,,
```

#### results/baseline/A2C_0.001_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best A2C model (beta = 0.001): -183.23 +/- 1.11
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/A2C_0.001_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by A2C (beta = 0.001): -183.04 +/- 1.21
Mean reward of the best model trained by A2C (beta = 0.001): -183.15 +/- 1.13
Total A2C (beta = 0.001) runtime: 10.79 hours
Evaluation episodes: 20
```

#### results/baseline/A2C_0.01_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best A2C model (beta = 0.01): -183.33 +/- 1.10
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/A2C_0.01_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by A2C (beta = 0.01): -183.22 +/- 1.10
Mean reward of the best model trained by A2C (beta = 0.01): -183.25 +/- 1.12
Total A2C (beta = 0.01) runtime: 8.45 hours
Evaluation episodes: 20
```

#### results/baseline/A2C_0.0_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best A2C model (beta = 0.0): -183.09 +/- 1.14
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/A2C_0.0_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by A2C (beta = 0.0): -183.16 +/- 0.92
Mean reward of the best model trained by A2C (beta = 0.0): -183.02 +/- 1.14
Total A2C (beta = 0.0) runtime: 10.47 hours
Evaluation episodes: 20
```

#### results/baseline/A2C_0.1_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best A2C model (beta = 0.1): -184.41 +/- 69.65
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/A2C_0.1_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by A2C (beta = 0.1): -182.68 +/- 0.99
Mean reward of the best model trained by A2C (beta = 0.1): -176.45 +/- 61.49
Total A2C (beta = 0.1) runtime: 8.09 hours
Evaluation episodes: 20
```

#### results/baseline/PPO_0.001_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best PPO model (beta = 0.001): -183.29 +/- 1.22
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/PPO_0.001_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by PPO (beta = 0.001): -183.56 +/- 0.79
Mean reward of the best model trained by PPO (beta = 0.001): -183.04 +/- 1.16
Total PPO (beta = 0.001) runtime: 11.62 hours
Evaluation episodes: 20
```

#### results/baseline/PPO_0.01_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best PPO model (beta = 0.01): -183.35 +/- 1.31
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/PPO_0.01_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by PPO (beta = 0.01): -183.17 +/- 0.98
Mean reward of the best model trained by PPO (beta = 0.01): -183.06 +/- 0.77
Total PPO (beta = 0.01) runtime: 11.67 hours
Evaluation episodes: 20
```

#### results/baseline/PPO_0.0_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best PPO model (beta = 0.0): -64.62 +/- 5.60
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/PPO_0.0_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by PPO (beta = 0.0): -56.11 +/- 1.79
Mean reward of the best model trained by PPO (beta = 0.0): -65.41 +/- 4.30
Total PPO (beta = 0.0) runtime: 16.30 hours
Evaluation episodes: 20
```

#### results/baseline/PPO_0.1_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best PPO model (beta = 0.1): -111.35 +/- 8.18
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/PPO_0.1_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by PPO (beta = 0.1): -183.50 +/- 1.31
Mean reward of the best model trained by PPO (beta = 0.1): -110.16 +/- 7.20
Total PPO (beta = 0.1) runtime: 13.46 hours
Evaluation episodes: 20
```

#### results/baseline/TRPO_0.001_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best TRPO model (beta = 0.001): -183.36 +/- 1.20
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/TRPO_0.001_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by TRPO (beta = 0.001): -183.01 +/- 0.99
Mean reward of the best model trained by TRPO (beta = 0.001): -182.93 +/- 1.20
Total TRPO (beta = 0.001) runtime: 17.10 hours
Evaluation episodes: 20
```

#### results/baseline/TRPO_0.01_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best TRPO model (beta = 0.01): -109.03 +/- 90.65
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/TRPO_0.01_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by TRPO (beta = 0.01): -183.37 +/- 1.34
Mean reward of the best model trained by TRPO (beta = 0.01): -90.62 +/- 84.90
Total TRPO (beta = 0.01) runtime: 15.79 hours
Evaluation episodes: 20
```

#### results/baseline/TRPO_0.0_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best TRPO model (beta = 0.0): -183.02 +/- 0.93
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/TRPO_0.0_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by TRPO (beta = 0.0): -182.84 +/- 1.15
Mean reward of the best model trained by TRPO (beta = 0.0): -183.14 +/- 1.00
Total TRPO (beta = 0.0) runtime: 11.23 hours
Evaluation episodes: 20
```

#### results/baseline/TRPO_0.1_out_of_sample_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Out-of-sample mean reward of best TRPO model (beta = 0.1): -183.18 +/- 1.07
Evaluation episodes: 20
Cell lines: random
Diffusion settings: random
Drugs: random
```

#### results/baseline/TRPO_0.1_training_baseline.txt

```
Experiment label: baseline
Environment kwargs: default
Training hyperparameters: default
Mean reward of the last model trained by TRPO (beta = 0.1): -187.32 +/- 3.32
Mean reward of the best model trained by TRPO (beta = 0.1): -183.13 +/- 0.95
Total TRPO (beta = 0.1) runtime: 13.40 hours
Evaluation episodes: 20
```

#### results/baseline/aggregate_metrics_beta_0.001_baseline.csv

```
algo,beta,split,mean_reward,std_reward,experiment,context
PPO,0.001,in_sample,-183.04424135000002,1.1594728422348752,baseline,standard
PPO,0.001,out_of_sample,-183.29472479999998,1.215936047761496,baseline,"cell_lines=random, diffusions=random, drugs=random"
TRPO,0.001,in_sample,-182.92539195,1.2003535651463906,baseline,standard
TRPO,0.001,out_of_sample,-183.36401845,1.1979494587154136,baseline,"cell_lines=random, diffusions=random, drugs=random"
A2C,0.001,in_sample,-183.14955870000003,1.1328069456677123,baseline,standard
A2C,0.001,out_of_sample,-183.23462030000002,1.1115439702692889,baseline,"cell_lines=random, diffusions=random, drugs=random"
```

#### results/baseline/aggregate_metrics_beta_0.01_baseline.csv

```
algo,beta,split,mean_reward,std_reward,experiment,context
PPO,0.01,in_sample,-183.06073505,0.7741866244893063,baseline,standard
PPO,0.01,out_of_sample,-183.34875425,1.3101357108541791,baseline,"cell_lines=random, diffusions=random, drugs=random"
TRPO,0.01,in_sample,-90.6227288,84.90233097130373,baseline,standard
TRPO,0.01,out_of_sample,-109.02980755000002,90.65039824885484,baseline,"cell_lines=random, diffusions=random, drugs=random"
A2C,0.01,in_sample,-183.24970785,1.1191620744173878,baseline,standard
A2C,0.01,out_of_sample,-183.32754265,1.0996380132428734,baseline,"cell_lines=random, diffusions=random, drugs=random"
```

#### results/baseline/aggregate_metrics_beta_0.0_baseline.csv

```
algo,beta,split,mean_reward,std_reward,experiment,context
TRPO,0.0,in_sample,-183.137246,0.9993568168750339,baseline,standard
TRPO,0.0,out_of_sample,-183.0176288,0.9258979513447796,baseline,"cell_lines=random, diffusions=random, drugs=random"
PPO,0.0,in_sample,-65.41456170000001,4.3023773648114485,baseline,standard
PPO,0.0,out_of_sample,-64.6204703,5.5985069618575904,baseline,"cell_lines=random, diffusions=random, drugs=random"
A2C,0.0,in_sample,-183.02007495,1.142244227826844,baseline,standard
A2C,0.0,out_of_sample,-183.09081569999998,1.1390763696889732,baseline,"cell_lines=random, diffusions=random, drugs=random"
```

#### results/baseline/aggregate_metrics_beta_0.1_baseline.csv

```
algo,beta,split,mean_reward,std_reward,experiment,context
PPO,0.1,in_sample,-110.15758165000003,7.196616204796746,baseline,standard
PPO,0.1,out_of_sample,-111.3541769,8.184170169724528,baseline,"cell_lines=random, diffusions=random, drugs=random"
TRPO,0.1,in_sample,-183.12622580000001,0.9531631691739638,baseline,standard
TRPO,0.1,out_of_sample,-183.18278195,1.0653470012379767,baseline,"cell_lines=random, diffusions=random, drugs=random"
A2C,0.1,in_sample,-176.44972454999998,61.48613725081005,baseline,standard
A2C,0.1,out_of_sample,-184.41325239999998,69.64735626874041,baseline,"cell_lines=random, diffusions=random, drugs=random"
```

#### results/baseline/runtime_profile_beta_0.001_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
PPO: 11.62 hours
TRPO: 17.10 hours
A2C: 10.79 hours
standard_env_acquisition: 0.17 seconds
oos_env_acquisition: 0.32 seconds
standard_env_acquisition: 0.17 seconds
PPO_runtime: 78196.70 seconds
oos_env_acquisition: 0.13 seconds
TRPO_runtime: 97820.23 seconds
standard_env_acquisition: 0.12 seconds
oos_env_acquisition: 0.06 seconds
A2C_runtime: 55717.83 seconds
Total wall time: 37.20 hours
```

#### results/baseline/runtime_profile_beta_0.01_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
PPO: 11.67 hours
TRPO: 15.79 hours
A2C: 8.45 hours
standard_env_acquisition: 0.39 seconds
oos_env_acquisition: 0.12 seconds
standard_env_acquisition: 0.19 seconds
oos_env_acquisition: 0.13 seconds
PPO_runtime: 78185.78 seconds
TRPO_runtime: 80939.58 seconds
standard_env_acquisition: 0.11 seconds
oos_env_acquisition: 0.06 seconds
A2C_runtime: 47301.49 seconds
Total wall time: 34.86 hours
```

#### results/baseline/runtime_profile_beta_0.0_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
TRPO: 11.23 hours
PPO: 16.30 hours
A2C: 10.47 hours
standard_env_acquisition: 0.19 seconds
oos_env_acquisition: 0.13 seconds
standard_env_acquisition: 0.18 seconds
TRPO_runtime: 76634.11 seconds
oos_env_acquisition: 0.12 seconds
PPO_runtime: 94310.71 seconds
standard_env_acquisition: 0.11 seconds
oos_env_acquisition: 0.07 seconds
A2C_runtime: 54621.05 seconds
Total wall time: 36.46 hours
```

#### results/baseline/runtime_profile_beta_0.1_baseline.txt

```
Runtime profile
Parallel workers: 2
Env cache enabled: False
Device: auto
Env cache hits: 0
Env cache misses: 6
PPO: 13.46 hours
TRPO: 13.40 hours
A2C: 8.09 hours
standard_env_acquisition: 0.26 seconds
standard_env_acquisition: 0.13 seconds
oos_env_acquisition: 0.11 seconds
oos_env_acquisition: 0.12 seconds
PPO_runtime: 83689.98 seconds
TRPO_runtime: 84344.75 seconds
standard_env_acquisition: 0.10 seconds
oos_env_acquisition: 0.07 seconds
A2C_runtime: 46016.56 seconds
Total wall time: 36.03 hours
```

#### results/baseline/tables/aggregate_metrics_beta_0.001_baseline.tex

```
\begin{tabular}{lrlrrll}
\toprule
algo & beta & split & mean_reward & std_reward & experiment & context \\
\midrule
PPO & 0.00 & in_sample & -183.04 & 1.16 & baseline & standard \\
PPO & 0.00 & out_of_sample & -183.29 & 1.22 & baseline & cell_lines=random, diffusions=random, drugs=random \\
TRPO & 0.00 & in_sample & -182.93 & 1.20 & baseline & standard \\
TRPO & 0.00 & out_of_sample & -183.36 & 1.20 & baseline & cell_lines=random, diffusions=random, drugs=random \\
A2C & 0.00 & in_sample & -183.15 & 1.13 & baseline & standard \\
A2C & 0.00 & out_of_sample & -183.23 & 1.11 & baseline & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/aggregate_metrics_beta_0.01_baseline.tex

```
\begin{tabular}{lrlrrll}
\toprule
algo & beta & split & mean_reward & std_reward & experiment & context \\
\midrule
PPO & 0.01 & in_sample & -183.06 & 0.77 & baseline & standard \\
PPO & 0.01 & out_of_sample & -183.35 & 1.31 & baseline & cell_lines=random, diffusions=random, drugs=random \\
TRPO & 0.01 & in_sample & -90.62 & 84.90 & baseline & standard \\
TRPO & 0.01 & out_of_sample & -109.03 & 90.65 & baseline & cell_lines=random, diffusions=random, drugs=random \\
A2C & 0.01 & in_sample & -183.25 & 1.12 & baseline & standard \\
A2C & 0.01 & out_of_sample & -183.33 & 1.10 & baseline & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/aggregate_metrics_beta_0.0_baseline.tex

```
\begin{tabular}{lrlrrll}
\toprule
algo & beta & split & mean_reward & std_reward & experiment & context \\
\midrule
TRPO & 0.00 & in_sample & -183.14 & 1.00 & baseline & standard \\
TRPO & 0.00 & out_of_sample & -183.02 & 0.93 & baseline & cell_lines=random, diffusions=random, drugs=random \\
PPO & 0.00 & in_sample & -65.41 & 4.30 & baseline & standard \\
PPO & 0.00 & out_of_sample & -64.62 & 5.60 & baseline & cell_lines=random, diffusions=random, drugs=random \\
A2C & 0.00 & in_sample & -183.02 & 1.14 & baseline & standard \\
A2C & 0.00 & out_of_sample & -183.09 & 1.14 & baseline & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/aggregate_metrics_beta_0.1_baseline.tex

```
\begin{tabular}{lrlrrll}
\toprule
algo & beta & split & mean_reward & std_reward & experiment & context \\
\midrule
PPO & 0.10 & in_sample & -110.16 & 7.20 & baseline & standard \\
PPO & 0.10 & out_of_sample & -111.35 & 8.18 & baseline & cell_lines=random, diffusions=random, drugs=random \\
TRPO & 0.10 & in_sample & -183.13 & 0.95 & baseline & standard \\
TRPO & 0.10 & out_of_sample & -183.18 & 1.07 & baseline & cell_lines=random, diffusions=random, drugs=random \\
A2C & 0.10 & in_sample & -176.45 & 61.49 & baseline & standard \\
A2C & 0.10 & out_of_sample & -184.41 & 69.65 & baseline & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/best_by_split_beta_0.001_baseline.tex

```
\begin{tabular}{lllrrrl}
\toprule
experiment & split & algo & beta & mean_reward & std_reward & context \\
\midrule
baseline & in_sample & TRPO & 0.00 & -182.93 & 1.20 & standard \\
baseline & out_of_sample & A2C & 0.00 & -183.23 & 1.11 & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/best_by_split_beta_0.01_baseline.tex

```
\begin{tabular}{lllrrrl}
\toprule
experiment & split & algo & beta & mean_reward & std_reward & context \\
\midrule
baseline & in_sample & TRPO & 0.01 & -90.62 & 84.90 & standard \\
baseline & out_of_sample & TRPO & 0.01 & -109.03 & 90.65 & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/best_by_split_beta_0.0_baseline.tex

```
\begin{tabular}{lllrrrl}
\toprule
experiment & split & algo & beta & mean_reward & std_reward & context \\
\midrule
baseline & in_sample & PPO & 0.00 & -65.41 & 4.30 & standard \\
baseline & out_of_sample & PPO & 0.00 & -64.62 & 5.60 & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/baseline/tables/best_by_split_beta_0.1_baseline.tex

```
\begin{tabular}{lllrrrl}
\toprule
experiment & split & algo & beta & mean_reward & std_reward & context \\
\midrule
baseline & in_sample & PPO & 0.10 & -110.16 & 7.20 & standard \\
baseline & out_of_sample & PPO & 0.10 & -111.35 & 8.18 & cell_lines=random, diffusions=random, drugs=random \\
\bottomrule
\end{tabular}
```

#### results/out_of_sample_plan.txt

```
Out-of-sample evaluation targets
Cell lines: random
Drugs: random
Diffusion regimes: random
```


