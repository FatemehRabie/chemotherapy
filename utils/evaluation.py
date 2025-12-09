import os
import random
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from baselines import is_baseline, make_baseline_policy
from utils.training import train
from custom_policies import parse_algo_name

DEFAULT_EVAL_EPISODES = 50


@dataclass
class EpisodePlotData:
    algo: str
    beta: float
    cell_line: Any  # noqa: ANN401 - plotting paths accept arbitrary labels
    dose: np.ndarray
    drug_type: np.ndarray
    drug: np.ndarray
    states: np.ndarray
    episodic_rewards: np.ndarray
    results_dir: str
    suffix: str = ""


@dataclass
class EvaluationProfiler:
    cache_hits: int = 0
    cache_misses: int = 0
    steps: List[Tuple[str, float]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, label: str, start: float) -> None:
        self.steps.append((label, time.time() - start))

    def bump_hits(self) -> None:
        self.cache_hits += 1

    def bump_misses(self) -> None:
        self.cache_misses += 1

    def write(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            file.write("Runtime profile\n")
            file.write(f"Total wall time: {(time.time() - self.started_at)/3600:.2f} hours\n")
            file.write(f"Env cache hits: {self.cache_hits}\n")
            file.write(f"Env cache misses: {self.cache_misses}\n")
            for label, duration in self.steps:
                file.write(f"{label}: {duration:.2f} seconds\n")


class EvaluationEnvCache:
    def __init__(self, enabled: bool = True, profiler: Optional[EvaluationProfiler] = None):
        self.enabled = enabled
        self.profiler = profiler
        self._cache: Dict[Hashable, Monitor] = {}

    def get(self, key: Hashable, builder: Callable[[], Monitor]) -> Monitor:
        if self.enabled and key in self._cache:
            if self.profiler:
                self.profiler.bump_hits()
            return self._cache[key]

        env = builder()
        if self.enabled:
            self._cache[key] = env
        if self.profiler:
            self.profiler.bump_misses()
        return env

    def close_all(self) -> None:
        for env in self._cache.values():
            if hasattr(env, "close"):
                env.close()


class OutOfSampleEnvWrapper(gym.Wrapper):
    """Force evaluation episodes to use held-out cell lines, drugs, or diffusion settings."""

    def __init__(self, env, cell_lines=None, diffusions=None, drugs=None):
        super().__init__(env)
        self.cell_line_indices = self._prepare_cell_lines(cell_lines)
        self.diffusion_settings = self._prepare_diffusions(diffusions)
        self.drug_indices = self._prepare_drugs(drugs)
        self.combinations = self._build_combinations()
        self._combo_cursor = 0

    def _prepare_cell_lines(self, cell_lines):
        if cell_lines is None:
            return []
        if not hasattr(self.env, "cancer_cell_lines"):
            raise AttributeError("Underlying environment must expose `cancer_cell_lines` for cell line selection")
        lookup = {name: idx for idx, name in enumerate(self.env.cancer_cell_lines)}
        indices = []
        for value in cell_lines:
            if isinstance(value, str):
                if value not in lookup:
                    raise ValueError(f"Unknown cell line '{value}' requested for out-of-sample evaluation")
                indices.append(lookup[value])
            else:
                indices.append(int(value))
        return indices

    @staticmethod
    def _prepare_diffusions(diffusions):
        if diffusions is None:
            return []
        prepared = []
        for setting in diffusions:
            if len(setting) != 3:
                raise ValueError("Diffusion settings must contain exactly three values")
            prepared.append([float(val) for val in setting])
        return prepared

    def _prepare_drugs(self, drugs):
        if drugs is None:
            return []
        if not hasattr(self.env, "drug_names"):
            raise AttributeError("Underlying environment must expose `drug_names` for drug selection")
        lookup = {name: idx for idx, name in enumerate(self.env.drug_names)}
        indices = []
        for value in drugs:
            if isinstance(value, str):
                if value not in lookup:
                    raise ValueError(f"Unknown drug '{value}' requested for out-of-sample evaluation")
                indices.append(lookup[value])
            else:
                indices.append(int(value))
        return indices

    def _build_combinations(self):
        cell_lines = self.cell_line_indices or [None]
        diffusions = self.diffusion_settings or [None]
        drugs = self.drug_indices or [None]
        return list(product(cell_lines, diffusions, drugs))

    def reset(self, *, seed=None, options=None):
        options = options or {}
        cell_line_idx = options.get("cell_line")
        diffusion = options.get("diffusion")
        drug = options.get("drug")
        if cell_line_idx is None or diffusion is None or drug is None:
            cell_line_idx, diffusion, drug = self._combinations_next(cell_line_idx, diffusion, drug)
        return self.env.reset(seed=seed, options={"cell_line": cell_line_idx, "diffusion": diffusion, "drug": drug})

    def _combinations_next(self, cell_line_idx, diffusion, drug):
        combo_cell_line, combo_diffusion, combo_drug = self.combinations[self._combo_cursor]
        self._combo_cursor = (self._combo_cursor + 1) % len(self.combinations)
        return cell_line_idx or combo_cell_line, diffusion or combo_diffusion, drug or combo_drug


def _ensure_results_dir(path="./results"):
    os.makedirs(path, exist_ok=True)
    return path


def _run_episode(model, eval_env, cell_line_idx, diffusion=None, drug=None, seed=19):
    obs, info = eval_env.reset(seed=seed, options={"cell_line": cell_line_idx, "diffusion": diffusion, "drug": drug})
    cell_line = info.get("cell_line", cell_line_idx)
    dose, drug_type, drug, states, episodic_rewards = [], [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        dose.append(action[0:2])
        applied_drug = getattr(eval_env.unwrapped, "drug_idx", None)
        drug_type.append(applied_drug if applied_drug is not None else action[2])
        for _ in range(action[0] + 1):
            drug.append(0.1 * action[1])
        for counter in range(len(info) - 1):
            states.append(info[counter])
        episodic_rewards.append(reward)
    return cell_line, np.array(dose), np.array(drug_type), np.array(drug), np.array(states), np.array(episodic_rewards)


def _plot_episode(algo, beta, cell_line, dose, drug_type, drug, states, episodic_rewards, results_dir, suffix=""):
    safe_cell_line = str(cell_line).replace(" ", "_")
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(1, len(dose) + 1), dose[:, 0] + 1, "rs", label="Duration")
    plt.plot(np.arange(1, len(dose) + 1), dose[:, 1], "g^", label="Dose")
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Action", fontsize=20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(results_dir, f"{algo}_{beta}_cell_line_{safe_cell_line}_actions{suffix}.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    drug_extended = list(drug) + [drug[-1]]
    plt.step(range(len(drug_extended)), drug_extended, where="post")
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Dose", fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(results_dir, f"{algo}_{beta}_cell_line_{safe_cell_line}_dose{suffix}.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(drug_type) + 1), drug_type, marker="o")
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Drug Type", fontsize=20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(results_dir, f"{algo}_{beta}_cell_line_{safe_cell_line}_drug_type{suffix}.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(states, label=["Normal cells", "Tumor cells", "Immune cells", "Chemotherapy"])
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Concentration", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(results_dir, f"{algo}_{beta}_cell_line_{safe_cell_line}_concentrations{suffix}.png"))
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(episodic_rewards) + 1), episodic_rewards, marker="o")
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Reward", fontsize=20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(results_dir, f"{algo}_{beta}_cell_line_{safe_cell_line}_reward{suffix}.png"))
    plt.close()


def _evaluate_model(model, eval_env, n_eval_episodes):
    return evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)


def _write_results(file_path, lines):
    with open(file_path, "w") as file:
        for line in lines:
            file.write(f"{line}\n")


_ENV_ALREADY_REGISTERED = False


def _ensure_env_registration():
    global _ENV_ALREADY_REGISTERED
    if _ENV_ALREADY_REGISTERED:
        return

    if "ReactionDiffusion-v0" not in gym.envs.registry:
        gym.envs.registration.register(
            id="ReactionDiffusion-v0",
            entry_point="env.reaction_diffusion:ReactionDiffusionEnv",
            kwargs={"render_mode": "human"}
        )

    _ENV_ALREADY_REGISTERED = True


def _make_eval_env(env_kwargs):
    _ensure_env_registration()
    return Monitor(gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs))


def _unwrap_env(env):
    """Return the base environment beneath any gym wrappers."""

    current = env
    while hasattr(current, "env") and getattr(current, "env") is not current:
        current = getattr(current, "env")

    return getattr(current, "unwrapped", current)


def _prepare_search_candidates(search_config, base_num_steps, max_trials, rng):
    base_num_steps = base_num_steps or 32
    learning_rates = search_config.get("learning_rates") or [3e-4, 1e-4]
    n_steps_options = search_config.get("n_steps") or [base_num_steps, max(base_num_steps // 2, 8)]
    entropy_or_kl = search_config.get("entropy_or_kl") or [0.001, 0.01]
    gamma_options = search_config.get("gamma") or [0.95, 0.99]

    mode = search_config.get("mode", "grid")
    if mode not in {"grid", "random"}:
        raise ValueError("Search mode must be 'grid' or 'random'")

    if mode == "random":
        candidates = []
        for _ in range(max_trials):
            candidates.append(
                {
                    "learning_rate": rng.choice(learning_rates),
                    "n_steps": int(rng.choice(n_steps_options)),
                    "entropy_or_kl": rng.choice(entropy_or_kl),
                    "gamma": rng.choice(gamma_options),
                }
            )
        return candidates

    combos = list(product(learning_rates, n_steps_options, entropy_or_kl, gamma_options))
    rng.shuffle(combos)
    combos = combos[:max_trials]
    return [
        {
            "learning_rate": lr,
            "n_steps": int(steps),
            "entropy_or_kl": entropy,
            "gamma": gm,
        }
        for lr, steps, entropy, gm in combos
    ]


def run_hyperparameter_search(
    algos,
    base_config,
    beta,
    experiment_label,
    env_kwargs=None,
    search_config=None,
):
    search_config = search_config or {}
    if not search_config.get("enabled"):
        return {}

    safe_label = experiment_label.replace(" ", "_") if experiment_label else ""
    label_suffix = f"_{safe_label}" if safe_label else ""
    results_dir = _ensure_results_dir(os.path.join("results", safe_label) if safe_label else "results")

    seed = search_config.get("seed", base_config.get("seed", 19))
    rng = random.Random(seed)
    max_trials = max(1, int(search_config.get("max_trials", 5)))
    max_seconds = float(search_config.get("max_seconds", 900))
    start_time = time.time()

    base_num_steps = base_config.get("num_steps")
    candidates = _prepare_search_candidates(search_config, base_num_steps, max_trials, rng)

    eval_episodes = search_config.get("eval_episodes", base_config.get("eval_episodes", DEFAULT_EVAL_EPISODES))
    overrides_by_algo = {}

    for algo in algos:
        if is_baseline(algo):
            continue

        best_result = None
        timed_out = False
        for idx, candidate in enumerate(candidates, start=1):
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                timed_out = True
                break

            trial_seed = seed + idx
            log_folder = f"./logs_{algo}_{beta}_search{label_suffix}_trial{idx}"
            overrides = {
                "learning_rate": candidate["learning_rate"],
                "n_steps": candidate["n_steps"],
                "gamma": candidate["gamma"],
                "seed": trial_seed,
                "log_folder_base": log_folder,
            }
            if algo == "TRPO":
                overrides["target_kl"] = candidate["entropy_or_kl"]
            else:
                overrides["entropy_coef"] = candidate["entropy_or_kl"]

            env, model = train(
                algo,
                base_config.get("total_steps"),
                overrides["n_steps"],
                beta,
                base_config.get("number_of_envs"),
                overrides["seed"],
                env_kwargs=env_kwargs,
                log_folder_base=log_folder,
                learning_rate=overrides.get("learning_rate"),
                gamma=overrides.get("gamma"),
                entropy_coef=overrides.get("entropy_coef"),
                target_kl=overrides.get("target_kl"),
            )
            eval_env = _make_eval_env(env_kwargs or {})
            mean_reward, std_reward = _evaluate_model(model, eval_env, eval_episodes)

            if not best_result or mean_reward > best_result["mean_reward"]:
                best_result = {
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "config": overrides,
                }

        if best_result:
            final_log_folder = f"./logs_{algo}_{beta}{label_suffix}_best"
            overrides_to_use = {**best_result["config"], "log_folder_base": final_log_folder}
            overrides_by_algo[algo] = overrides_to_use
            file_path = os.path.join(results_dir, f"{algo}_{beta}_hyperparam_search{label_suffix}.txt")
            _write_results(
                file_path,
                [
                    f"Experiment label: {experiment_label or 'baseline'}",
                    f"Search mode: {search_config.get('mode', 'grid')} (max trials: {max_trials}, max seconds: {max_seconds})",
                    f"Evaluation episodes: {eval_episodes}",
                    "Best configuration:",
                    f"  learning_rate: {best_result['config'].get('learning_rate')}",
                    f"  n_steps: {best_result['config'].get('n_steps')}",
                    f"  gamma: {best_result['config'].get('gamma')}",
                    f"  entropy_coef: {best_result['config'].get('entropy_coef')}",
                    f"  target_kl: {best_result['config'].get('target_kl')}",
                    f"  seed: {best_result['config'].get('seed')}",
                    f"Mean reward: {best_result['mean_reward']:.2f} +/- {best_result['std_reward']:.2f}",
                    f"Timed out: {timed_out}",
                ],
            )

    return overrides_by_algo


def evaluate(
    algos,
    total_steps,
    num_steps,
    beta,
    number_of_envs,
    number_of_eval_episodes=None,
    seed=19,
    out_of_sample=None,
    env_kwargs=None,
    experiment_label=None,
    training_overrides_by_algo=None,
    parallel_workers=1,
    reuse_eval_envs=True,
    defer_plots=False,
    plot_episode_stride=1,
):
    env_kwargs = env_kwargs or {}
    number_of_eval_episodes = number_of_eval_episodes or DEFAULT_EVAL_EPISODES
    training_overrides_by_algo = training_overrides_by_algo or {}
    plot_episode_stride = max(1, plot_episode_stride)
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(12, 8))

    safe_label = experiment_label.replace(" ", "_") if experiment_label else ""
    label_suffix = f"_{safe_label}" if safe_label else ""
    results_dir = _ensure_results_dir(os.path.join("results", safe_label) if safe_label else "results")
    _ensure_env_registration()

    profiler = EvaluationProfiler()
    reuse_eval_envs = reuse_eval_envs and parallel_workers == 1
    defer_plots = defer_plots or parallel_workers > 1
    env_cache = EvaluationEnvCache(enabled=reuse_eval_envs, profiler=profiler)

    reward_curves = []
    baseline_rewards = []
    aggregate_metrics = []
    plots_to_generate: List[EpisodePlotData] = []
    runtime_records = []

    def _cache_key(env_kwargs_map: Dict, namespace: str, extra_parts: Optional[Iterable] = None) -> Tuple:
        extras = tuple(extra_parts) if extra_parts else tuple()
        return (namespace, tuple(sorted(env_kwargs_map.items())), extras)

    def _get_eval_env(env_kwargs_map: Dict, cache_namespace: str, extra_parts: Optional[Iterable] = None, builder: Optional[Callable[[], Monitor]] = None):
        cache_key = _cache_key(env_kwargs_map, cache_namespace, extra_parts)
        start = time.time()
        env = env_cache.get(cache_key, builder or (lambda: _make_eval_env(env_kwargs_map)))
        profiler.record(f"{cache_namespace}_env_acquisition", start)
        return env

    def _queue_plot(data: EpisodePlotData):
        if defer_plots:
            plots_to_generate.append(data)
        else:
            _plot_episode(
                data.algo,
                data.beta,
                data.cell_line,
                data.dose,
                data.drug_type,
                data.drug,
                data.states,
                data.episodic_rewards,
                data.results_dir,
                suffix=data.suffix,
            )

    out_of_sample = out_of_sample or {}
    oos_cell_lines = out_of_sample.get("cell_lines")
    oos_diffusions = out_of_sample.get("diffusions")
    oos_drugs = out_of_sample.get("drugs")
    oos_context = f"cell_lines={oos_cell_lines if oos_cell_lines else 'random'}, diffusions={oos_diffusions if oos_diffusions else 'random'}, drugs={oos_drugs if oos_drugs else 'random'}"

    def _evaluate_algo(algo: str):
        start_time = time.time()
        base_algo, _ = parse_algo_name(algo)
        local_reward_curves = []
        local_baseline_rewards = []
        local_aggregate_metrics = []
        local_plots: List[EpisodePlotData] = []

        if is_baseline(algo):
            eval_env = _get_eval_env(env_kwargs, "standard")
            model = make_baseline_policy(algo, eval_env.action_space, seed=seed)
            mean_reward, std_reward = _evaluate_model(model, eval_env, number_of_eval_episodes)
            end_time = time.time()
            total_runtime = (end_time - start_time) / 3600
            file_path = os.path.join(results_dir, f"{algo}_{beta}_evaluation{label_suffix}.txt")
            _write_results(
                file_path,
                [
                    f"Experiment label: {experiment_label or 'baseline'}",
                    f"Environment kwargs: {env_kwargs if env_kwargs else 'default'}",
                    f"Mean reward of the {algo} baseline (beta = {beta}): {mean_reward:.2f} +/- {std_reward:.2f}",
                    f"Total {algo} runtime: {total_runtime:.2f} hours",
                    f"Evaluation episodes: {number_of_eval_episodes}",
                ],
            )

            local_aggregate_metrics.append(
                {
                    "algo": algo,
                    "beta": beta,
                    "split": "in_sample",
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "experiment": experiment_label or "baseline",
                    "context": "standard",
                }
            )

            base_env = _unwrap_env(eval_env)
            for cancer in range(0, min(4, len(base_env.cancer_cell_lines)), plot_episode_stride):
                cell_line, dose, drug_type, drug, states, episodic_rewards = _run_episode(
                    model, eval_env, cell_line_idx=cancer, diffusion=[0.001, 0.001, 0.001], seed=seed
                )
                local_plots.append(
                    EpisodePlotData(algo, beta, cell_line, dose, drug_type, drug, states, episodic_rewards, results_dir)
                )

            local_baseline_rewards.append((algo, mean_reward, std_reward))

            if out_of_sample:
                wrapped_env_builder = lambda: Monitor(
                    OutOfSampleEnvWrapper(
                        gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs), oos_cell_lines, oos_diffusions, oos_drugs
                    )
                )
                extra_parts = (tuple(oos_cell_lines or []), tuple(map(tuple, oos_diffusions or [])), tuple(oos_drugs or []))
                oos_eval_env = _get_eval_env(env_kwargs, "oos", extra_parts=extra_parts, builder=wrapped_env_builder)
                oos_mean, oos_std = _evaluate_model(model, oos_eval_env, number_of_eval_episodes)
                oos_file_path = os.path.join(results_dir, f"{algo}_{beta}_out_of_sample{label_suffix}.txt")
                _write_results(
                    oos_file_path,
                    [
                        f"Experiment label: {experiment_label or 'baseline'}",
                        f"Environment kwargs: {env_kwargs if env_kwargs else 'default'}",
                        f"Out-of-sample mean reward of {algo} baseline (beta = {beta}): {oos_mean:.2f} +/- {oos_std:.2f}",
                        f"Evaluation episodes: {number_of_eval_episodes}",
                        f"Cell lines: {oos_cell_lines if oos_cell_lines else 'random'}",
                        f"Diffusion settings: {oos_diffusions if oos_diffusions else 'random'}",
                        f"Drugs: {oos_drugs if oos_drugs else 'random'}",
                    ],
                )

                local_aggregate_metrics.append(
                    {
                        "algo": algo,
                        "beta": beta,
                        "split": "out_of_sample",
                        "mean_reward": oos_mean,
                        "std_reward": oos_std,
                        "experiment": experiment_label or "baseline",
                        "context": oos_context,
                    }
                )

                if hasattr(oos_eval_env, "env") and isinstance(oos_eval_env.env, OutOfSampleEnvWrapper):
                    wrapped_env = oos_eval_env.env
                else:
                    wrapped_env = OutOfSampleEnvWrapper(
                        gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs), oos_cell_lines, oos_diffusions, oos_drugs
                    )
                base_env = _unwrap_env(wrapped_env)

                for idx, (cell_line_idx, diffusion, drug_idx) in enumerate(wrapped_env.combinations):
                    if idx % plot_episode_stride != 0:
                        continue
                    label = base_env.cancer_cell_lines[cell_line_idx] if cell_line_idx is not None else "random_cell_line"
                    cell_line, dose, drug_type, drug, states, episodic_rewards = _run_episode(
                        model, oos_eval_env, cell_line_idx=cell_line_idx, diffusion=diffusion, drug=drug_idx, seed=seed
                    )
                    drug_label = (
                        base_env.drug_names[drug_idx]
                        if drug_idx is not None and hasattr(base_env, "drug_names")
                        else str(drug_idx)
                    )
                    suffix = f"_oos_diff_{str(diffusion).replace(' ', '')}_drug_{drug_label}"
                    local_plots.append(
                        EpisodePlotData(
                            algo,
                            beta,
                            label or cell_line,
                            dose,
                            drug_type,
                            drug,
                            states,
                            episodic_rewards,
                            results_dir,
                            suffix=suffix,
                        )
                    )

            return {
                "reward_curves": local_reward_curves,
                "baseline_rewards": local_baseline_rewards,
                "aggregate_metrics": local_aggregate_metrics,
                "plots": local_plots,
                "runtime": {"algo": algo, "runtime_hours": (time.time() - start_time) / 3600},
            }

        overrides = training_overrides_by_algo.get(algo, {})
        log_folder_base = overrides.get("log_folder_base") or f"./logs_{algo}_{beta}{label_suffix}"
        train_num_steps = overrides.get("n_steps", num_steps)
        train_seed = overrides.get("seed", seed)
        env, model = train(
            algo,
            total_steps,
            train_num_steps,
            beta,
            number_of_envs,
            train_seed,
            env_kwargs=env_kwargs,
            log_folder_base=log_folder_base,
            learning_rate=overrides.get("learning_rate"),
            gamma=overrides.get("gamma"),
            entropy_coef=overrides.get("entropy_coef"),
            target_kl=overrides.get("target_kl"),
        )
        end_time = time.time()
        total_runtime = (end_time - start_time) / 3600

        file_path = os.path.join(results_dir, f"{algo}_{beta}_training{label_suffix}.txt")
        mean_reward_last, std_reward_last = _evaluate_model(model, model.get_env(), number_of_eval_episodes)

        best_model_path = os.path.join(log_folder_base, "best_model")
        if base_algo == "PPO":
            model = PPO.load(best_model_path)
        elif base_algo == "TRPO":
            model = TRPO.load(best_model_path)
        elif base_algo == "A2C":
            model = A2C.load(best_model_path)

        eval_env = _get_eval_env(env_kwargs, "standard")
        mean_reward_best, std_reward_best = _evaluate_model(model, eval_env, number_of_eval_episodes)

        local_aggregate_metrics.append(
            {
                "algo": algo,
                "beta": beta,
                "split": "in_sample",
                "mean_reward": mean_reward_best,
                "std_reward": std_reward_best,
                "experiment": experiment_label or "baseline",
                "context": "standard",
            }
        )

        _write_results(
            file_path,
            [
                f"Experiment label: {experiment_label or 'baseline'}",
                f"Environment kwargs: {env_kwargs if env_kwargs else 'default'}",
                f"Training hyperparameters: {overrides if overrides else 'default'}",
                f"Mean reward of the last model trained by {algo} (beta = {beta}): {mean_reward_last:.2f} +/- {std_reward_last:.2f}",
                f"Mean reward of the best model trained by {algo} (beta = {beta}): {mean_reward_best:.2f} +/- {std_reward_best:.2f}",
                f"Total {algo} (beta = {beta}) runtime: {total_runtime:.2f} hours",
                f"Evaluation episodes: {number_of_eval_episodes}",
            ],
        )

        base_env = _unwrap_env(eval_env)
        for cancer in range(0, min(4, len(base_env.cancer_cell_lines)), plot_episode_stride):
            cell_line, dose, drug_type, drug, states, episodic_rewards = _run_episode(
                model, eval_env, cell_line_idx=cancer, diffusion=[0.001, 0.001, 0.001], seed=seed
            )
            local_plots.append(
                EpisodePlotData(algo, beta, cell_line, dose, drug_type, drug, states, episodic_rewards, results_dir)
            )

        log_data = np.load(os.path.join(log_folder_base, "evaluations.npz"))
        ep_rewards = log_data["results"]
        ep_rew_mean = ep_rewards.mean(axis=1)
        ep_rew_std = ep_rewards.std(axis=1)
        ep_rew_mean_series = pd.Series(ep_rew_mean)
        ep_rew_std_series = pd.Series(ep_rew_std)
        window_size = 5
        smoothed_rewards = ep_rew_mean_series.rolling(window=window_size).mean()
        smoothed_std = ep_rew_std_series.rolling(window=window_size).mean()
        local_reward_curves.append((log_data["timesteps"], smoothed_rewards.to_numpy(), smoothed_std.to_numpy(), algo))

        if out_of_sample:
            wrapped_env_builder = lambda: Monitor(
                OutOfSampleEnvWrapper(
                    gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs), oos_cell_lines, oos_diffusions, oos_drugs
                )
            )
            extra_parts = (tuple(oos_cell_lines or []), tuple(map(tuple, oos_diffusions or [])), tuple(oos_drugs or []))
            oos_eval_env = _get_eval_env(env_kwargs, "oos", extra_parts=extra_parts, builder=wrapped_env_builder)
            oos_mean, oos_std = _evaluate_model(model, oos_eval_env, number_of_eval_episodes)
        oos_file_path = os.path.join(results_dir, f"{algo}_{beta}_out_of_sample{label_suffix}.txt")
        _write_results(
            oos_file_path,
            [
                f"Experiment label: {experiment_label or 'baseline'}",
                f"Environment kwargs: {env_kwargs if env_kwargs else 'default'}",
                f"Out-of-sample mean reward of best {algo} model (beta = {beta}): {oos_mean:.2f} +/- {oos_std:.2f}",
                f"Evaluation episodes: {number_of_eval_episodes}",
                f"Cell lines: {oos_cell_lines if oos_cell_lines else 'random'}",
                f"Diffusion settings: {oos_diffusions if oos_diffusions else 'random'}",
                f"Drugs: {oos_drugs if oos_drugs else 'random'}",
            ],
        )

        local_aggregate_metrics.append(
            {
                "algo": algo,
                "beta": beta,
                "split": "out_of_sample",
                "mean_reward": oos_mean,
                "std_reward": oos_std,
                "experiment": experiment_label or "baseline",
                "context": oos_context,
            }
        )

        if hasattr(oos_eval_env, "env") and isinstance(oos_eval_env.env, OutOfSampleEnvWrapper):
            wrapped_env = oos_eval_env.env
        else:
            wrapped_env = OutOfSampleEnvWrapper(
                gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs), oos_cell_lines, oos_diffusions, oos_drugs
            )
        base_env = _unwrap_env(wrapped_env)

        for idx, (cell_line_idx, diffusion, drug_idx) in enumerate(wrapped_env.combinations):
            if idx % plot_episode_stride != 0:
                continue
            label = base_env.cancer_cell_lines[cell_line_idx] if cell_line_idx is not None else "random_cell_line"
            cell_line, dose, drug_type, drug, states, episodic_rewards = _run_episode(
                model, oos_eval_env, cell_line_idx=cell_line_idx, diffusion=diffusion, drug=drug_idx, seed=seed
            )
            drug_label = (
                base_env.drug_names[drug_idx]
                if drug_idx is not None and hasattr(base_env, "drug_names")
                else str(drug_idx)
            )
            suffix = f"_oos_diff_{str(diffusion).replace(' ', '')}_drug_{drug_label}"
            local_plots.append(
                EpisodePlotData(
                    algo,
                    beta,
                    label or cell_line,
                    dose,
                    drug_type,
                    drug,
                    states,
                    episodic_rewards,
                    results_dir,
                    suffix=suffix,
                )
            )

        profiler.record(f"{algo}_runtime", start_time)

        return {
            "reward_curves": local_reward_curves,
            "baseline_rewards": local_baseline_rewards,
            "aggregate_metrics": local_aggregate_metrics,
            "plots": local_plots,
            "runtime": {"algo": algo, "runtime_hours": total_runtime},
        }

    def _process_result(result):
        reward_curves.extend(result.get("reward_curves", []))
        baseline_rewards.extend(result.get("baseline_rewards", []))
        aggregate_metrics.extend(result.get("aggregate_metrics", []))
        runtime_records.append(result.get("runtime", {}))
        for plot_data in result.get("plots", []):
            _queue_plot(plot_data)

    if parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(_evaluate_algo, algo): algo for algo in algos}
            for future in as_completed(futures):
                _process_result(future.result())
    else:
        for algo in algos:
            _process_result(_evaluate_algo(algo))

    for timesteps, rewards, stds, label in reward_curves:
        sns.lineplot(x=timesteps, y=rewards, label=label, ax=ax)
        ax.fill_between(timesteps, rewards - stds, rewards + stds, alpha=0.3)

    if baseline_rewards:
        x_min, x_max = ax.get_xlim()
        if x_min == x_max:
            x_min, x_max = 0, total_steps
        for label, mean, std in baseline_rewards:
            ax.axhline(mean, linestyle="--", label=f"{label} mean reward")
            ax.fill_between([x_min, x_max], [mean - std, mean - std], [mean + std, mean + std], alpha=0.1)

    ax.set_xlabel("Training Steps", fontsize=24)
    ax.set_ylabel("Episode Reward", fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(fontsize=18)
    _ensure_results_dir(results_dir)
    plt.savefig(os.path.join(results_dir, f"rewards_beta_{beta}{label_suffix}.png"))

    if aggregate_metrics:
        metrics_df = pd.DataFrame(aggregate_metrics)
        metrics_path = os.path.join(results_dir, f"aggregate_metrics_beta_{beta}{label_suffix}.csv")
        metrics_df.to_csv(metrics_path, index=False)

        plt.figure(figsize=(12, 8))
        ax_metrics = sns.barplot(data=metrics_df, x="algo", y="mean_reward", hue="split", errorbar=None)
        for patch, (_, row) in zip(ax_metrics.patches, metrics_df.iterrows()):
            ax_metrics.errorbar(
                patch.get_x() + patch.get_width() / 2,
                row["mean_reward"],
                yerr=row["std_reward"],
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=4,
            )
        ax_metrics.set_ylabel("Mean episode reward", fontsize=16)
        ax_metrics.set_xlabel("Algorithm", fontsize=16)
        ax_metrics.legend(title="Split", fontsize=12)
        ax_metrics.tick_params(axis="both", labelsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"aggregate_rewards_beta_{beta}{label_suffix}.png"))
        plt.close()

    if defer_plots:
        for plot_data in plots_to_generate:
            _plot_episode(
                plot_data.algo,
                plot_data.beta,
                plot_data.cell_line,
                plot_data.dose,
                plot_data.drug_type,
                plot_data.drug,
                plot_data.states,
                plot_data.episodic_rewards,
                plot_data.results_dir,
                suffix=plot_data.suffix,
            )

    runtime_profile_path = os.path.join(results_dir, f"runtime_profile_beta_{beta}{label_suffix}.txt")
    runtime_lines = [
        "Runtime profile",
        f"Parallel workers: {parallel_workers}",
        f"Env cache enabled: {reuse_eval_envs}",
        f"Env cache hits: {profiler.cache_hits}",
        f"Env cache misses: {profiler.cache_misses}",
    ]
    for record in runtime_records:
        runtime_lines.append(
            f"{record.get('algo')}: {record.get('runtime_hours', 0):.2f} hours"
        )
    for label, duration in profiler.steps:
        runtime_lines.append(f"{label}: {duration:.2f} seconds")
    runtime_lines.append(f"Total wall time: {(time.time() - profiler.started_at)/3600:.2f} hours")
    _write_results(runtime_profile_path, runtime_lines)

    env_cache.close_all()

    return {
        "aggregate_metrics": aggregate_metrics,
        "baseline_rewards": baseline_rewards,
        "reward_curves": reward_curves,
        "runtime": runtime_records,
    }
