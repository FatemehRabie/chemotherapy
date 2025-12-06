import os
import time
from itertools import product

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

from utils.training import train

DEFAULT_EVAL_EPISODES = 50


class OutOfSampleEnvWrapper(gym.Wrapper):
    """Force evaluation episodes to use held-out cell lines or diffusion settings."""

    def __init__(self, env, cell_lines=None, diffusions=None):
        super().__init__(env)
        self.cell_line_indices = self._prepare_cell_lines(cell_lines)
        self.diffusion_settings = self._prepare_diffusions(diffusions)
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

    def _build_combinations(self):
        cell_lines = self.cell_line_indices or [None]
        diffusions = self.diffusion_settings or [None]
        return list(product(cell_lines, diffusions))

    def reset(self, *, seed=None, options=None):
        options = options or {}
        cell_line_idx = options.get("cell_line")
        diffusion = options.get("diffusion")
        if cell_line_idx is None or diffusion is None:
            cell_line_idx, diffusion = self._combinations_next(cell_line_idx, diffusion)
        return self.env.reset(seed=seed, options={"cell_line": cell_line_idx, "diffusion": diffusion})

    def _combinations_next(self, cell_line_idx, diffusion):
        combo_cell_line, combo_diffusion = self.combinations[self._combo_cursor]
        self._combo_cursor = (self._combo_cursor + 1) % len(self.combinations)
        return cell_line_idx or combo_cell_line, diffusion or combo_diffusion


def _ensure_results_dir(path="./results"):
    os.makedirs(path, exist_ok=True)
    return path


def _run_episode(model, eval_env, cell_line_idx, diffusion=None, seed=19):
    obs, info = eval_env.reset(seed=seed, options={"cell_line": cell_line_idx, "diffusion": diffusion})
    cell_line = info.get("cell_line", cell_line_idx)
    dose, drug_type, drug, states, episodic_rewards = [], [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        dose.append(action[0:2])
        drug_type.append(action[2])
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
):
    env_kwargs = env_kwargs or {}
    number_of_eval_episodes = number_of_eval_episodes or DEFAULT_EVAL_EPISODES
    sns.set_theme()
    plt.figure(figsize=(12, 8))

    safe_label = experiment_label.replace(" ", "_") if experiment_label else ""
    label_suffix = f"_{safe_label}" if safe_label else ""
    results_dir = _ensure_results_dir(os.path.join("results", safe_label) if safe_label else "results")

    for algo in algos:
        start_time = time.time()
        log_folder_base = f"./logs_{algo}_{beta}{label_suffix}"
        env, model = train(
            algo,
            total_steps,
            num_steps,
            beta,
            number_of_envs,
            seed,
            env_kwargs=env_kwargs,
            log_folder_base=log_folder_base,
        )
        end_time = time.time()
        total_runtime = (end_time - start_time) / 3600

        file_path = os.path.join(results_dir, f"{algo}_{beta}_training{label_suffix}.txt")
        mean_reward_last, std_reward_last = _evaluate_model(model, model.get_env(), number_of_eval_episodes)

        best_model_path = os.path.join(log_folder_base, "best_model")
        if algo == "PPO":
            model = PPO.load(best_model_path)
        elif algo == "TRPO":
            model = TRPO.load(best_model_path)
        elif algo == "A2C":
            model = A2C.load(best_model_path)

        if "ReactionDiffusion-v0" not in gym.envs.registry:
            gym.envs.registration.register(
                id="ReactionDiffusion-v0",
                entry_point="env.reaction_diffusion:ReactionDiffusionEnv",
                kwargs={"render_mode": "human"}
            )

        eval_env = Monitor(gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs))
        mean_reward_best, std_reward_best = _evaluate_model(model, eval_env, number_of_eval_episodes)

        _write_results(
            file_path,
            [
                f"Experiment label: {experiment_label or 'baseline'}",
                f"Environment kwargs: {env_kwargs if env_kwargs else 'default'}",
                f"Mean reward of the last model trained by {algo} (beta = {beta}): {mean_reward_last:.2f} +/- {std_reward_last:.2f}",
                f"Mean reward of the best model trained by {algo} (beta = {beta}): {mean_reward_best:.2f} +/- {std_reward_best:.2f}",
                f"Total {algo} (beta = {beta}) runtime: {total_runtime:.2f} hours",
                f"Evaluation episodes: {number_of_eval_episodes}",
            ],
        )

        for cancer in range(min(4, len(eval_env.env.cancer_cell_lines))):
            cell_line, dose, drug_type, drug, states, episodic_rewards = _run_episode(
                model, eval_env, cell_line_idx=cancer, diffusion=[0.001, 0.001, 0.001], seed=seed
            )
            _plot_episode(algo, beta, cell_line, dose, drug_type, drug, states, episodic_rewards, results_dir)

        log_data = np.load(os.path.join(log_folder_base, "evaluations.npz"))
        ep_rewards = log_data["results"]
        ep_rew_mean = ep_rewards.mean(axis=1)
        ep_rew_std = ep_rewards.std(axis=1)
        ep_rew_mean_series = pd.Series(ep_rew_mean)
        ep_rew_std_series = pd.Series(ep_rew_std)
        window_size = 5
        smoothed_rewards = ep_rew_mean_series.rolling(window=window_size).mean()
        smoothed_std = ep_rew_std_series.rolling(window=window_size).mean()
        sns.lineplot(x=log_data["timesteps"], y=smoothed_rewards, label=algo)
        plt.fill_between(log_data["timesteps"], smoothed_rewards - smoothed_std, smoothed_rewards + smoothed_std, alpha=0.3)

        if out_of_sample:
            oos_cell_lines = out_of_sample.get("cell_lines")
            oos_diffusions = out_of_sample.get("diffusions")
            wrapped_env = OutOfSampleEnvWrapper(gym.make("ReactionDiffusion-v0", render_mode="human", **env_kwargs), oos_cell_lines, oos_diffusions)
            oos_eval_env = Monitor(wrapped_env)
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
                ],
            )

            for cell_line_idx, diffusion in wrapped_env.combinations:
                label = wrapped_env.env.cancer_cell_lines[cell_line_idx] if cell_line_idx is not None else "random_cell_line"
                cell_line, dose, drug_type, drug, states, episodic_rewards = _run_episode(
                    model, oos_eval_env, cell_line_idx=cell_line_idx, diffusion=diffusion, seed=seed
                )
                suffix = f"_oos_{str(diffusion).replace(' ', '')}"
                _plot_episode(algo, beta, label or cell_line, dose, drug_type, drug, states, episodic_rewards, results_dir, suffix=suffix)

    plt.xlabel("Training Steps", fontsize=24)
    plt.ylabel("Episode Reward", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=26)
    _ensure_results_dir(results_dir)
    plt.savefig(os.path.join(results_dir, f"rewards_beta_{beta}{label_suffix}.png"))
