import os
import time
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from sb3_contrib import TRPO
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from utils.baselines import (
    BaselineController,
    DrugRotationController,
    HeuristicScheduleController,
    PIDLikeTumorController,
)
from utils.training import train


def _ensure_env_registered() -> None:
    """Register the custom environment once."""

    if "ReactionDiffusion-v0" not in gym.envs.registry:
        gym.envs.registration.register(
            id="ReactionDiffusion-v0",
            entry_point="env.reaction_diffusion:ReactionDiffusionEnv",
            kwargs={"render_mode": "human"},
        )


def _collect_episode(
    policy: Any,
    eval_env: gym.Env,
    seed: int,
    cell_line: int,
    diffusion: List[float],
) -> Tuple[int, Dict[str, List[Any]]]:
    """Run a single episode and collect trajectories for plotting."""

    obs, info = eval_env.reset(seed=seed, options={"cell_line": cell_line, "diffusion": diffusion})
    dose, drug_type, drug, states, episodic_rewards = [], [], [], [], []
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, done, _, info = eval_env.step(action)
        dose.append(action[0:2])
        drug_type.append(action[2])
        for _ in range(action[0] + 1):
            drug.append(0.1 * action[1])
        for counter in range(len(info) - 1):
            states.append(info[counter])
        episodic_rewards.append(reward)
    return info["cell_line"], {
        "dose": dose,
        "drug_type": drug_type,
        "drug": drug,
        "states": states,
        "episodic_rewards": episodic_rewards,
    }


def _save_episode_plots(run_label: str, beta_label: str, cell_line: str, episode_log: Dict[str, List[Any]]) -> None:
    """Persist the standard set of plots for a rollout."""

    dose = episode_log["dose"]
    drug = episode_log["drug"]
    drug_type = episode_log["drug_type"]
    states = episode_log["states"]
    episodic_rewards = episode_log["episodic_rewards"]

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(1, len(dose) + 1), np.array(dose)[:, 0] + 1, "rs", label="Duration")
    plt.plot(np.arange(1, len(dose) + 1), np.array(dose)[:, 1], "g^", label="Dose")
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Action", fontsize=20)
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"./results/{run_label}_{beta_label}_cell_line_{cell_line}_actions.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    drug_extended = list(drug) + [drug[-1]] if drug else []
    plt.step(range(len(drug_extended)), drug_extended, where="post")
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Dose", fontsize=20)
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"./results/{run_label}_{beta_label}_cell_line_{cell_line}_dose.png")
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
    plt.savefig(f"./results/{run_label}_{beta_label}_cell_line_{cell_line}_drug_type.png")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(states, label=["Normal cells", "Tumor cells", "Immune cells", "Chemotherapy"])
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Concentration", fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"./results/{run_label}_{beta_label}_cell_line_{cell_line}_concentrations.png")
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
    plt.savefig(f"./results/{run_label}_{beta_label}_cell_line_{cell_line}_reward.png")
    plt.close()

def evaluate(algos, total_steps, num_steps, beta, number_of_envs, number_of_eval_episodes, seed):
    # Set the style and context to make the plot more "publication-ready"
    sns.set_theme()

    # Create the plot
    plt.figure(figsize=(12, 8))

    for algo in algos:
        # Record the start time
        start_time = time.time()
        # Train the agent
        env, model = train(algo, total_steps, num_steps, beta, number_of_envs, seed)
        # Record the end time
        end_time = time.time()
        # Calculate the total runtime
        total_runtime = (end_time - start_time)/3600
        # Specify the file to save the runtime and performance
        file_path = f'{algo}_{beta}_training.txt'
        # Evaluate the trained agent
        mean_reward_last, std_reward_last = evaluate_policy(model, model.get_env(), n_eval_episodes=number_of_eval_episodes)
        # Load the best model saved by EvalCallback
        best_model_path = f'./logs_{algo}_{beta}/best_model'
        if algo == 'PPO':
            model = PPO.load(best_model_path)
        elif algo == 'TRPO':
            model = TRPO.load(best_model_path)
        elif algo == 'A2C':
            model = A2C.load(best_model_path)
        _ensure_env_registered()
        eval_env = Monitor(gym.make('ReactionDiffusion-v0', render_mode='human'))
        # Evaluate the trained agent
        mean_reward_best, std_reward_best = evaluate_policy(model, eval_env, n_eval_episodes=number_of_eval_episodes)
        # Write the total runtime in seconds and the evaluation results to the file
        with open(file_path, 'w') as file:
            file.write(f"Mean reward of the last model trained by {algo} (beta = {beta}): {mean_reward_last:.2f} +/- {std_reward_last:.2f}\n")
            file.write(f"Mean reward of the best model trained by {algo} (beta = {beta}): {mean_reward_best:.2f} +/- {std_reward_best:.2f}\n")
            file.write(f"Total {algo} (beta = {beta}) runtime: {total_runtime:.2f} hours")
        # Plot the results
        available_cell_lines = range(len(eval_env.env.unwrapped.cancer_cell_lines))
        rollout_cell_lines = list(available_cell_lines)[: number_of_eval_episodes or len(eval_env.env.unwrapped.cancer_cell_lines)]

        for cancer in rollout_cell_lines:
            cell_line, episode_log = _collect_episode(
                policy=model,
                eval_env=eval_env,
                seed=19,
                cell_line=cancer,
                diffusion=[0.001, 0.001, 0.001],
            )
            _save_episode_plots(algo, str(beta), cell_line, episode_log)

        # Load log data from evaluations.npz
        log_path = f'./logs_{algo}_{beta}/evaluations.npz'
        if not os.path.exists(log_path):
            continue

        log_data = np.load(log_path)

        # Access 'results' from the log data
        ep_rewards = log_data['results']  # Shape: (num_evaluations, num_parallel_envs)

        # Compute the mean and standard deviation across the parallel environments
        ep_rew_mean = ep_rewards.mean(axis=1)  # Mean reward across parallel environments
        ep_rew_std = ep_rewards.std(axis=1)    # Standard deviation across parallel environments

        # Convert to pandas Series to use rolling functions
        ep_rew_mean_series = pd.Series(ep_rew_mean)
        ep_rew_std_series = pd.Series(ep_rew_std)

        # Define a window size (e.g., 5 evaluations)
        window_size = 5

        # Apply smoothing to the mean rewards
        smoothed_rewards = ep_rew_mean_series.rolling(window=window_size).mean()
        smoothed_std = ep_rew_std_series.rolling(window=window_size).mean()

        # Plot the smoothed line
        sns.lineplot(x=log_data['timesteps'], y=smoothed_rewards, label=algo)

        # Plot the shaded area for standard deviation
        plt.fill_between(log_data['timesteps'], smoothed_rewards - smoothed_std, smoothed_rewards + smoothed_std, alpha=0.3)
    
    # Add titles and labels with increased font size
    plt.xlabel('Training Steps', fontsize=24)
    plt.ylabel('Episode Reward', fontsize=24)

    # Increase the font size of the tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=26)
    plt.savefig(f'./results/rewards_beta_{beta}.png')
    plt.close()


def evaluate_baselines(
    number_of_eval_episodes: int,
    seed: int,
    diffusion: List[float] = None,
    controllers: List[BaselineController] = None,
) -> None:
    """Run hand-crafted controllers with the same metrics/plots as RL agents."""

    sns.set_theme()
    _ensure_env_registered()
    diffusion = diffusion or [0.001, 0.001, 0.001]
    eval_env = Monitor(gym.make("ReactionDiffusion-v0", render_mode="human"))
    env_unwrapped = eval_env.env.unwrapped

    if controllers is None:
        controllers = [
            HeuristicScheduleController(drug_index=0),
            PIDLikeTumorController(drug_index=0),
            DrugRotationController(num_drugs=len(env_unwrapped.drug_names)),
        ]

    available_cell_lines = range(len(env_unwrapped.cancer_cell_lines))
    rollout_cell_lines = list(available_cell_lines)[: number_of_eval_episodes or len(env_unwrapped.cancer_cell_lines)]

    for controller in controllers:
        rewards: List[float] = []
        for cell_line in rollout_cell_lines:
            controller.reset(cell_line, env_unwrapped)
            _, episode_log = _collect_episode(
                policy=controller,
                eval_env=eval_env,
                seed=seed,
                cell_line=cell_line,
                diffusion=diffusion,
            )
            _save_episode_plots(controller.name, "baseline", cell_line, episode_log)
            rewards.append(sum(episode_log["episodic_rewards"]))
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        with open(f"baseline_{controller.name}_evaluation.txt", "w") as file:
            file.write(
                f"Mean episodic reward of {controller.name}: {mean_reward:.2f} +/- {std_reward:.2f}\n"
            )
