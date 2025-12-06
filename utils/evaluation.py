import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from sb3_contrib import TRPO
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from utils.training import train, build_run_suffix


def evaluate(
    algos,
    total_steps,
    num_steps,
    beta,
    number_of_envs,
    number_of_eval_episodes,
    seed,
    learning_rate=None,
    summary_log_path=None,
):
    """Train and evaluate algorithms for a single hyperparameter configuration."""
    sns.set_theme()

    plt.figure(figsize=(12, 8))
    summary_records = []

    os.makedirs("results", exist_ok=True)

    for algo in algos:
        run_suffix = build_run_suffix(beta, num_steps, learning_rate)
        start_time = time.time()
        env, model = train(algo, total_steps, num_steps, beta, number_of_envs, seed, learning_rate)
        end_time = time.time()
        total_runtime = (end_time - start_time) / 3600

        training_log_path = f"{algo}_{run_suffix}_training.txt"
        mean_reward_last, std_reward_last = evaluate_policy(
            model, model.get_env(), n_eval_episodes=number_of_eval_episodes
        )

        best_model_path = f"./logs_{algo}_{run_suffix}/best_model"
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
                kwargs={"render_mode": "human"},
            )

        eval_env = Monitor(gym.make("ReactionDiffusion-v0", render_mode="human"))
        mean_reward_best, std_reward_best = evaluate_policy(
            model, eval_env, n_eval_episodes=number_of_eval_episodes
        )

        with open(training_log_path, "w") as file:
            file.write(
                f"Mean reward of the last model trained by {algo} (beta = {beta}, n_steps = {num_steps}, lr = {learning_rate}): "
                f"{mean_reward_last:.2f} +/- {std_reward_last:.2f}\n"
            )
            file.write(
                f"Mean reward of the best model trained by {algo} (beta = {beta}, n_steps = {num_steps}, lr = {learning_rate}): "
                f"{mean_reward_best:.2f} +/- {std_reward_best:.2f}\n"
            )
            file.write(
                f"Total {algo} (beta = {beta}, n_steps = {num_steps}, lr = {learning_rate}) runtime: {total_runtime:.2f} hours"
            )

        for cancer in range(4):
            obs, info = eval_env.reset(
                seed=19, options={"cell_line": cancer, "diffusion": [0.001, 0.001, 0.001]}
            )
            cell_line = info["cell_line"]
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
            plt.savefig(f"./results/{algo}_{run_suffix}_cell_line_{cell_line}_actions.png")
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
            plt.savefig(f"./results/{algo}_{run_suffix}_cell_line_{cell_line}_dose.png")
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
            plt.savefig(f"./results/{algo}_{run_suffix}_cell_line_{cell_line}_drug_type.png")
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(states, label=["Normal cells", "Tumor cells", "Immune cells", "Chemotherapy"])
            plt.xlabel("Time", fontsize=20)
            plt.ylabel("Concentration", fontsize=20)
            plt.legend(fontsize=20)
            plt.grid(True)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(f"./results/{algo}_{run_suffix}_cell_line_{cell_line}_concentrations.png")
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
            plt.savefig(f"./results/{algo}_{run_suffix}_cell_line_{cell_line}_reward.png")
            plt.close()

        log_data = np.load(f"./logs_{algo}_{run_suffix}/evaluations.npz")
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

        summary_records.append(
            {
                "algo": algo,
                "beta": beta,
                "n_steps": num_steps,
                "learning_rate": learning_rate if learning_rate is not None else "default",
                "mean_reward_last": mean_reward_last,
                "mean_reward_best": mean_reward_best,
                "runtime_hours": total_runtime,
                "total_steps": total_steps,
            }
        )

    plt.xlabel("Training Steps", fontsize=24)
    plt.ylabel("Episode Reward", fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=26)

    if summary_log_path:
        summary_df = pd.DataFrame(summary_records)
        os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)
        header = not os.path.exists(summary_log_path)
        summary_df.to_csv(summary_log_path, mode="a", index=False, header=header)

    plt.savefig(f"rewards_{build_run_suffix(beta, num_steps, learning_rate)}.png")
    return summary_records
