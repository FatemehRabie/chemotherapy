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
from utils.training import train

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
        total_runtime = end_time - start_time
        # Specify the file to save the runtime and performance
        file_path = f'{algo}_{beta}_training.txt'
        # Evaluate the trained agent
        mean_reward_last, std_reward_last = evaluate_policy(model, model.get_env(), n_eval_episodes=number_of_eval_episodes)
        # Load the best model saved by EvalCallback
        best_model_path = f'./logs_{algo}_{beta}/best_model.zip'
        if algo == 'PPO':
            model = PPO.load(best_model_path)
        elif algo == 'TRPO':
            model = TRPO.load(best_model_path)
        elif algo == 'A2C':
            model = A2C.load(best_model_path)
        # Avoid re-registering if the environment is already registered
        if 'ReactionDiffusion-v0' not in gym.envs.registry:
            # Register the custom environment with Gym for easy creation
            gym.envs.registration.register(
                id='ReactionDiffusion-v0',
                entry_point='env.reaction_diffusion:ReactionDiffusionEnv',
                kwargs={'render_mode': 'human'}
            )
        # Create a separate evaluation environment
        eval_env = Monitor(gym.make('ReactionDiffusion-v0', render_mode='human'))
        # Evaluate the trained agent
        mean_reward_best, std_reward_best = evaluate_policy(model, eval_env, n_eval_episodes=number_of_eval_episodes)
        # Write the total runtime in seconds and the evaluation results to the file
        with open(file_path, 'w') as file:
            file.write(f"Mean reward of the last model trained by {algo} (beta = {beta}): {mean_reward_last:.2f} +/- {std_reward_last:.2f}\n")
            file.write(f"Mean reward of the best model trained by {algo} (beta = {beta}): {mean_reward_best:.2f} +/- {std_reward_best:.2f}")
            file.write(f"Total {algo} (beta = {beta}) runtime: {total_runtime:.2f} seconds")
        # Plot the results
        for cancer in range(4):
            obs, info = eval_env.reset(seed = 19, options = {'cell_line': cancer, 'diffusion': [0.001, 0.001, 0.001]})  # Reset the environment to start a new episode
            cell_line = info['cell_line']
            dose, drug_type, drug, states, episodic_rewards = [], [], [], [], [] # Lists to store data for plotting
            done = False
            # Run a single episode
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)  # Take a step in the environment
                # Append the current state and reward to the lists for analysis
                dose.append(action[0:2])
                drug_type.append(action[2])
                for _ in range(action[0]+1):
                    drug.append(0.1*action[1])
                for counter in range(len(info)-1):
                    states.append(info[counter])
                episodic_rewards.append(reward)  # Reward is also wrapped in an extra dimension
            # First Plot: Dose over Steps
            # Plotting test results
            plt.figure(figsize=(15, 6))
            plt.plot(np.arange(1, len(dose) + 1), np.array(dose)[:, 0] + 1, 'rs', label='Duration')
            plt.plot(np.arange(1, len(dose) + 1), np.array(dose)[:, 1], 'g^', label='Dose')
            plt.xlabel('Step', fontsize=20)
            plt.ylabel('Action', fontsize=20)
            # Get the current axes
            ax = plt.gca()
            # Set integer ticks for both axes
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure x-axis has integers
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer labels
            plt.legend(fontsize=20)
            plt.grid(True)
            # Increase the font size of the tick labels
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Save the figure
            plt.savefig(f'./results/{algo}_{beta}_cell_line_{cell_line}_actions.png')
            plt.close()
            # Second Plot: Drug over Time with y-axis floating points from 0.0 to 1.0
            # Plotting test results
            plt.figure(figsize=(12, 8))
            drug_extended = list(drug) + [drug[-1]]  # Extend with the last value to make the plot step-like
            plt.step(range(len(drug_extended)), drug_extended, where='post')
            plt.xlabel('Time', fontsize=20)
            plt.ylabel('Dose', fontsize=20)
            # Get the current axes
            ax = plt.gca()
            # Set integer ticks for both axes
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            plt.grid(True)
            # Increase the font size of the tick labels
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Save the figure
            plt.savefig(f'./results/{algo}_{beta}_cell_line_{cell_line}_dose.png')
            plt.close()
            # Third Plot: Drug Type over Steps with integer labels on both axes
            # Plotting test results
            plt.figure(figsize=(12, 8))
            plt.plot(np.arange(1, len(drug_type) + 1), drug_type, marker='o')
            plt.xlabel('Step', fontsize=20)
            plt.ylabel('Drug Type', fontsize=20)
            # Get the current axes
            ax = plt.gca()
            # Set integer ticks for both axes
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure y-axis has integer labels
            plt.grid(True)
            # Increase the font size of the tick labels
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Save the figure
            plt.savefig(f'./results/{algo}_{beta}_cell_line_{cell_line}_drug_type.png')
            plt.close()
            # Fourth Plot: States over Time
            # Plotting test results
            plt.figure(figsize=(12, 8))
            plt.plot(states, label=['Normal cells', 'Tumor cells', 'Immune cells', 'Chemotherapy'])
            plt.xlabel('Time', fontsize=20)
            plt.ylabel('Concentration', fontsize=20)
            plt.legend(fontsize=20)
            plt.grid(True)
            # Increase the font size of the tick labels
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Save the figure
            plt.savefig(f'./results/{algo}_{beta}_cell_line_{cell_line}_concentrations.png')
            plt.close()
            # Fifth Plot: Rewards over Steps with integer x-axis labels
            # Plotting test results
            plt.figure(figsize=(12, 8))
            plt.plot(np.arange(1, len(episodic_rewards) + 1), episodic_rewards, marker='o')
            plt.xlabel('Step', fontsize=20)
            plt.ylabel('Reward', fontsize=20)
            # Get the current axes
            ax = plt.gca()
            # Set integer ticks for both axes
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # Ensure x-axis has integers
            plt.grid(True)
            # Increase the font size of the tick labels
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Save the figure
            plt.savefig(f'./results/{algo}_{beta}_cell_line_{cell_line}_reward.png')
            plt.close()

        # Load log data from evaluations.npz
        log_data = np.load(f'./logs_{algo}_{beta}/evaluations.npz')

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

    # Show or save the plot
    plt.savefig(f'rewards_beta_{beta}.png')