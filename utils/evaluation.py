import time
from stable_baselines3.common.evaluation import evaluate_policy
from utils.training import train

def evaluate(algos, total_steps, num_steps, number_of_envs, number_of_eval_episodes, seed):
    for algo in algos:
        # Record the start time
        start_time = time.time()
        # Train the agent
        env, model = train(algo, total_steps, num_steps, number_of_envs, seed)
        # Record the end time
        end_time = time.time()
        # Calculate the total runtime
        total_runtime = end_time - start_time
        # Specify the file to save the runtime and performance
        file_path = f'{algo}_training.txt'
        # Evaluate the trained agent
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=number_of_eval_episodes)
        # Write the total runtime in seconds and the evaluation results to the file
        with open(file_path, 'w') as file:
            file.write(f"Mean reward of {algo}: {mean_reward:.2f} +/- {std_reward:.2f}\n")
            file.write(f"Total {algo} runtime: {total_runtime:.2f} seconds")