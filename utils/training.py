import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

def train(algo, total_steps, num_steps, beta, number_of_envs, seed):
    # Avoid re-registering if the environment is already registered
    if 'ReactionDiffusion-v0' not in gym.envs.registry:
        # Register the custom environment with Gym for easy creation
        gym.envs.registration.register(
            id='ReactionDiffusion-v0',
            entry_point='env.reaction_diffusion:ReactionDiffusionEnv',
            kwargs={'render_mode': 'human'}
        )
    # Use 'make_vec_env' to create multiple parallel environments
    env = make_vec_env('ReactionDiffusion-v0', n_envs=number_of_envs, seed=seed, env_kwargs = {'render_mode': 'human'})
    # Create an evaluation environment (optional, for monitoring performance)
    eval_env = Monitor(gym.make('ReactionDiffusion-v0', render_mode='human'))
    # Define an evaluation callback to log the performance
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./logs_{algo}_{beta}/',
                                log_path=f'./logs_{algo}_{beta}/', eval_freq=128,
                                deterministic=True, render=False)
    # Initialize the logger to write both to the console and files
    new_logger = configure(folder=f"./logs_{algo}_{beta}/")
    # Initialize the agent
    if algo == 'PPO':
        from stable_baselines3 import PPO
        model = PPO('MultiInputPolicy', env, n_steps=num_steps, ent_coef = beta, verbose=0, gamma=1.0, seed=seed)
    elif algo == 'TRPO':
        from sb3_contrib import TRPO
        model = TRPO('MultiInputPolicy', env, n_steps=num_steps, target_kl = beta, verbose=0, gamma=1.0, seed=seed)
    elif algo == 'A2C':
        from stable_baselines3 import A2C
        model = A2C('MultiInputPolicy', env, n_steps=num_steps, ent_coef = beta, verbose=0, gamma=1.0, seed=seed)
    else:
        raise NotImplementedError()
    # Set the logger for the model
    model.set_logger(new_logger)
    # Train and evaluate the agent
    model.learn(total_timesteps=total_steps, callback=eval_callback)
    return env, model