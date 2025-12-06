import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# --- Adaptive Beta Callback ---
class AdaptiveBetaCallback(BaseCallback):
    """
    A custom callback to adaptively change the beta parameter (e.g., ent_coef) during training.
    """
    def __init__(self, initial_beta, scheduling_logic_function, verbose=0):
        super(AdaptiveBetaCallback, self).__init__(verbose)
        self.current_beta = initial_beta
        self.scheduling_logic_function = scheduling_logic_function
    
    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        It updates the beta parameter and then returns True to continue training.
        """
        if not super()._on_step():
            return False
        new_beta = self.scheduling_logic_function(
            timesteps=self.num_timesteps,
            current_beta=self.current_beta,
            logger_info=self.logger.name_to_value
        )
        self.current_beta = new_beta
        # Update the model's parameter
        if hasattr(self.model, 'ent_coef'):
            self.model.ent_coef = self.current_beta
        elif hasattr(self.model, 'target_kl'):
            self.model.target_kl = self.current_beta
        # Log the new beta value
        self.logger.record("train/adaptive_beta", self.current_beta)
        # Return True to continue training
        return True

# --- Scheduling logic function ---
def beta_scheduling_algorithm(timesteps, current_beta, logger_info, **kwargs):
    """
    :param timesteps: Current number of timesteps.
    :param current_beta: The current beta value.
    :return: The new beta value.
    """
    if timesteps > 0 and timesteps % 5000 == 0: # adjust beta every 5000 timesteps
        new_beta = current_beta * 0.95 # decay beta
        if "verbose" in kwargs and kwargs["verbose"] > 0:
             print(f"Novel scheduler: At timestep {timesteps}, adaptively changing beta from {current_beta:.4f} to {new_beta:.4f}")
        return max(new_beta, 0.0001) # Ensure beta doesn't get too small or negative
    return current_beta

def train(algo, total_steps, num_steps, beta, number_of_envs, number_of_eval_episodes, seed):
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
    callbacks = []
    log_folder_base = f"./logs_{algo}_{beta}"
    if beta == 0:
        # --- Adaptive beta mode ---
        beta = 0.5  # Initial beta value for adaptive scheduling
        # Define an evaluation callback to log the performance
        adaptive_beta_cb = AdaptiveBetaCallback(
            initial_beta=beta,
            scheduling_logic_function=beta_scheduling_algorithm,
            verbose=1
        )
        callbacks.append(adaptive_beta_cb)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{log_folder_base}/',
        log_path=f'{log_folder_base}/',
        eval_freq=128,
        n_eval_episodes=number_of_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.insert(0, eval_callback)
    # Use a CallbackList to manage all callbacks
    callback_list = CallbackList(callbacks)
    # Initialize the logger to write both to the console and files
    new_logger = configure(folder=log_folder_base)
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
    model.learn(total_timesteps=total_steps, callback=callback_list)
    return env, model