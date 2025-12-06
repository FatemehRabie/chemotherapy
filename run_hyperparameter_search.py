"""Run a lightweight grid search over core hyperparameters."""

from utils.hyperparameter_search import run_hyperparameter_search


if __name__ == "__main__":
    run_hyperparameter_search(
        algos=["PPO", "TRPO", "A2C"],
        total_steps=20000,
        beta_values=[0.01, 0.1],
        n_step_values=[16, 32],
        learning_rates=[3e-4, 1e-3],
        number_of_envs=2,
        number_of_eval_episodes=5,
        seed=19,
        sample_size=None,
    )
