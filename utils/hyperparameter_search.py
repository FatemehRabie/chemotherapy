import itertools
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.evaluation import evaluate


def _ensure_results_dir() -> Path:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _build_config_grid(
    beta_values: Sequence[float],
    n_step_values: Sequence[int],
    learning_rates: Sequence[float],
    sample_size: int | None = None,
) -> List[dict]:
    grid = [
        {"beta": b, "n_steps": n, "learning_rate": lr}
        for b, n, lr in itertools.product(beta_values, n_step_values, learning_rates)
    ]
    if sample_size and sample_size < len(grid):
        return random.sample(grid, sample_size)
    return grid


def run_hyperparameter_search(
    algos: Iterable[str],
    total_steps: int,
    beta_values: Sequence[float],
    n_step_values: Sequence[int],
    learning_rates: Sequence[float],
    number_of_envs: int,
    number_of_eval_episodes: int,
    seed: int,
    sample_size: int | None = None,
):
    """
    Run a small hyperparameter search and summarize results.

    A simple grid is built from the provided beta, n_steps and learning-rate values.
    Optionally a random subset of ``sample_size`` combinations is sampled to keep the
    sweep light-weight.
    """

    results_dir = _ensure_results_dir()
    summary_path = results_dir / "hyperparam_search_summary.csv"
    search_grid = _build_config_grid(beta_values, n_step_values, learning_rates, sample_size)
    all_records = []

    for config in search_grid:
        records = evaluate(
            algos,
            total_steps=total_steps,
            num_steps=config["n_steps"],
            beta=config["beta"],
            number_of_envs=number_of_envs,
            number_of_eval_episodes=number_of_eval_episodes,
            seed=seed,
            learning_rate=config["learning_rate"],
            summary_log_path=str(summary_path),
        )
        all_records.extend(records)

    summary_df = pd.DataFrame(all_records)
    summary_df.to_csv(summary_path, index=False)

    summary_df["config"] = summary_df.apply(
        lambda row: f"n={row['n_steps']}, lr={row['learning_rate']}, beta={row['beta']}", axis=1
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(data=summary_df, x="config", y="mean_reward_best", hue="algo")
    plt.title("Mean reward by hyperparameter configuration", fontsize=16)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(results_dir / "hyperparam_reward_comparison.png")
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=summary_df,
        x="runtime_hours",
        y="mean_reward_best",
        hue="algo",
        style="config",
        s=120,
    )
    plt.title("Runtime vs. reward across hyperparameter settings", fontsize=16)
    plt.tight_layout()
    plt.savefig(results_dir / "hyperparam_runtime_vs_reward.png")
    plt.close()

    return summary_df

