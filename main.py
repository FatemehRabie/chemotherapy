import argparse
import json
import pandas as pd

from utils.evaluation import DEFAULT_EVAL_EPISODES, evaluate, run_hyperparameter_search


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate chemotherapy RL agents")
    parser.add_argument("--file", default="./GDSC2_fitted_dose_response_27Oct23.xlsx", help="Path to the raw GDSC dataset")
    parser.add_argument("--params-file", help="Optional JSON file with run parameters")
    parser.add_argument("--algos", nargs="+", help="Algorithms to train (e.g., PPO TRPO A2C)")
    parser.add_argument("--betas", nargs="+", type=float, help="Beta or KL targets to sweep")
    parser.add_argument("--total-steps", type=int, help="Training steps per run")
    parser.add_argument("--num-steps", type=int, help="Rollout steps per update")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, help="Override the number of evaluation episodes")
    parser.add_argument(
        "--out-of-sample-cell-lines",
        nargs="*",
        help="Held-out cell lines for out-of-sample evaluation (names or indices)",
    )
    parser.add_argument(
        "--out-of-sample-diffusions",
        help="JSON list of diffusion triplets for out-of-sample evaluation",
    )
    parser.add_argument("--pinned-cell-line", help="Cell line name or index for single-cell-line experiments")
    return parser.parse_args()


def load_params_file(params_path):
    if not params_path:
        return {}
    with open(params_path) as fh:
        return json.load(fh)


def build_config(args, file_config):
    config = {
        "algos": ["PPO", "TRPO", "A2C", "FixedSchedule", "ProportionalControl", "RandomPolicy"],
        "betas": [0.0, 0.001, 0.01, 0.1],
        "total_steps": 40000,
        "num_steps": 32,
        "number_of_envs": 4,
        "seed": 19,
        "eval_episodes": DEFAULT_EVAL_EPISODES,
        "hyperparam_search": {"enabled": False, "mode": "grid", "max_trials": 3, "max_seconds": 900},
        "out_of_sample": {"cell_lines": None, "diffusions": None},
        "pinned_cell_line": None,
        "experiments": None,
    }

    config.update(file_config)

    if args.algos:
        config["algos"] = args.algos
    if args.betas:
        config["betas"] = args.betas
    if args.total_steps:
        config["total_steps"] = args.total_steps
    if args.num_steps:
        config["num_steps"] = args.num_steps
    if args.num_envs:
        config["number_of_envs"] = args.num_envs
    if args.seed:
        config["seed"] = args.seed
    if args.eval_episodes:
        config["eval_episodes"] = args.eval_episodes
    if args.pinned_cell_line:
        config["pinned_cell_line"] = args.pinned_cell_line

    out_sample = config.get("out_of_sample", {}) or {}
    if args.out_of_sample_cell_lines is not None:
        out_sample["cell_lines"] = args.out_of_sample_cell_lines
    if args.out_of_sample_diffusions:
        out_sample["diffusions"] = json.loads(args.out_of_sample_diffusions)
    config["out_of_sample"] = out_sample
    return config


def process_and_evaluate(file_path, config):
    df = pd.read_excel(file_path)
    df_subset = df[["CELL_LINE_NAME", "DRUG_NAME", "AUC"]]

    pivot_table = df_subset.pivot_table(index="CELL_LINE_NAME", columns="DRUG_NAME", values="AUC", aggfunc="size")
    complete_rows_mask = pivot_table.notna().all(axis=1)
    valid_cell_lines = pivot_table[complete_rows_mask].index
    filtered_df = df_subset[df_subset["CELL_LINE_NAME"].isin(valid_cell_lines)]
    df_no_duplicates = filtered_df.groupby(["CELL_LINE_NAME", "DRUG_NAME"], as_index=False).agg({"AUC": "mean"})
    output_file_path_no_duplicates = "./Filtered_GDSC2_No_Duplicates_Averaged.xlsx"
    df_no_duplicates.to_excel(output_file_path_no_duplicates, index=False)

    valid_cell_line = valid_cell_lines[0] if len(valid_cell_lines) else None
    experiments = config.get("experiments") or [
        {"name": "baseline", "env_options": {}},
        {"name": "no-noise", "env_options": {"observation_noise": 0.0, "process_noise": 0.0}},
    ]

    pinned_cell_line = config.get("pinned_cell_line", valid_cell_line) or valid_cell_line
    single_line_present = any(exp.get("name") == "single-cell-line" for exp in experiments)
    if pinned_cell_line is not None and not single_line_present:
        experiments.append({"name": "single-cell-line", "env_options": {"cell_line": pinned_cell_line}})

    eval_episodes = config.get("eval_episodes", DEFAULT_EVAL_EPISODES)
    for beta in config["betas"]:
        for experiment in experiments:
            env_options = experiment.get("env_options")
            overrides = run_hyperparameter_search(
                config["algos"],
                config,
                beta,
                experiment.get("name"),
                env_kwargs=env_options,
                search_config=config.get("hyperparam_search"),
            )
            evaluate(
                config["algos"],
                total_steps=config["total_steps"],
                num_steps=config["num_steps"],
                beta=beta,
                number_of_envs=config["number_of_envs"],
                number_of_eval_episodes=eval_episodes,
                seed=config["seed"],
                out_of_sample=config.get("out_of_sample"),
                env_kwargs=env_options,
                experiment_label=experiment.get("name"),
                training_overrides_by_algo=overrides,
            )


def main():
    args = parse_args()
    file_config = load_params_file(args.params_file)
    config = build_config(args, file_config)
    process_and_evaluate(args.file, config)


if __name__ == "__main__":
    main()
