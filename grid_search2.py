import sys
import random
import time
import argparse
import os
import csv
import numpy as np
import gymnasium as gym
import pickle
from itertools import product
from bettermdptools.utils.blackjack_wrapper import BlackjackWrapper
from bettermdptools.utils.test_env import TestEnv
from bettermdptools.algorithms.planner import Planner
from bettermdptools.algorithms.rl import RL

# Define parameter grids for each algorithm
SEED = 666
N_ITERATIONS = 10000

ql_params = {
    "gamma": np.linspace(0.9, 1.0, 5),
    "init_alpha": np.linspace(0.1, 0.9, 5),
    "min_alpha": np.linspace(0.01, 0.1, 3),
    "alpha_decay_ratio": np.linspace(0.2, 0.8, 4),
    "init_epsilon": np.linspace(0.7, 1.0, 4),
    "min_epsilon": np.linspace(0.01, 0.2, 3),
    "epsilon_decay_ratio": np.linspace(0.6, 0.99, 5),
    "n_episodes": N_ITERATIONS,
}

vi_pi_params = {
    "gamma": np.linspace(0.9, 1, 5),
    "theta": np.logspace(-12, -4, 5),
    "n_iters": N_ITERATIONS,
}


def main():
    # CLI params
    parser = argparse.ArgumentParser(description="Grid search MDP algorithms")
    parser.add_argument(
        "--env",
        type=str,
        default="FrozenLake8x8-v1",
        choices=["FrozenLake8x8-v1", "FrozenLake16x16-v1", "Blackjack-v1"],
        help="Environment ID",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="Q-Learning",
        choices=["Q-Learning", "Value Iteration", "Policy Iteration", "VI", "PI", "QL"],
        help="Algorithm name",
    )
    parser.add_argument(
        "--output_path", type=str, default=".", help="Output path to save results"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    # Create environment
    set_seed(seed=args.seed)

    if args.env == "Blackjack-v1":
        base_env = gym.make("Blackjack-v1", render_mode=None)
        env = BlackjackWrapper(base_env)
    else:
        env = gym.make(args.env)

    # Determine which parameter grid to use
    if args.algorithm in ["Q-Learning", "QL"]:
        param_grid = ql_params
    elif args.algorithm in ["Value Iteration", "VI", "Policy Iteration", "PI"]:
        param_grid = vi_pi_params
    else:
        raise ValueError("Invalid algorithm name")

    # Run grid search
    output_path = os.path.join(args.output_path, args.env)
    best_score, best_params, best_pi = grid_search(
        env, param_grid, algo_name=args.algorithm, output_path=output_path
    )

    print(f"Best score: {best_score}")
    print(f"Best params: {best_params}")

    # Save results
    output_path = os.path.join(
        output_path, f"{args.algorithm.replace(' ', '_')}_policy.pkl"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(best_pi, f)
    print(f"Policy saved to {output_path}")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    gym.utils.seeding.np_random(seed)


def check_ql_convergence(Q_track, threshold=1e-4):
    for i in range(1, len(Q_track)):
        # Calculate the absolute change from the previous iteration to the current one
        delta = np.abs(Q_track[i] - Q_track[i - 1])
        max_change = np.max(delta)
        # Check if the maximum change is below the threshold
        if max_change < threshold:
            return i  
    return None


def check_convergence(V_track, threshold=1e-4):
    for i in range(1, len(V_track)):
        # Calculate the absolute change from the previous iteration to the current one
        delta = np.abs(V_track[i] - V_track[i - 1])
        max_change = np.max(delta)
        # Check if the maximum change is below the threshold
        if max_change < threshold:
            return i
    return None


def grid_search(env, param_grid, algo_name, output_path):
    best_score = -np.inf
    best_params = None
    best_pi = None

    param_names = list(param_grid.keys())
    param_values = [
        param_grid[name]
        for name in param_names
        if name != "n_episodes" and name != "n_iters"
    ]
    n_iters = param_grid.get(
        "n_episodes", param_grid.get("n_iters", N_ITERATIONS)
    )

    tot = np.prod([len(values) for values in param_values])
    output_path = os.path.join(output_path, f"{algo_name}_grid_search_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            param_names + ["cumulative_score", "runtime", "iterations_to_converge"]
        )
        for i, combination in enumerate(product(*param_values)):
            param_combination = dict(zip(param_names, combination))
            param_combination[
                "n_episodes" if "n_episodes" in param_names else "n_iters"
            ] = n_iters
            print(f"[{i+1}/{tot}] {param_combination}")

            start_time = time.time()

            if algo_name in ["Q-Learning", "QL"]:
                Q, V, pi, Q_track, pi_track = RL(env).q_learning(**param_combination)
                iterations_to_converge = check_ql_convergence(Q_track)

            elif algo_name in ["Value Iteration", "VI"]:
                V, V_track, pi = Planner(env.P).value_iteration(**param_combination)
                iterations_to_converge = check_convergence(V_track)

            elif algo_name in ["Policy Iteration", "PI"]:
                V, V_track, pi = Planner(env.P).policy_iteration(**param_combination)
                iterations_to_converge = check_convergence(V_track)

            runtime = time.time() - start_time

            episode_rewards = TestEnv.test_env(env=env, n_iters=100, pi=pi)
            cum_score = np.sum(episode_rewards)
            writer.writerow(
                list(param_combination.values())
                + [cum_score, runtime, iterations_to_converge]
            )

            if cum_score > best_score:
                best_score = cum_score
                best_params = param_combination
                best_pi = pi

    print(f"Saving grid search results to {output_path}")
    return best_score, best_params, best_pi


if __name__ == "__main__":
    main()
