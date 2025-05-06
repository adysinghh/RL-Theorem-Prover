import argparse
import json
import os
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from logic_env import LogicEnv
from utils import load_theorems


def run_random_episode(env: LogicEnv, max_steps: int) -> Dict[str, float]:
    """
    Run a single episode with a random policy (only action 0).

    Args:
        env: Initialized LogicEnv instance
        max_steps: Maximum steps per episode

    Returns:
        Dict containing total_reward, success (0 or 1), and steps
    """
    obs = env.reset()
    total_reward = 0.0
    success = 0
    for step in range(1, max_steps + 1):
        obs, reward, done, info = env.step(0)
        total_reward += reward
        if done:
            success = 1 if env.target in env.known_facts else 0
            return {"total_reward": total_reward, "success": success, "steps": step}
    # Episode ended without proving target
    return {"total_reward": total_reward, "success": success, "steps": max_steps}


def evaluate_baseline(theorems: List[Dict], episodes: int, max_steps: int) -> Dict[str, List]:
    """
    Evaluate random baseline on each theorem.

    Args:
        theorems: List of theorem dicts
        episodes: Number of episodes per theorem
        max_steps: Max steps per episode

    Returns:
        Dictionary mapping metric names to lists (aligned with theorems order)
    """
    success_rates = []
    avg_rewards = []
    avg_steps = []

    for thm in tqdm(theorems, desc="Evaluating Theorems"):
        env = LogicEnv(thm["axioms"], thm["target"], max_steps=max_steps)
        rewards = []
        successes = []
        steps_list = []
        for _ in range(episodes):
            result = run_random_episode(env, max_steps)
            rewards.append(result["total_reward"])
            successes.append(result["success"])
            steps_list.append(result["steps"])
        success_rates.append(np.mean(successes))
        avg_rewards.append(np.mean(rewards))
        avg_steps.append(np.mean(steps_list))

    return {
        "ids": [thm["id"] for thm in theorems],
        "success_rate": success_rates,
        "avg_reward": avg_rewards,
        "avg_steps": avg_steps,
    }


def plot_metrics(metrics: Dict[str, List], out_dir: str) -> None:
    """
    Plot and save baseline metrics.

    Args:
        metrics: Dict from evaluate_baseline
        out_dir: Directory to save plots
    """
    os.makedirs(out_dir, exist_ok=True)
    ids = metrics["ids"]

    # Success rate bar chart
    plt.figure()
    plt.bar(ids, metrics["success_rate"])
    plt.xlabel("Theorem ID")
    plt.ylabel("Success Rate")
    plt.title("Random Policy Success Rate")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "success_rate.png"))
    plt.close()

    # Avg reward
    plt.figure()
    plt.bar(ids, metrics["avg_reward"])
    plt.xlabel("Theorem ID")
    plt.ylabel("Average Reward")
    plt.title("Random Policy Average Reward")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_reward.png"))
    plt.close()

    # Avg steps
    plt.figure()
    plt.bar(ids, metrics["avg_steps"])
    plt.xlabel("Theorem ID")
    plt.ylabel("Average Steps")
    plt.title("Random Policy Average Steps to Termination")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "avg_steps.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a random Modus Ponens agent on a set of theorems."
    )
    parser.add_argument(
        "--theorems", type=str, required=True,
        help="Path to theorems.json file"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of episodes per theorem"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--output_dir", type=str, default="baseline_results",
        help="Directory to save metric plots"
    )
    args = parser.parse_args()

    theorems = load_theorems(args.theorems)
    metrics = evaluate_baseline(theorems, args.episodes, args.max_steps)

    # Print summary
    print("\nBaseline Results:")
    print(f"{'ID':<20}{'SuccessRate':<15}{'AvgReward':<15}{'AvgSteps':<15}")
    for i, thm_id in enumerate(metrics['ids']):
        print(f"{thm_id:<20}{metrics['success_rate'][i]:<15.2f}"
              f"{metrics['avg_reward'][i]:<15.2f}"
              f"{metrics['avg_steps'][i]:<15.2f}")

    # Plot and save
    plot_metrics(metrics, args.output_dir)
    print(f"\nMetric plots saved to '{args.output_dir}'")

if __name__ == '__main__':
    main()
