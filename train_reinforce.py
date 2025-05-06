import argparse
import os
import json
from typing import Dict, Any

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

from logic_env import LogicEnv
from model import PolicyNet
from utils import load_theorems


def discount_and_normalize_rewards(rewards, gamma):
    """
    Compute discounted rewards and normalize.
    """
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    discounted = torch.tensor(discounted, dtype=torch.float)
    # Normalize
    if discounted.std() > 1e-8:
        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)
    return discounted


def main():
    parser = argparse.ArgumentParser(
        description="Train a REINFORCE agent on a single theorem task."
    )
    parser.add_argument("--theorems", type=str, required=True,
                        help="Path to theorems.json file.")
    parser.add_argument("--theorem_id", type=str, required=True,
                        help="ID of the theorem to train on.")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes.")
    parser.add_argument("--max_steps", type=int, default=50,
                        help="Maximum steps per episode.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--curiosity", action="store_true",
                        help="Use intrinsic curiosity reward (default off).")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension of policy network.")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Logging interval (episodes).")
    parser.add_argument("--save_interval", type=int, default=200,
                        help="Model checkpoint save interval (episodes).")
    parser.add_argument("--output_dir", type=str, default="reinforce_results",
                        help="Directory to save logs and checkpoints.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and select task
    theorems = load_theorems(args.theorems)
    task = next((t for t in theorems if t["id"] == args.theorem_id), None)
    if task is None:
        raise ValueError(f"Theorem ID '{args.theorem_id}' not found in {args.theorems}.")

    env = LogicEnv(task["axioms"], task["target"], max_steps=args.max_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # should be 1 for Modus Ponens

    # Initialize policy and optimizer
    policy = PolicyNet(state_dim, action_dim, args.hidden_dim)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Stats
    episode_rewards = []
    episode_successes = []
    episode_lengths = []

    # Training loop
    for ep in trange(1, args.episodes + 1, desc="Training REINFORCE"):
        state = env.reset()
        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float()
            logits = policy(state_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze(0)

            next_state, reward, done, info = env.step(action.item())
            # Separate intrinsic/extrinsic for curiosity toggle
            intrinsic = 0.1 * len(info.get("new_lemmas", []))
            extrinsic = reward - intrinsic
            final_reward = reward if args.curiosity else extrinsic

            log_probs.append(log_prob)
            rewards.append(final_reward)
            state = next_state

        # Calculate metrics
        total_reward = sum(rewards)
        success = 1 if task["target"] in env.known_facts else 0
        length = len(rewards)

        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_lengths.append(length)

        # Compute loss and backprop
        discounted = discount_and_normalize_rewards(rewards, args.gamma)
        loss = -torch.sum(torch.stack(log_probs) * discounted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        if ep % args.log_interval == 0:
            avg_r = np.mean(episode_rewards[-args.log_interval:])
            avg_s = np.mean(episode_successes[-args.log_interval:])
            avg_l = np.mean(episode_lengths[-args.log_interval:])
            print(f"Ep {ep:4d} | AvgR: {avg_r:.2f} | Success: {avg_s:.2f} | Len: {avg_l:.2f}")

        # Checkpoint
        if ep % args.save_interval == 0 or ep == args.episodes:
            ckpt_path = os.path.join(args.output_dir, f"policy_ep{ep}.pt")
            policy.save(ckpt_path)

    # Save training metrics
    metrics = {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "episode_lengths": episode_lengths,
        "args": vars(args)
    }
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training complete. Models and logs in '{args.output_dir}'")


if __name__ == "__main__":
    main()
