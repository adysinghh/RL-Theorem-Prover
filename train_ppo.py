#!/usr/bin/env python
import argparse, os
from stable_baselines3 import PPO
from logic_env_extended import LogicEnvV2
from utils import load_theorems

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorems", default="theorems.json")
    parser.add_argument("--theorem_id", default=None,
                        help="ID from theorems.json")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--model_dir", default="ppo_results")
    parser.add_argument("--tb_log", default="ppo_tb")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    theorems = load_theorems(args.theorems)
    if args.theorem_id:
        theorems = [t for t in theorems if t["id"] == args.theorem_id]
        if not theorems:
            raise ValueError(f"No theorem with id {args.theorem_id}")
    task = theorems[0]

    env = LogicEnvV2(task["axioms"], task["target"], max_steps=args.max_steps)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.tb_log)
    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(args.model_dir, f"ppo_{task['id']}"))
    print(f"Saved PPO model for {task['id']}")

if __name__ == "__main__":
    main()
