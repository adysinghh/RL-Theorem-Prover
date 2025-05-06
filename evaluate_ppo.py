#!/usr/bin/env python
import argparse, os, json
import numpy as np
from stable_baselines3 import PPO
from logic_env_extended import LogicEnvV2
from utils import load_theorems, pretty_print_trace

def evaluate(task, model_path, episodes, max_steps, output_dir):
    # Env for metrics
    eval_env = LogicEnvV2(task["axioms"], task["target"], max_steps=max_steps)
    model = PPO.load(model_path, env=eval_env)

    successes, total_reward, total_steps = 0, 0.0, 0
    for _ in range(episodes):
        state = eval_env.reset()
        done = False
        ep_reward, ep_steps = 0.0, 0
        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, rew, done, _ = eval_env.step(int(action))
            ep_reward += rew
            ep_steps += 1
        total_reward += ep_reward
        total_steps += ep_steps
        if task["target"] in eval_env.known_facts:
            successes += 1

    # One fresh env for trace
    trace_env = LogicEnvV2(task["axioms"], task["target"], max_steps=max_steps)
    state = trace_env.reset()
    trace = [set(trace_env.known_facts)]
    done = False
    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, _, done, _ = trace_env.step(int(action))
        trace.append(set(trace_env.known_facts))

    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        "task_id": task["id"],
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes
    }
    with open(os.path.join(output_dir, f"metrics_{task['id']}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    txt = "\n".join(f"[Step {i}] Known: {sorted(facts)}" for i, facts in enumerate(trace))
    with open(os.path.join(output_dir, f"trace_{task['id']}.txt"), "w") as f:
        f.write(txt)

    print(f"Evaluated {task['id']} → {metrics}")
    print(f"Trace saved to {output_dir}/trace_{task['id']}.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theorems", type=str, default="theorems.json")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--theorem_id", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="ppo_eval")
    args = parser.parse_args()

    theorems = load_theorems(args.theorems)
    tasks = ( [t for t in theorems if t["id"]==args.theorem_id]
              if args.theorem_id else theorems )

    for task in tasks:
        evaluate(task, args.model_path, args.episodes, args.max_steps, args.output_dir)

if __name__ == "__main__":
    main()
