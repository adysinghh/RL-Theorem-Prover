import os
import torch
from logic_env import LogicEnv
from model import PolicyNet
from utils import load_theorems, pretty_print_trace

# 1. Locate your checkpoints directory
reinforce_dir = "reinforce_results"
ep = sorted([f for f in os.listdir(reinforce_dir) if f.startswith("policy_ep")])[-1]
checkpoint_path = os.path.join(reinforce_dir, ep)

# 2. Load the exact theorem you trained on
theorems = load_theorems("theorems.json")
task = theorems[0]  # or find by task["id"] == "chain_ABC"

# 3. Reconstruct the same env
env = LogicEnv(task["axioms"], task["target"], max_steps=50)
state_dim = env.observation_space.shape[0]

# 4. Reload policy with matching dimensions
hidden_dim = 128   # must match what you used during training
policy = PolicyNet.load(
    checkpoint_path,
    state_dim=state_dim,
    action_dim=1,
    hidden_dim=hidden_dim
)

# 5. Roll out a proof trace
trace = []
state = env.reset()
trace.append(set(env.known_facts))

done = False
while not done:
    s = torch.from_numpy(state).float()
    action, _ = policy.get_action(s)
    state, _, done, _ = env.step(action)
    trace.append(set(env.known_facts))

# 6. Pretty-print
pretty_print_trace(trace)
