#!/usr/bin/env python
import streamlit as st
import numpy as np
from stable_baselines3 import PPO
from logic_env import LogicEnv      # v1: MP only
from logic_env_extended import LogicEnvV2  # v2: all rules
from utils import load_theorems

st.title("RL Theorem Prover Demo")

axioms_input = st.text_area("Axioms (comma-separated)", "A∧B, B→C")
target = st.text_input("Target", "C")
env_version = st.radio("Environment", ["V1 (Modus Ponens)", "V2 (All Rules)"])
policy_type = st.radio("Policy", ["Random", "PPO"])

if st.button("Run Proof"):
    axioms = [a.strip() for a in axioms_input.split(",") if a.strip()]
    if env_version.startswith("V1"):
        env = LogicEnv(axioms, target)
    else:
        env = LogicEnvV2(axioms, target)

    if policy_type == "PPO":
        model = PPO.load(f"ppo_results/ppo_{st.sidebar.text_input('Theorem ID','simple_chain')}")
    trace = []
    state = env.reset()
    trace.append(set(env.known_facts))

    done = False
    while not done:
        if policy_type == "Random":
            action = env.action_space.sample()
        else:
            action, _ = model.predict(state, deterministic=True)
        state, _, done, _ = env.step(int(action))
        trace.append(set(env.known_facts))

    for i, facts in enumerate(trace):
        st.markdown(f"**Step {i}**: {sorted(facts)}")
