````markdown
# RL-Driven Theorem Prover

This project is a reinforcement learning system that **learns to prove logic theorems** by applying formal inference rules—just like a curious student exploring proofs step by step.

Inspired by educational AI Research (like Nick Haber's work at Stanford), it turns theorem proving into an environment where agents earn rewards for making valid deductions using **Modus Ponens**, **AND-Elimination**, **OR-Introduction**, and **Resolution**.

---

## What It Does

- Represents logic problems as **RL environments** (Gym-style)
- Lets agents **learn proof strategies** through trial and error
- Uses **intrinsic rewards** for discovering intermediate steps
- Supports **Random baselines** and **PPO-based agents**
- Comes with a full **Streamlit UI** to try your own axioms & targets

---

## Components

| File/Folder            | Description |
|------------------------|-------------|
| `logic_env.py`         | V1: Simple logic environment using only Modus Ponens |
| `logic_env_extended.py`| V2: Extended environment with 4 inference rules |
| `train_ppo.py`         | Trains a PPO agent on a chosen theorem |
| `train_random.py`      | Runs a random policy for baseline comparison |
| `evaluate_ppo.py`      | Evaluates trained PPO agent and generates metrics |
| `evaluate_policy.py`   | Evaluation script for REINFORCE (if used) |
| `theorems.json`        | List of toy theorem problems and axioms |
| `app.py`               | Streamlit-based interactive demo |
| `utils.py`             | Common helpers: loading theorems, formatting traces |
| `report.ipynb`         | Notebook to plot results and replay proof traces |

---

## Quickstart

### 1. Set up the environment

```bash
python3 -m venv mvnv
source mvnv/bin/activate
pip install -r requirements.txt
````

### 2. Train a PPO agent

```bash
python train_ppo.py --theorem_id simple_chain --timesteps 100000
```

### 3. Evaluate the agent

```bash
python evaluate_ppo.py --model_path ppo_results/ppo_simple_chain.zip --episodes 100
```

### 4. Run the demo

```bash
streamlit run app.py
```

---

## Example Problem

Input:

* Axioms: `"A"`, `"A→B"`, `"B→C"`
* Target: `"C"`

The agent learns to:

1. Derive `"B"` using Modus Ponens on `"A"` and `"A→B"`
2. Then derive `"C"` using `"B"` and `"B→C"`

---

## Reinforcement Learning Details

* **State** = Binary vector of known facts
* **Actions** = Apply one of the inference rules
* **Rewards**

  * +0.1 for each new fact (intrinsic curiosity)
  * +1.0 for reaching the target theorem (extrinsic goal)

Training is done using:

* **PPO (Proximal Policy Optimization)** from Stable-Baselines3
* Optional baseline with **random policies**

---

## Output & Logs

| Directory           | Contents                       |
| ------------------- | ------------------------------ |
| `ppo_results/`      | Saved PPO model checkpoints    |
| `ppo_eval/`         | Evaluation traces and metrics  |
| `ppo_tb/`           | TensorBoard training logs      |
| `baseline_results/` | Random policy benchmark graphs |

---
