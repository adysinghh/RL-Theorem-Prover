import gym
from gym import spaces
import numpy as np
from typing import List, Set, Tuple, Dict, Any

class LogicEnv(gym.Env):
    """
    Gym environment for propositional-logic theorem proving using Modus Ponens.

    Each episode begins with a set of axioms and a target theorem. The agent's sole action
    applies Modus Ponens across all known facts. Intrinsic rewards are given for each new
    derived fact; an extrinsic reward is granted when the target theorem is derived.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        axioms: List[str],
        target: str,
        max_steps: int = 50,
    ):
        super().__init__()
        self.axioms = [a.strip() for a in axioms]
        self.target = target.strip()
        self.max_steps = max_steps

        # Build the complete set of facts (atoms + implications)
        self.all_facts = self._collect_all_facts()
        self.fact_to_idx = {fact: idx for idx, fact in enumerate(self.all_facts)}

        # Observation and action spaces
        self.observation_space = spaces.MultiBinary(len(self.all_facts))
        self.action_space = spaces.Discrete(1)  # Only one action: apply Modus Ponens

        # Initialize environment state
        self.reset()

    def _collect_all_facts(self) -> List[str]:
        atoms: Set[str] = set()
        implications: Set[str] = set()
        for fact in self.axioms + [self.target]:
            t = fact.strip()
            if "→" in t:
                p, c = map(str.strip, t.split("→"))
                atoms.add(p)
                atoms.add(c)
                implications.add(f"{p}→{c}")
            else:
                atoms.add(t)
        # Deterministic ordering
        return sorted(atoms) + sorted(implications)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(len(self.all_facts), dtype=np.int8)
        for f in self.known_facts:
            obs[self.fact_to_idx[f]] = 1
        return obs

    def reset(self) -> np.ndarray:
        """Reset environment: known_facts ← axioms, step count ← 0."""
        self.known_facts: Set[str] = set(self.axioms)
        self.current_step = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step: always applies Modus Ponens.

        Returns:
            obs: New observation vector.
            reward: 0.1 per new fact, +1.0 if target proved.
            done: True if target proved or max_steps reached.
            info: {'new_lemmas': List[str], 'step': int}
        """
        if action != 0:
            raise ValueError("Only action 0 (Modus Ponens) is supported.")

        # Derive new lemmas
        new_lemmas: Set[str] = set()
        for fact in list(self.known_facts):
            if "→" in fact:
                premise, conclusion = map(str.strip, fact.split("→"))
                if premise in self.known_facts and conclusion not in self.known_facts:
                    new_lemmas.add(conclusion)

        # Update known facts and compute reward
        for lemma in new_lemmas:
            self.known_facts.add(lemma)
        intrinsic = 0.1 * len(new_lemmas)
        extrinsic = 1.0 if self.target in self.known_facts else 0.0
        reward = intrinsic + extrinsic

        # Increment step count and check terminal
        self.current_step += 1
        done = (self.target in self.known_facts) or (self.current_step >= self.max_steps)

        return self._get_obs(), reward, done, {
            "new_lemmas": list(new_lemmas),
            "step": self.current_step
        }

    def render(self, mode: str = "human") -> None:
        """Print the current set of known facts."""
        print(f"Step {self.current_step} | Known facts: {sorted(self.known_facts)}")
