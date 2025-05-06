import gym
from gym import spaces
import numpy as np
from typing import List, Set, Tuple, Dict, Any

class LogicEnvV2(gym.Env):
    """
    Gym env with 4 actions:
      0: Modus Ponens
      1: AND-Elimination
      2: OR-Introduction
      3: Resolution
    """
    metadata = {"render.modes": ["human"]}
    RULES = ["mp", "and_elim", "or_intro", "resolution"]

    def __init__(self, axioms: List[str], target: str, max_steps: int = 50):
        super().__init__()
        self.axioms = [a.strip() for a in axioms]
        self.target = target.strip()
        self.max_steps = max_steps

        # 1) Collect atoms from axioms+target
        atoms: Set[str] = set()
        impls, ands, ors = set(), set(), set()
        for f in self.axioms + [self.target]:
            t = f.strip()
            if "→" in t:
                p, c = map(str.strip, t.split("→"))
                atoms |= {p, c}
                impls.add(f"{p}→{c}")
            elif "∧" in t:
                a, b = map(str.strip, t.split("∧"))
                atoms |= {a, b}
                ands.add(f"{a}∧{b}")
            elif "∨" in t:
                a, b = map(str.strip, t.split("∨"))
                atoms |= {a, b}
                ors.add(f"{a}∨{b}")
            else:
                atoms.add(t)

        # 2) Generate *all* possible disjunctions A∨B for distinct A,B
        for a in atoms:
            for b in atoms:
                if a != b:
                    ors.add(f"{a}∨{b}")

        # 3) Build the full fact universe
        self.all_facts: List[str] = (
            sorted(atoms)
            + sorted(impls)
            + sorted(ands)
            + sorted(ors)
        )
        self.fact_to_idx: Dict[str,int] = {
            fact: idx for idx, fact in enumerate(self.all_facts)
        }

        # Gym spaces
        self.observation_space = spaces.MultiBinary(len(self.all_facts))
        self.action_space = spaces.Discrete(len(self.RULES))

        # Initialize state
        self.reset()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(len(self.all_facts), dtype=np.int8)
        for f in self.known_facts:
            obs[self.fact_to_idx[f]] = 1
        return obs

    def reset(self) -> np.ndarray:
        """Start fresh: only axioms known, step=0."""
        self.known_facts: Set[str] = set(self.axioms)
        self.current_step = 0
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str,Any]]:
        rule = self.RULES[action]
        new_lemmas: Set[str] = set()

        if rule == "mp":
            # Modus Ponens
            for f in list(self.known_facts):
                if "→" in f:
                    p, c = map(str.strip, f.split("→"))
                    if p in self.known_facts and c not in self.known_facts:
                        new_lemmas.add(c)

        elif rule == "and_elim":
            # From (A∧B) infer A and B
            for f in list(self.known_facts):
                if "∧" in f:
                    a, b = map(str.strip, f.split("∧"))
                    if a not in self.known_facts: new_lemmas.add(a)
                    if b not in self.known_facts: new_lemmas.add(b)

        elif rule == "or_intro":
            # From A infer all (A∨X), where X≠A
            atoms = [f for f in self.all_facts if all(sym not in f for sym in ("→","∧","∨"))]
            for f in list(self.known_facts):
                if f in atoms:
                    for x in atoms:
                        if f != x:
                            disj = f"{f}∨{x}"
                            if disj not in self.known_facts:
                                new_lemmas.add(disj)

        elif rule == "resolution":
            # From (A∨B) and (¬B∨C) infer (A∨C)
            clauses = [f for f in self.known_facts if "∨" in f]
            for c1 in clauses:
                a, b = map(str.strip, c1.split("∨"))
                for c2 in clauses:
                    x, y = map(str.strip, c2.split("∨"))
                    # match b with ¬x or x with ¬b
                    if b == f"¬{x}" or x == f"¬{b}":
                        resolvent = f"{a}∨{y}"
                        if resolvent not in self.known_facts:
                            new_lemmas.add(resolvent)

        # Update and reward
        for lem in new_lemmas:
            self.known_facts.add(lem)
        intrinsic = 0.1 * len(new_lemmas)
        extrinsic = 1.0 if self.target in self.known_facts else 0.0
        reward = intrinsic + extrinsic

        self.current_step += 1
        done = (self.target in self.known_facts) or (self.current_step >= self.max_steps)
        return self._get_obs(), reward, done, {
            "new_lemmas": list(new_lemmas),
            "step": self.current_step
        }

    def render(self, mode="human") -> None:
        print(f"Step {self.current_step} | Known: {sorted(self.known_facts)}")
