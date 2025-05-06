import json
from typing import List, Dict, Any, Set
import numpy as np


def load_theorems(path: str) -> List[Dict[str, Any]]:
    """
    Load a list of theorem tasks from a JSON file.

    Args:
        path: Path to theorems.json
    Returns:
        List of dicts, each with keys 'id', 'axioms', and 'target'.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def encode_state(known_facts: Set[str], fact_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Convert a set of known facts into a binary vector.

    Args:
        known_facts: Set of fact strings currently known.
        fact_to_idx: Mapping from fact string to index in vector.

    Returns:
        np.ndarray of shape (len(fact_to_idx),) with 1 for present facts.
    """
    vec = np.zeros(len(fact_to_idx), dtype=np.int8)
    for fact in known_facts:
        idx = fact_to_idx.get(fact)
        if idx is not None:
            vec[idx] = 1
    return vec


def pretty_print_trace(trace: List[Set[str]]) -> None:
    """
    Print the sequence of known-fact sets at each inference step.

    Args:
        trace: List of sets, where each set is the known_facts after a step.
    """
    for step, known in enumerate(trace, start=1):
        sorted_facts = sorted(known)
        print(f"[Step {step}] Known: {sorted_facts}")
