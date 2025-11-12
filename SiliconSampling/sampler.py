"""Sampling utilities for persona-based identity generation.

This module reads generated personas with demographics, personality traits,
and descriptions, and samples from them for CTR prediction.
"""

import json
import math
import os
import random
from typing import Any, Dict, List


def load_identity_bank(path: str) -> List[Dict[str, Any]]:
    """Load a generated personas JSON file from disk.

    Args:
        path: Filesystem path to the generated personas JSON.

    Returns:
        Parsed JSON as a list of persona dictionaries.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_identity(personas: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    """Sample a single persona from the list.

    Args:
        personas: List of generated personas.
        rng: Random number generator.

    Returns:
        A randomly selected persona dictionary.
    """
    return rng.choice(personas)


def sample_identities(n: int, personas: List[Dict[str, Any]], seed: int | None = None) -> List[Dict[str, Any]]:
    """Sample ``n`` personas from the provided list (with replacement).

    Args:
        n: Number of personas to sample.
        personas: List of generated personas.
        seed: Optional seed for reproducibility.

    Returns:
        List of sampled persona dictionaries.
    """
    rng = random.Random(seed)
    return [sample_identity(personas, rng) for _ in range(n)]


__all__ = [
    "load_identity_bank",
    "sample_identities",
    "sample_identity",
]
