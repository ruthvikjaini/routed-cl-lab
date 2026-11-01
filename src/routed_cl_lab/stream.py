"""Heterogeneous task streams for routed continual learning.

The stream mirrors the structure of Philip et al. (arXiv:2610.38796):
six analytic task families plus one multi-turn banking family, interleaved
in random order while preserving each family's canonical order, for 398
instances per rollout (301 analytic + 97 banking).

Every instance is synthetic. Each family has a latent skill profile (a
sparse vector over a shared skill space), a difficulty, and a schema
affinity: the memory system that the paper's ground-truth assignment pairs
with that family. Nothing here is a real benchmark item; the stream is a
controlled testbed for the routing mechanism itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SKILL_DIM = 12

# Family definitions. ``core`` lists the skill indices the family exercises
# most; ``affinity`` is the memory system the paper's ground-truth
# assignment pairs with the family (Table 8 discussion: Code -> ICL,
# Sales -> hierarchical regression); ``base_logit`` shifts the family's
# stateless difficulty; ``count`` is the per-rollout instance count.
FAMILIES = {
    "sql": {
        "core": [0, 1, 2],
        "affinity": "icl",
        "base_logit": 0.10,
        "count": 50,
        "label": "text-to-SQL",
    },
    "poker": {
        "core": [3, 4, 5],
        "affinity": "reasoning_bank",
        "base_logit": 0.00,
        "count": 50,
        "label": "poker vs deterministic opponents",
    },
    "sales": {
        "core": [6, 7],
        "affinity": "hierarchical_regression",
        "base_logit": 0.05,
        "count": 50,
        "label": "sales prediction (keyed scalars)",
    },
    "code": {
        "core": [1, 8, 9],
        "affinity": "icl",
        "base_logit": -0.10,
        "count": 50,
        "label": "codebase adaptation",
    },
    "cohort": {
        "core": [7, 10],
        "affinity": "persistent_entity",
        "base_logit": 0.00,
        "count": 50,
        "label": "cohort analysis (noisy entities)",
    },
    "spectrum": {
        "core": [5, 11],
        "affinity": "discrete_event",
        "base_logit": 0.05,
        "count": 51,
        "label": "spectrum (event-free curves)",
    },
    "banking": {
        "core": [2, 4, 10],
        "affinity": "reasoning_bank",
        "base_logit": -0.25,
        "count": 97,
        "label": "banking policy dialogues",
    },
}

FAMILY_ORDER = ["sql", "poker", "sales", "code", "cohort", "spectrum", "banking"]

# Per-family key spaces for the structured memory systems. Sales instances
# carry a product-region key; cohort instances carry an entity id; spectrum
# instances carry a horizon index. Other families carry no key.
KEY_SPACES = {"sales": 12, "cohort": 20, "spectrum": 8}


@dataclass
class Instance:
    """One synthetic task instance."""

    uid: int
    family: str
    skill: np.ndarray  # unit-norm vector over SKILL_DIM
    difficulty: float  # in [0, 1]
    key: int | None = None  # structured key for sales/cohort/spectrum
    horizon: int | None = None

    @property
    def affinity(self) -> str:
        return FAMILIES[self.family]["affinity"]


def family_skill_profile(family: str, rng: np.random.Generator) -> np.ndarray:
    """Skill profile for a family: strong on its core skills, weak elsewhere.

    A small amount of cross-family overlap is deliberate: families share
    skills (e.g. sql and code share skill 1), so similarity-based routing
    is informative but not trivially perfect.
    """
    vec = np.full(SKILL_DIM, 0.08)
    for idx in FAMILIES[family]["core"]:
        vec[idx] = 1.0
    vec = vec + rng.normal(0.0, 0.06, size=SKILL_DIM)
    vec = np.clip(vec, 0.01, None)
    return vec / np.linalg.norm(vec)


def make_family_sequence(
    family: str, start_uid: int, rng: np.random.Generator
) -> list[Instance]:
    """Canonical ordered sequence for one family."""
    count = FAMILIES[family]["count"]
    instances = []
    for i in range(count):
        skill = family_skill_profile(family, rng)
        difficulty = float(rng.uniform(0.15, 0.85))
        key = None
        horizon = None
        if family in KEY_SPACES:
            key = int(rng.integers(0, KEY_SPACES[family]))
        if family == "spectrum":
            horizon = int(rng.integers(0, KEY_SPACES["spectrum"]))
        instances.append(
            Instance(
                uid=start_uid + i,
                family=family,
                skill=skill,
                difficulty=difficulty,
                key=key,
                horizon=horizon,
            )
        )
    return instances


def interleave(sequences: list[list[Instance]], rng: np.random.Generator) -> list[Instance]:
    """Merge-shuffle family sequences, preserving each family's order."""
    pointers = [0] * len(sequences)
    remaining = [len(s) for s in sequences]
    stream: list[Instance] = []
    while sum(remaining) > 0:
        available = [i for i, r in enumerate(remaining) if r > 0]
        choice = int(rng.choice(available))
        stream.append(sequences[choice][pointers[choice]])
        pointers[choice] += 1
        remaining[choice] -= 1
    return stream


def make_stream(seed: int, order: str = "interleaved") -> list[Instance]:
    """Build one 398-instance stream.

    ``order`` is "interleaved" (random merge preserving per-family order,
    as in the paper) or "blocked" (each family's instances in one block,
    the control condition for the interleaving ablation).
    """
    if order not in ("interleaved", "blocked"):
        raise ValueError(f"unknown order: {order}")
    rng = np.random.default_rng(seed)
    sequences = []
    uid = 0
    for family in FAMILY_ORDER:
        seq = make_family_sequence(family, uid, rng)
        sequences.append(seq)
        uid += len(seq)
    if order == "blocked":
        return [inst for seq in sequences for inst in seq]
    return interleave(sequences, rng)


def stateless_reference(seed: int = 12345, n: int = 20000) -> dict[str, float]:
    """Expected stateless reward per family (the external reference).

    Computed by fixed-seed Monte Carlo over the difficulty distribution
    using the agent's base success probability (see agent.base_probability).
    The stateless condition's normalized reward is therefore ~0 by
    construction, matching the paper's convention that the external
    stateless reference maps to 0 and the task maximum to 1.
    """
    from routed_cl_lab.agent import base_probability

    rng = np.random.default_rng(seed)
    ref: dict[str, float] = {}
    for family in FAMILY_ORDER:
        difficulties = rng.uniform(0.15, 0.85, size=n)
        probs = [
            base_probability(family, float(d), relevant_mass=0.0, irrelevant_mass=0.0)
            for d in difficulties
        ]
        ref[family] = float(np.mean(probs))
    return ref
