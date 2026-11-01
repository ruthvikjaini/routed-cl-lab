"""Simulated agent: success probability from retrieved memory.

The solver is deliberately simple and mechanistic. An instance has a base
success probability set by its family and difficulty. Retrieved memory
moves that probability up through *relevant* mass (experience that matches
the instance, weighted by how faithful the memory system is for the
instance's schema) and down through *irrelevant* mass (experience from
other families visible in the same context, which dilutes attention).

This is the mechanism behind the paper's headline: a single global history
mixes families together, so its relevant share collapses under
interleaving, while routed clusters keep each context mostly relevant.
"""

from __future__ import annotations

import numpy as np

from routed_cl_lab.stream import FAMILIES

# Gain from perfectly relevant memory (logit units) and the interference
# penalty from perfectly irrelevant context. Chosen so that memory can move
# a middling instance from ~0.4 to ~0.8, and a fully diluted context can
# pull it back below its stateless level -- the qualitative regime the
# paper reports -- without hard-coding any headline number.
GAIN_LOGIT = 2.2
INTERFERENCE_LOGIT = 1.5

# Fidelity of each memory system for each schema (family affinity).
# Generic systems (ICL, ReasoningBank) work on every schema at moderate
# fidelity; the structured systems are near-perfect on their own schema
# and near-useless elsewhere. This is the paper's premise: no single
# memory system is optimal across applications.
FIDELITY: dict[str, dict[str, float]] = {
    "icl": {
        "icl": 0.75,
        "reasoning_bank": 0.50,
        "hierarchical_regression": 0.50,
        "discrete_event": 0.50,
        "persistent_entity": 0.50,
    },
    "reasoning_bank": {
        "icl": 0.50,
        "reasoning_bank": 0.75,
        "hierarchical_regression": 0.50,
        "discrete_event": 0.50,
        "persistent_entity": 0.50,
    },
    "hierarchical_regression": {
        "icl": 0.05,
        "reasoning_bank": 0.05,
        "hierarchical_regression": 0.95,
        "discrete_event": 0.05,
        "persistent_entity": 0.05,
    },
    "discrete_event": {
        "icl": 0.05,
        "reasoning_bank": 0.05,
        "hierarchical_regression": 0.05,
        "discrete_event": 0.95,
        "persistent_entity": 0.05,
    },
    "persistent_entity": {
        "icl": 0.05,
        "reasoning_bank": 0.05,
        "hierarchical_regression": 0.05,
        "discrete_event": 0.05,
        "persistent_entity": 0.95,
    },
}

SYSTEMS = list(FIDELITY.keys())


def fidelity(system: str, affinity: str) -> float:
    return FIDELITY[system][affinity]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


def base_probability(
    family: str,
    difficulty: float,
    relevant_mass: float = 0.0,
    irrelevant_mass: float = 0.0,
    system: str | None = None,
    affinity: str | None = None,
) -> float:
    """Success probability for one instance.

    ``relevant_mass`` and ``irrelevant_mass`` are in [0, 1]. When a memory
    system is named, the gain is scaled by that system's fidelity for the
    instance's schema affinity; otherwise the caller passes a
    fidelity-adjusted mass already.
    """
    logit = 0.4 - 1.2 * difficulty + FAMILIES[family]["base_logit"]
    if relevant_mass > 0.0:
        fid = 1.0
        if system is not None and affinity is not None:
            fid = fidelity(system, affinity)
        logit += GAIN_LOGIT * relevant_mass * fid
    if irrelevant_mass > 0.0:
        logit -= INTERFERENCE_LOGIT * irrelevant_mass
    return float(np.clip(_sigmoid(logit), 0.01, 0.99))


def solve(
    family: str,
    difficulty: float,
    relevant_mass: float,
    irrelevant_mass: float,
    system: str | None,
    affinity: str,
    rng: np.random.Generator,
) -> int:
    """Sample a Bernoulli outcome (1 = success) for one instance."""
    p = base_probability(
        family, difficulty, relevant_mass, irrelevant_mass, system, affinity
    )
    return int(rng.random() < p)
