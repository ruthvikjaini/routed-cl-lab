"""Metrics: normalized reward, stateless-reference gain, cost, Pareto.

Definitions follow the paper (Section 5, Metrics):
- normalized reward: the external stateless reference maps to 0 and the
  task maximum to 1; scores below the reference are negative. Here the
  reference is the expected stateless reward per family (see
  stream.stateless_reference), and the task maximum is 1 (binary reward).
- stateless-reference gain: each method compared with Stateless ICL on
  the same matched rollout: 0 is matched Stateless ICL, 1 is the task
  maximum.
- estimated API cost: every model call priced under the frozen schedule
  in costs.py.
"""

from __future__ import annotations

import numpy as np


def normalized_reward(reward: int, family: str, reference: dict[str, float]) -> float:
    ref = reference[family]
    return (reward - ref) / (1.0 - ref)


def stateless_gain(mean_reward: float, stateless_mean_reward: float) -> float:
    return (mean_reward - stateless_mean_reward) / (1.0 - stateless_mean_reward)


def mean_and_range(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.min()), float(arr.max())


def pareto_frontier(points: list[tuple[float, float]]) -> list[bool]:
    """points are (cost, reward); True where the point is non-dominated.

    A point is on the frontier if no other point has both lower-or-equal
    cost and higher-or-equal reward with at least one strict.
    """
    on_frontier = []
    for i, (c_i, r_i) in enumerate(points):
        dominated = any(
            (c_j <= c_i and r_j >= r_i) and (c_j < c_i or r_j > r_i)
            for j, (c_j, r_j) in enumerate(points)
            if j != i
        )
        on_frontier.append(not dominated)
    return on_frontier
