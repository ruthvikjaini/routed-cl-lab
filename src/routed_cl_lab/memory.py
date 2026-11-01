"""The five memory systems (paper Table 2) plus cluster state.

Three representation families:
- Retained trajectories: In-context Learning (ICL) -- the whole visible
  history is injected, so a mixed history dilutes the relevant share.
- Distilled reasoning memory: ReasoningBank (RB) -- up to three strategy
  items per trajectory, retrieved selectively by similarity, so cross-family
  pollution stays low even in a mixed cluster.
- Structured numerical state: Hierarchical Regression (HR), Hierarchical
  Discrete-Event State (HDE), Persistent-Entity State (PES) -- compact
  keyed state that is near-perfect on its own schema and near-useless
  elsewhere.

Each system reports the tokens it injects, split into fresh vs cached
input (repeated context is a cache hit), and the relevant / irrelevant
mass the agent sees. The agent converts those masses into a success
probability (see agent.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from routed_cl_lab.stream import FAMILIES, Instance

ICL_TOKENS_PER_TRAJECTORY = 400
ICL_CONTEXT_BUDGET_TOKENS = 100_000  # paper: 100K-token ICL budget
RB_TOKENS_PER_ITEM = 60
RB_MAX_ITEMS = 60
RB_ITEMS_PER_TRAJECTORY = 3
STRUCTURED_TOKENS_PER_RETRIEVAL = 40

# Maturity of structured state: 1 - exp(-n / tau), where n is the number
# of same-schema observations the cluster has accumulated. The state is
# a fitted mapping (a regression, a curve, an entity model), so it
# improves with every same-schema observation -- not only with repeats
# of one key. tau = 8: ~10 observations reach 71% maturity.
STRUCTURED_TAU = 8.0


@dataclass
class Retrieval:
    fresh_tokens: int = 0
    cached_tokens: int = 0
    relevant_mass: float = 0.0
    irrelevant_mass: float = 0.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@dataclass
class Trajectory:
    uid: int
    family: str
    skill: np.ndarray
    outcome: int


class MemorySystem:
    """Base class: update on a completed trajectory, retrieve for a new one."""

    name: str = "base"

    def update(self, instance: Instance, outcome: int) -> None:
        raise NotImplementedError

    def retrieve(self, instance: Instance) -> Retrieval:
        raise NotImplementedError

    def summary_vector(self) -> np.ndarray | None:
        raise NotImplementedError

    def state_size(self) -> int:
        raise NotImplementedError


class ICLMemory(MemorySystem):
    """Ordered visible trajectories, oldest dropped at the context budget."""

    name = "icl"

    def __init__(self) -> None:
        self.history: list[Trajectory] = []
        self._seen: set[int] = set()

    def update(self, instance: Instance, outcome: int) -> None:
        self.history.append(
            Trajectory(instance.uid, instance.family, instance.skill.copy(), outcome)
        )
        max_trajs = ICL_CONTEXT_BUDGET_TOKENS // ICL_TOKENS_PER_TRAJECTORY
        if len(self.history) > max_trajs:
            dropped = self.history[: len(self.history) - max_trajs]
            self.history = self.history[-max_trajs:]
            for traj in dropped:
                self._seen.discard(traj.uid)

    def retrieve(self, instance: Instance) -> Retrieval:
        if not self.history:
            return Retrieval()
        sims = [_cosine(instance.skill, t.skill) for t in self.history]
        n = len(self.history)
        rel = sum(s for s, t in zip(sims, self.history) if t.family == instance.family) / n
        irr = sum(s for s, t in zip(sims, self.history) if t.family != instance.family) / n
        fresh = sum(1 for t in self.history if t.uid not in self._seen)
        cached = n - fresh
        for t in self.history:
            self._seen.add(t.uid)
        return Retrieval(
            fresh_tokens=fresh * ICL_TOKENS_PER_TRAJECTORY,
            cached_tokens=cached * ICL_TOKENS_PER_TRAJECTORY,
            relevant_mass=float(np.clip(rel, 0.0, 1.0)),
            irrelevant_mass=float(np.clip(irr, 0.0, 1.0)),
        )

    def summary_vector(self) -> np.ndarray | None:
        if not self.history:
            return None
        return np.mean([t.skill for t in self.history], axis=0)

    def state_size(self) -> int:
        return len(self.history)


@dataclass
class RBItem:
    skill: np.ndarray
    family: str
    sign: int  # +1: strategy from a success; -1: lesson from a failure
    uid: int


class ReasoningBankMemory(MemorySystem):
    """Strategies induced from self-judged successes and failures."""

    name = "reasoning_bank"

    def __init__(self, rng: np.random.Generator) -> None:
        self.items: list[RBItem] = []
        self._seen: set[int] = set()
        self._rng = rng
        self._counter = 0

    def update(self, instance: Instance, outcome: int) -> None:
        # The judge extracts up to three items; extraction is noisy, so
        # sometimes fewer (or, for a failure, a lesson instead of a
        # strategy). Items carry the source trajectory's skill vector.
        n_items = int(self._rng.integers(1, RB_ITEMS_PER_TRAJECTORY + 1))
        for _ in range(n_items):
            skill = instance.skill + self._rng.normal(0.0, 0.08, size=instance.skill.shape)
            skill = np.clip(skill, 0.01, None)
            skill = skill / np.linalg.norm(skill)
            self.items.append(
                RBItem(skill=skill, family=instance.family, sign=1 if outcome else -1, uid=self._counter)
            )
            self._counter += 1
        if len(self.items) > RB_MAX_ITEMS:
            dropped = self.items[: len(self.items) - RB_MAX_ITEMS]
            self.items = self.items[-RB_MAX_ITEMS:]
            for item in dropped:
                self._seen.discard(item.uid)

    def retrieve(self, instance: Instance) -> Retrieval:
        if not self.items:
            return Retrieval()
        scored = sorted(
            self.items, key=lambda it: _cosine(instance.skill, it.skill), reverse=True
        )
        top = scored[:RB_ITEMS_PER_TRAJECTORY]
        rel = sum(_cosine(instance.skill, it.skill) for it in top if it.family == instance.family)
        irr = sum(_cosine(instance.skill, it.skill) for it in top if it.family != instance.family)
        rel /= RB_ITEMS_PER_TRAJECTORY
        irr /= RB_ITEMS_PER_TRAJECTORY
        fresh = sum(1 for it in top if it.uid not in self._seen)
        cached = len(top) - fresh
        for it in top:
            self._seen.add(it.uid)
        return Retrieval(
            fresh_tokens=fresh * RB_TOKENS_PER_ITEM,
            cached_tokens=cached * RB_TOKENS_PER_ITEM,
            relevant_mass=float(np.clip(rel, 0.0, 1.0)),
            irrelevant_mass=float(np.clip(irr, 0.0, 1.0)),
        )

    def summary_vector(self) -> np.ndarray | None:
        if not self.items:
            return None
        return np.mean([it.skill for it in self.items], axis=0)

    def state_size(self) -> int:
        return len(self.items)


class StructuredMemory(MemorySystem):
    """Shared machinery for HR / HDE / PES: keyed counts with maturity."""

    name = "structured"
    # The family whose keys this system records. Instances of other
    # families are ignored on update and score zero on retrieve.
    schema_family: str = ""

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self._skills: list[np.ndarray] = []
        self.n_schema: int = 0  # same-schema observations accumulated

    def _key_of(self, instance: Instance) -> int | None:
        if instance.family != self.schema_family:
            return None
        return instance.key if instance.key is not None else instance.uid % 997

    def update(self, instance: Instance, outcome: int) -> None:
        key = self._key_of(instance)
        if key is None:
            return
        self.counts[key] = self.counts.get(key, 0) + 1
        self.n_schema += 1
        self._skills.append(instance.skill.copy())

    def retrieve(self, instance: Instance) -> Retrieval:
        key = self._key_of(instance)
        if key is None:
            # Wrong schema for this system: the compact state is injected
            # anyway (small cost) but carries nothing usable.
            return Retrieval(
                fresh_tokens=STRUCTURED_TOKENS_PER_RETRIEVAL,
                relevant_mass=0.0,
                irrelevant_mass=0.0,
            )
        # Maturity grows with every same-schema observation in the
        # cluster: the fitted mapping sharpens as evidence accumulates.
        maturity = 1.0 - float(np.exp(-self.n_schema / STRUCTURED_TAU))
        return Retrieval(
            fresh_tokens=STRUCTURED_TOKENS_PER_RETRIEVAL,
            relevant_mass=maturity,
            irrelevant_mass=0.0,
        )

    def summary_vector(self) -> np.ndarray | None:
        if not self._skills:
            return None
        return np.mean(self._skills, axis=0)

    def state_size(self) -> int:
        return len(self.counts)


class HierarchicalRegressionMemory(StructuredMemory):
    """Repeated keyed scalar outcomes (sales prediction)."""

    name = "hierarchical_regression"
    schema_family = "sales"


class DiscreteEventMemory(StructuredMemory):
    """Monotone event-free curves over ordered horizons (spectrum)."""

    name = "discrete_event"
    schema_family = "spectrum"


class PersistentEntityMemory(StructuredMemory):
    """Noisy measurement sets with uncertain entity correspondence (cohort)."""

    name = "persistent_entity"
    schema_family = "cohort"


def make_system(name: str, rng: np.random.Generator | None = None) -> MemorySystem:
    if name == "icl":
        return ICLMemory()
    if name == "reasoning_bank":
        return ReasoningBankMemory(rng if rng is not None else np.random.default_rng(0))
    if name == "hierarchical_regression":
        return HierarchicalRegressionMemory()
    if name == "discrete_event":
        return DiscreteEventMemory()
    if name == "persistent_entity":
        return PersistentEntityMemory()
    raise ValueError(f"unknown memory system: {name}")


@dataclass
class Cluster:
    """A cluster couples one memory system with the trajectories it holds."""

    system_name: str
    system: MemorySystem
    members: list[Instance] = field(default_factory=list)
    created_at: int = 0

    def summary_vector(self) -> np.ndarray | None:
        # Prefer the members' skill vectors: they are the cluster's
        # identity as the router sees it. Fall back to the memory
        # system's own summary (structured state has no skill vectors
        # until its schema has been observed).
        if self.members:
            return np.mean([m.skill for m in self.members], axis=0)
        return self.system.summary_vector()

    def describe(self) -> str:
        return (
            f"{self.system_name} cluster with {len(self.members)} trajectories "
            f"({self.system.state_size()} stored items)"
        )
