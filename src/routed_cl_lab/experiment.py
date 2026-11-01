"""Experiment runner: the paper's staged conditions on matched rollouts.

Conditions (paper Section 5):
- stateless: no cross-instance state.
- global_icl: one shared ICL history across the stream (100K budget).
- global_rb: one shared ReasoningBank store across the stream.
- routed_icl: router-partitioned ICL histories (ICL only, per cluster).
- routed_rb: router-partitioned ReasoningBank stores.
- routed_cl: the full architecture -- router picks cluster and memory
  system from the five-system library.
- ground_truth: privileged diagnostic -- source-task identity plus the
  fixed ideal system mapping; an upper bound, not a method.

Every condition in a rollout sees the same stream in the same order
(matched rollouts), and the outcome RNG is shared across conditions so
paired comparisons use common random numbers. The router and
ReasoningBank extraction use their own separately seeded RNGs, so the
outcome draws stay in lockstep across conditions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from routed_cl_lab.agent import solve
from routed_cl_lab.costs import TokenBill, base_bill, router_bill
from routed_cl_lab.memory import Cluster, ICLMemory, ReasoningBankMemory, make_system
from routed_cl_lab.metrics import normalized_reward, stateless_gain
from routed_cl_lab.router import Router, ground_truth_cluster
from routed_cl_lab.stream import FAMILY_ORDER, Instance, make_stream, stateless_reference

CONDITIONS = [
    "stateless",
    "global_icl",
    "global_rb",
    "routed_icl",
    "routed_rb",
    "routed_cl",
    "ground_truth",
]

N_ROLLOUTS = 5


@dataclass
class ConditionResult:
    condition: str
    order: str
    seed: int
    n: int
    mean_reward: float
    normalized_reward: float
    stateless_gain: float
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    per_family_normalized: dict[str, float]
    router_agreement: dict[str, float] | None = None
    n_clusters: int = 0
    cluster_systems: dict[str, int] = field(default_factory=dict)


def _run_one(
    condition: str,
    stream: list[Instance],
    seed: int,
    reference: dict[str, float],
    router_noise: float = 0.10,
) -> ConditionResult:
    outcome_rng = np.random.default_rng(seed)
    router_rng = np.random.default_rng(seed + 1000)
    memory_rng = np.random.default_rng(seed + 2000)

    router: Router | None = None
    global_system = None
    gt_clusters: dict[str, Cluster] = {}
    if condition == "global_icl":
        global_system = ICLMemory()
    elif condition == "global_rb":
        global_system = ReasoningBankMemory(memory_rng)
    elif condition == "routed_icl":
        router = Router(router_rng, noise=router_noise, allowed_systems=["icl"])
    elif condition == "routed_rb":
        router = Router(router_rng, noise=router_noise, allowed_systems=["reasoning_bank"])
    elif condition == "routed_cl":
        router = Router(router_rng, noise=router_noise)

    rewards: list[int] = []
    norm_rewards: list[float] = []
    per_family: dict[str, list[float]] = {f: [] for f in FAMILY_ORDER}
    total_bill = TokenBill()
    cluster_systems: dict[str, int] = {}

    for step, instance in enumerate(stream):
        bill = base_bill()
        if condition == "stateless":
            rel, irr, system_name = 0.0, 0.0, None
        elif condition in ("global_icl", "global_rb"):
            retrieval = global_system.retrieve(instance)
            rel, irr = retrieval.relevant_mass, retrieval.irrelevant_mass
            system_name = global_system.name
            bill.fresh_input += retrieval.fresh_tokens
            bill.cached_input += retrieval.cached_tokens
        elif condition in ("routed_icl", "routed_rb", "routed_cl"):
            assert router is not None
            bill.fresh_input += router_bill().fresh_input
            bill.cached_input += router_bill().cached_input
            bill.output += router_bill().output
            cluster = router.route(instance, step)
            retrieval = cluster.system.retrieve(instance)
            rel, irr = retrieval.relevant_mass, retrieval.irrelevant_mass
            system_name = cluster.system_name
            bill.fresh_input += retrieval.fresh_tokens
            bill.cached_input += retrieval.cached_tokens
        elif condition == "ground_truth":
            cluster = ground_truth_cluster(gt_clusters, instance, memory_rng, step)
            retrieval = cluster.system.retrieve(instance)
            rel, irr = retrieval.relevant_mass, retrieval.irrelevant_mass
            system_name = cluster.system_name
            bill.fresh_input += retrieval.fresh_tokens
            bill.cached_input += retrieval.cached_tokens
        else:
            raise ValueError(f"unknown condition: {condition}")

        outcome = solve(
            instance.family,
            instance.difficulty,
            rel,
            irr,
            system_name,
            instance.affinity,
            outcome_rng,
        )
        rewards.append(outcome)
        nr = normalized_reward(outcome, instance.family, reference)
        norm_rewards.append(nr)
        per_family[instance.family].append(nr)
        total_bill.fresh_input += bill.fresh_input
        total_bill.cached_input += bill.cached_input
        total_bill.output += bill.output

        # Update state with the completed trajectory.
        if condition in ("global_icl", "global_rb"):
            global_system.update(instance, outcome)
        elif condition in ("routed_icl", "routed_rb", "routed_cl"):
            cluster.system.update(instance, outcome)
            cluster.members.append(instance)
        elif condition == "ground_truth":
            cluster.system.update(instance, outcome)
            cluster.members.append(instance)

    n_clusters = 0
    if router is not None:
        n_clusters = len(router.clusters)
        for c in router.clusters:
            cluster_systems[c.system_name] = cluster_systems.get(c.system_name, 0) + 1
    if gt_clusters:
        n_clusters = len(gt_clusters)
        for c in gt_clusters.values():
            cluster_systems[c.system_name] = cluster_systems.get(c.system_name, 0) + 1

    return ConditionResult(
        condition=condition,
        order="",
        seed=seed,
        n=len(stream),
        mean_reward=float(np.mean(rewards)),
        normalized_reward=float(np.mean(norm_rewards)),
        stateless_gain=0.0,  # filled in by run_rollout after stateless is known
        total_cost_usd=total_bill.cost_usd(),
        total_input_tokens=total_bill.total_input(),
        total_output_tokens=total_bill.output,
        per_family_normalized={
            f: float(np.mean(v)) if v else 0.0 for f, v in per_family.items()
        },
        router_agreement=router.agreement_rates() if router is not None else None,
        n_clusters=n_clusters,
        cluster_systems=cluster_systems,
    )


def run_rollout(
    seed: int,
    order: str = "interleaved",
    conditions: list[str] | None = None,
    router_noise: float = 0.10,
    reference: dict[str, float] | None = None,
) -> dict[str, ConditionResult]:
    """Run all conditions on one matched stream; returns results by name."""
    if conditions is None:
        conditions = CONDITIONS
    if reference is None:
        reference = stateless_reference()
    stream = make_stream(seed, order=order)
    results: dict[str, ConditionResult] = {}
    for cond in conditions:
        res = _run_one(cond, stream, seed, reference, router_noise=router_noise)
        res.order = order
        results[cond] = res
    stateless_mean = results["stateless"].mean_reward
    for cond, res in results.items():
        res.stateless_gain = stateless_gain(res.mean_reward, stateless_mean)
    return results


def run_grid(
    seeds: list[int] | None = None,
    order: str = "interleaved",
    router_noise: float = 0.10,
) -> dict[str, list[ConditionResult]]:
    """Run the full grid: every condition x every rollout seed."""
    if seeds is None:
        seeds = list(range(N_ROLLOUTS))
    reference = stateless_reference()
    grid: dict[str, list[ConditionResult]] = {c: [] for c in CONDITIONS}
    for seed in seeds:
        rollout = run_rollout(
            seed, order=order, router_noise=router_noise, reference=reference
        )
        for cond, res in rollout.items():
            grid[cond].append(res)
    return grid


# ---------------------------------------------------------------------------
# Isolated control: the paper's CL-Bench observation.
#
# CL-Bench evaluated each task as a separate homogeneous stream, with a
# fresh history per task, and there naive ICL was strongest. Interleaving
# is what breaks it. This control runs every condition separately per
# family (fresh state each time) and averages across families with equal
# weight, exactly as the paper weights its seven source tasks.
# ---------------------------------------------------------------------------


@dataclass
class IsolatedResult:
    condition: str
    seed: int
    mean_normalized_reward: float
    mean_stateless_gain: float
    mean_cost_usd: float
    per_family_normalized: dict[str, float]
    per_family_gain: dict[str, float]


def run_isolated(
    seed: int,
    conditions: list[str] | None = None,
    reference: dict[str, float] | None = None,
) -> dict[str, IsolatedResult]:
    from routed_cl_lab.stream import make_family_sequence

    if conditions is None:
        conditions = CONDITIONS
    if reference is None:
        reference = stateless_reference()
    per_family: dict[str, dict[str, ConditionResult]] = {
        f: {} for f in FAMILY_ORDER
    }
    for fi, family in enumerate(FAMILY_ORDER):
        rng = np.random.default_rng(seed * 1000 + fi)
        seq = make_family_sequence(family, 0, rng)
        for cond in conditions:
            per_family[family][cond] = _run_one(
                cond, seq, seed * 1000 + fi, reference
            )
    out: dict[str, IsolatedResult] = {}
    for cond in conditions:
        norms = [per_family[f][cond].normalized_reward for f in FAMILY_ORDER]
        gains = []
        costs = []
        for f in FAMILY_ORDER:
            s = per_family[f]["stateless"].mean_reward
            gains.append(stateless_gain(per_family[f][cond].mean_reward, s))
            costs.append(per_family[f][cond].total_cost_usd)
        out[cond] = IsolatedResult(
            condition=cond,
            seed=seed,
            mean_normalized_reward=float(np.mean(norms)),
            mean_stateless_gain=float(np.mean(gains)),
            mean_cost_usd=float(np.mean(costs)),
            per_family_normalized={
                f: per_family[f][cond].normalized_reward for f in FAMILY_ORDER
            },
            per_family_gain={f: g for f, g in zip(FAMILY_ORDER, gains)},
        )
    return out


# ---------------------------------------------------------------------------
# Concurrency extension: the paper's stated open question.
#
# The paper evaluates sequential rollouts only and notes that simultaneous
# updates, stale reads, and cross-model sharing are untested. Here N
# agents share one routed-cl memory and read it through local snapshots
# refreshed every ``staleness`` steps: between refreshes, other agents'
# writes are invisible (stale reads). Writes go to the shared system
# immediately. We measure how reward and cost degrade with staleness
# against the sequential (staleness 0) baseline.
# ---------------------------------------------------------------------------


@dataclass
class ConcurrencyResult:
    n_agents: int
    staleness: int
    seed: int
    mean_normalized_reward: float
    mean_cost_per_agent_usd: float
    mean_reward: float


def run_concurrent(
    n_agents: int = 4,
    staleness: int = 0,
    seed: int = 0,
    reference: dict[str, float] | None = None,
) -> ConcurrencyResult:
    """N agents, one shared routed-cl memory, snapshot reads.

    Each agent runs its own interleaved stream (different rollout seed).
    Agents step in lockstep rounds; agent i's local view of every shared
    cluster system is a deep copy refreshed every ``staleness`` steps
    (staleness 0 = always fresh). Updates are written to the shared system
    and to the writer's own local view.
    """
    if reference is None:
        reference = stateless_reference()
    streams = [make_stream(seed * 100 + i, order="interleaved") for i in range(n_agents)]
    outcome_rngs = [np.random.default_rng(seed * 100 + i) for i in range(n_agents)]
    router_rng = np.random.default_rng(seed + 5000)
    router = Router(router_rng, noise=0.10)
    # Shared cluster systems, keyed by cluster id.
    shared: dict[int, Cluster] = {}
    # Per-agent local views: agent -> {cluster id -> deep-copied Cluster}.
    views: list[dict[int, Cluster]] = [dict() for _ in range(n_agents)]
    refresh_every = max(staleness, 1)

    all_norm: list[float] = []
    all_rewards: list[int] = []
    total_cost = 0.0

    n_steps = len(streams[0])
    for step in range(n_steps):
        for agent_idx in range(n_agents):
            instance = streams[agent_idx][step]
            bill = base_bill()
            bill.fresh_input += router_bill().fresh_input
            bill.cached_input += router_bill().cached_input
            bill.output += router_bill().output
            cluster = router.route(instance, step)
            cid = id(cluster)
            if cid not in shared:
                shared[cid] = cluster
            # Refresh this agent's local view on schedule (or first sight).
            # staleness 0 means always fresh: refresh every step.
            if (
                cid not in views[agent_idx]
                or staleness == 0
                or (staleness > 0 and step % refresh_every == 0)
            ):
                views[agent_idx][cid] = copy.deepcopy(shared[cid])
            local = views[agent_idx][cid]
            retrieval = local.system.retrieve(instance)
            outcome = solve(
                instance.family,
                instance.difficulty,
                retrieval.relevant_mass,
                retrieval.irrelevant_mass,
                local.system_name,
                instance.affinity,
                outcome_rngs[agent_idx],
            )
            # Write through to the shared system and the writer's view.
            shared[cid].system.update(instance, outcome)
            shared[cid].members.append(instance)
            local.system.update(instance, outcome)
            local.members.append(instance)
            bill.fresh_input += retrieval.fresh_tokens
            bill.cached_input += retrieval.cached_tokens
            total_cost += bill.cost_usd()
            all_rewards.append(outcome)
            all_norm.append(normalized_reward(outcome, instance.family, reference))

    return ConcurrencyResult(
        n_agents=n_agents,
        staleness=staleness,
        seed=seed,
        mean_normalized_reward=float(np.mean(all_norm)),
        mean_cost_per_agent_usd=total_cost / n_agents,
        mean_reward=float(np.mean(all_rewards)),
    )
