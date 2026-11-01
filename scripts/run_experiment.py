"""Run the headline grid, ablations, and concurrency extension.

Writes:
- results/experiment.json: every per-rollout condition result (the raw
  rows everything else is computed from).
- results/measured.json: aggregates -- the single source of truth for
  every number cited in the README. scripts/crossfoot.py recomputes
  these aggregates from experiment.json and checks the README against
  them.

Usage: python scripts/run_experiment.py [--quick]
--quick runs 2 rollout seeds instead of 5 (for CI smoke runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from routed_cl_lab.experiment import (
    CONDITIONS,
    ConcurrencyResult,
    run_concurrent,
    run_grid,
    run_rollout,
)
from routed_cl_lab.metrics import mean_and_range
from routed_cl_lab.stream import stateless_reference

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

NOISE_LEVELS = [0.0, 0.10, 0.20, 0.40]
CONCURRENCY_STALENESS = [0, 1, 4, 16]
CONCURRENCY_SEEDS = [0, 1, 2]


def aggregate(grid: dict[str, list]) -> dict[str, dict]:
    """Per-condition aggregates across rollouts, with paired cost ratios."""
    out: dict[str, dict] = {}
    global_costs = [r.total_cost_usd for r in grid["global_icl"]]
    for cond, rows in grid.items():
        norm = [r.normalized_reward for r in rows]
        gain = [r.stateless_gain for r in rows]
        costs = [r.total_cost_usd for r in rows]
        # Paired cost reduction vs Global ICL on the same rollout.
        reductions = [
            (g - c) / g for g, c in zip(global_costs, costs)
        ]
        mean_norm, min_norm, max_norm = mean_and_range(norm)
        mean_gain, min_gain, max_gain = mean_and_range(gain)
        mean_cost, _, _ = mean_and_range(costs)
        mean_red, min_red, max_red = mean_and_range(reductions)
        entry: dict = {
            "mean_normalized_reward": round(mean_norm, 4),
            "normalized_reward_range": [round(min_norm, 4), round(max_norm, 4)],
            "mean_stateless_gain": round(mean_gain, 4),
            "stateless_gain_range": [round(min_gain, 4), round(max_gain, 4)],
            "mean_cost_usd": round(mean_cost, 4),
            "cost_reduction_vs_global_icl": round(mean_red, 4),
            "cost_reduction_range": [round(min_red, 4), round(max_red, 4)],
            "mean_clusters": round(float(np.mean([r.n_clusters for r in rows])), 2),
            "per_family_normalized": {
                fam: round(
                    float(np.mean([r.per_family_normalized[fam] for r in rows])), 4
                )
                for fam in rows[0].per_family_normalized
            },
            # Mean number of clusters using each memory system, across rollouts.
            "cluster_systems": {
                sys: round(
                    float(np.mean([r.cluster_systems.get(sys, 0) for r in rows])), 2
                )
                for sys in sorted({s for r in rows for s in r.cluster_systems})
            },
        }
        agreements = [r.router_agreement for r in rows if r.router_agreement]
        if agreements:
            entry["router_agreement"] = {
                k: round(float(np.mean([a[k] for a in agreements])), 4)
                for k in ("system", "partition", "both")
            }
        out[cond] = entry
    return out


def serialize_grid(grid: dict[str, list]) -> list[dict]:
    rows = []
    for cond, results in grid.items():
        for r in results:
            rows.append(
                {
                    "condition": r.condition,
                    "order": r.order,
                    "seed": r.seed,
                    "n": r.n,
                    "mean_reward": round(r.mean_reward, 6),
                    "normalized_reward": round(r.normalized_reward, 6),
                    "stateless_gain": round(r.stateless_gain, 6),
                    "total_cost_usd": round(r.total_cost_usd, 6),
                    "total_input_tokens": r.total_input_tokens,
                    "total_output_tokens": r.total_output_tokens,
                    "per_family_normalized": {
                        k: round(v, 6) for k, v in r.per_family_normalized.items()
                    },
                    "router_agreement": r.router_agreement,
                    "n_clusters": r.n_clusters,
                    "cluster_systems": r.cluster_systems,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    seeds = [0, 1] if args.quick else list(range(5))
    reference = stateless_reference()
    print(f"reference (expected stateless reward): { {k: round(v, 3) for k, v in reference.items()} }")

    print("headline grid: interleaved, seeds", seeds)
    grid = run_grid(seeds=seeds, order="interleaved")
    measured: dict = {
        "headline": aggregate(grid),
        "n_rollouts": len(seeds),
        "seeds": seeds,
        "stream": {"n_instances": 398, "analytic": 301, "banking": 97, "order": "interleaved"},
    }

    print("ablation: blocked order")
    blocked = run_grid(seeds=seeds, order="blocked")
    measured["blocked"] = aggregate(blocked)

    print("control: isolated streams (fresh history per family)")
    from routed_cl_lab.experiment import run_isolated
    from routed_cl_lab.stream import FAMILY_ORDER as _FAMS

    isolated_rows: dict[str, list] = {}
    for s in seeds:
        iso = run_isolated(s, reference=reference)
        for cond, res in iso.items():
            isolated_rows.setdefault(cond, []).append(res)
    measured["isolated"] = {
        cond: {
            "mean_normalized_reward": round(
                float(np.mean([r.mean_normalized_reward for r in rows])), 4
            ),
            "mean_stateless_gain": round(
                float(np.mean([r.mean_stateless_gain for r in rows])), 4
            ),
            "mean_cost_usd": round(
                float(np.mean([r.mean_cost_usd for r in rows])), 4
            ),
            "per_family_normalized": {
                fam: round(
                    float(np.mean([r.per_family_normalized[fam] for r in rows])), 4
                )
                for fam in _FAMS
            },
            "per_family_gain": {
                fam: round(
                    float(np.mean([r.per_family_gain[fam] for r in rows])), 4
                )
                for fam in _FAMS
            },
        }
        for cond, rows in isolated_rows.items()
    }

    print("ablation: router noise (routed_cl only)")
    noise_rows: dict[str, dict] = {}
    for noise in NOISE_LEVELS:
        rows = [
            run_rollout(s, order="interleaved", conditions=["stateless", "routed_cl"],
                        router_noise=noise, reference=reference)["routed_cl"]
            for s in seeds
        ]
        norm = [r.normalized_reward for r in rows]
        gain = [r.stateless_gain for r in rows]
        costs = [r.total_cost_usd for r in rows]
        mean_norm, min_norm, max_norm = mean_and_range(norm)
        mean_gain, _, _ = mean_and_range(gain)
        mean_cost, _, _ = mean_and_range(costs)
        agreements = [r.router_agreement for r in rows if r.router_agreement]
        noise_rows[str(noise)] = {
            "mean_normalized_reward": round(mean_norm, 4),
            "normalized_reward_range": [round(min_norm, 4), round(max_norm, 4)],
            "mean_stateless_gain": round(mean_gain, 4),
            "mean_cost_usd": round(mean_cost, 4),
            "router_agreement": {
                k: round(float(np.mean([a[k] for a in agreements])), 4)
                for k in ("system", "partition", "both")
            },
            "mean_clusters": round(float(np.mean([r.n_clusters for r in rows])), 2),
        }
    measured["router_noise"] = noise_rows

    print("extension: concurrent agents sharing routed-cl memory")
    conc_rows: list[dict] = []
    for staleness in CONCURRENCY_STALENESS:
        for s in CONCURRENCY_SEEDS:
            res = run_concurrent(n_agents=4, staleness=staleness, seed=s, reference=reference)
            conc_rows.append(
                {
                    "n_agents": res.n_agents,
                    "staleness": res.staleness,
                    "seed": res.seed,
                    "mean_normalized_reward": round(res.mean_normalized_reward, 4),
                    "mean_cost_per_agent_usd": round(res.mean_cost_per_agent_usd, 4),
                    "mean_reward": round(res.mean_reward, 4),
                }
            )
    by_stale: dict[int, list[dict]] = {}
    for row in conc_rows:
        by_stale.setdefault(row["staleness"], []).append(row)
    measured["concurrency"] = {
        str(stale): {
            "mean_normalized_reward": round(
                float(np.mean([r["mean_normalized_reward"] for r in rows])), 4
            ),
            "mean_cost_per_agent_usd": round(
                float(np.mean([r["mean_cost_per_agent_usd"] for r in rows])), 4
            ),
            "mean_reward": round(float(np.mean([r["mean_reward"] for r in rows])), 4),
        }
        for stale, rows in by_stale.items()
    }
    measured["concurrency_rows"] = conc_rows

    RESULTS.mkdir(exist_ok=True)
    experiment_rows = serialize_grid(grid) + serialize_grid(blocked)
    (RESULTS / "experiment.json").write_text(json.dumps(experiment_rows, indent=2) + "\n")
    (RESULTS / "measured.json").write_text(json.dumps(measured, indent=2) + "\n")
    print(json.dumps(measured["headline"], indent=2))
    print("wrote results/experiment.json and results/measured.json")


if __name__ == "__main__":
    main()
