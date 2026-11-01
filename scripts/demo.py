"""Small demo: one interleaved rollout, all conditions, printed plainly.

Usage: python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from routed_cl_lab.experiment import CONDITIONS, run_rollout

CONDITION_LABELS = {
    "stateless": "Stateless ICL      ",
    "global_icl": "Global ICL         ",
    "global_rb": "Global RB          ",
    "routed_icl": "Routed ICL         ",
    "routed_rb": "Routed RB          ",
    "routed_cl": "Routed CL          ",
    "ground_truth": "Ground truth       ",
}


def main() -> None:
    results = run_rollout(seed=0, order="interleaved")
    print("one interleaved rollout (seed 0, 398 instances)")
    print(f"{'condition':22s} {'norm reward':>12s} {'gain':>7s} {'cost $':>8s} {'clusters':>8s}")
    for cond in CONDITIONS:
        r = results[cond]
        print(
            f"{CONDITION_LABELS[cond]:22s} {r.normalized_reward:12.3f} "
            f"{r.stateless_gain:7.3f} {r.total_cost_usd:8.2f} {r.n_clusters:8d}"
        )


if __name__ == "__main__":
    main()
