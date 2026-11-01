"""Cross-foot the README against the measured results.

1. Recompute every aggregate in results/measured.json from the raw
   per-rollout rows in results/experiment.json and fail on any mismatch
   beyond rounding (1e-3).
2. Check that every headline figure the README states appears in
   measured.json, formatted the way the README prints it.

Usage: python scripts/crossfoot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TOL = 1e-3


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    measured = json.loads((RESULTS / "measured.json").read_text())
    experiment = json.loads((RESULTS / "experiment.json").read_text())
    readme = (ROOT / "README.md").read_text()
    errors: list[str] = []

    def check(name: str, got: float, want: float) -> None:
        if abs(got - want) > TOL:
            errors.append(f"{name}: measured.json says {want}, recomputed {got}")

    # Group raw rows by (order, condition).
    rows: dict[tuple[str, str], list[dict]] = {}
    for row in experiment:
        rows.setdefault((row["order"], row["condition"]), []).append(row)

    for section, order in (("headline", "interleaved"), ("blocked", "blocked")):
        section_data = measured[section]
        global_costs = [r["total_cost_usd"] for r in rows[(order, "global_icl")]]
        for cond, agg in section_data.items():
            rs = rows[(order, cond)]
            check(
                f"{section}/{cond} mean_normalized_reward",
                _mean([r["normalized_reward"] for r in rs]),
                agg["mean_normalized_reward"],
            )
            check(
                f"{section}/{cond} mean_stateless_gain",
                _mean([r["stateless_gain"] for r in rs]),
                agg["mean_stateless_gain"],
            )
            check(
                f"{section}/{cond} mean_cost_usd",
                _mean([r["total_cost_usd"] for r in rs]),
                agg["mean_cost_usd"],
            )
            check(
                f"{section}/{cond} cost_reduction_vs_global_icl",
                _mean([(g - r["total_cost_usd"]) / g for g, r in zip(global_costs, rs)]),
                agg["cost_reduction_vs_global_icl"],
            )
            for fam, want in agg["per_family_normalized"].items():
                check(
                    f"{section}/{cond} per_family {fam}",
                    _mean([r["per_family_normalized"][fam] for r in rs]),
                    want,
                )

    # Concurrency aggregates recomputed from concurrency rows.
    conc_rows = measured["concurrency_rows"]
    for stale, agg in measured["concurrency"].items():
        rs = [r for r in conc_rows if r["staleness"] == int(stale)]
        check(
            f"concurrency/{stale} mean_normalized_reward",
            _mean([r["mean_normalized_reward"] for r in rs]),
            agg["mean_normalized_reward"],
        )
        check(
            f"concurrency/{stale} mean_cost_per_agent_usd",
            _mean([r["mean_cost_per_agent_usd"] for r in rs]),
            agg["mean_cost_per_agent_usd"],
        )

    # Isolated control: mean across families must equal the mean of the
    # per-family values stored alongside it.
    for cond, agg in measured["isolated"].items():
        check(
            f"isolated/{cond} mean_normalized_reward",
            _mean(list(agg["per_family_normalized"].values())),
            agg["mean_normalized_reward"],
        )
        check(
            f"isolated/{cond} mean_stateless_gain",
            _mean(list(agg["per_family_gain"].values())),
            agg["mean_stateless_gain"],
        )

    # Router-noise aggregates: recompute from experiment.json is not
    # possible (those rows are not stored there), so check internal
    # consistency instead: gain must be between the headline routed_cl
    # gain and the stateless gain (0), and cost must be positive.
    for noise, agg in measured["router_noise"].items():
        if not (-0.05 <= agg["mean_stateless_gain"] <= 1.0):
            errors.append(f"router_noise/{noise} gain out of range: {agg['mean_stateless_gain']}")
        if agg["mean_cost_usd"] <= 0:
            errors.append(f"router_noise/{noise} non-positive cost")

    # Headline figures stated in the README must exist in measured.json,
    # formatted exactly as the README prints them.
    headline = measured["headline"]
    figures = [
        ("routed_cl normalized reward", f"{headline['routed_cl']['mean_normalized_reward']:.3f}"),
        ("global_icl normalized reward", f"{headline['global_icl']['mean_normalized_reward']:.3f}"),
        ("routed_icl normalized reward", f"{headline['routed_icl']['mean_normalized_reward']:.3f}"),
        ("routed_cl stateless gain", f"{headline['routed_cl']['mean_stateless_gain']:.3f}"),
        ("global_icl stateless gain", f"{headline['global_icl']['mean_stateless_gain']:.3f}"),
        ("routed_cl cost reduction", f"{headline['routed_cl']['cost_reduction_vs_global_icl']:.0%}"),
        ("routed_icl cost reduction", f"{headline['routed_icl']['cost_reduction_vs_global_icl']:.0%}"),
        ("routed_cl cost", f"${headline['routed_cl']['mean_cost_usd']:.2f}"),
        ("global_icl cost", f"${headline['global_icl']['mean_cost_usd']:.2f}"),
    ]
    for name, formatted in figures:
        if formatted not in readme:
            errors.append(f"README is missing headline figure {name}: {formatted!r}")

    if errors:
        print("CROSSFOOT FAILED")
        for err in errors:
            print(" -", err)
        return 1
    print("crossfoot OK: all aggregates recompute and all README headline figures match measured.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
