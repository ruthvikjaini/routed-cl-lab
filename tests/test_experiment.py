import numpy as np

from routed_cl_lab.experiment import (
    CONDITIONS,
    run_concurrent,
    run_grid,
    run_rollout,
)


def test_all_conditions_run_on_a_rollout():
    results = run_rollout(seed=0, order="interleaved")
    assert set(results) == set(CONDITIONS)
    for cond, res in results.items():
        assert res.n == 398
        assert -1.0 < res.normalized_reward < 1.0
        assert res.total_cost_usd > 0
        assert res.total_input_tokens > 0


def test_stateless_normalized_reward_near_zero():
    # The reference is the expected stateless reward, so the stateless
    # condition's normalized reward should be close to zero.
    results = run_rollout(seed=0, order="interleaved")
    assert abs(results["stateless"].normalized_reward) < 0.08
    assert abs(results["stateless"].stateless_gain) < 1e-9


def test_rollout_is_reproducible():
    a = run_rollout(seed=1, order="interleaved")
    b = run_rollout(seed=1, order="interleaved")
    for cond in CONDITIONS:
        assert a[cond].mean_reward == b[cond].mean_reward
        assert a[cond].total_cost_usd == b[cond].total_cost_usd
        assert a[cond].normalized_reward == b[cond].normalized_reward


def test_grid_runs_all_seeds():
    grid = run_grid(seeds=[0, 1], order="interleaved")
    for cond in CONDITIONS:
        assert len(grid[cond]) == 2


def test_routed_beats_global_on_interleaved_stream():
    # The paper's central qualitative finding: on an interleaved stream,
    # a single global history gains almost nothing over stateless, while
    # routed memory gains a lot.
    results = run_rollout(seed=0, order="interleaved")
    assert results["routed_cl"].stateless_gain > results["global_icl"].stateless_gain
    assert results["routed_cl"].stateless_gain > 0.10
    assert results["global_icl"].stateless_gain < 0.10


def test_routed_costs_less_than_global():
    results = run_rollout(seed=0, order="interleaved")
    assert results["routed_cl"].total_cost_usd < results["global_icl"].total_cost_usd


def test_blocked_stream_restores_global_icl():
    # Blocking helps Global ICL (each block starts mostly relevant) but
    # the history still accumulates cross-family contamination, so the
    # gain stays below the isolated control. The key comparison is
    # blocked > interleaved.
    interleaved = run_rollout(seed=0, order="interleaved")
    blocked = run_rollout(seed=0, order="blocked")
    assert blocked["global_icl"].stateless_gain > interleaved["global_icl"].stateless_gain


def test_isolated_streams_restore_global_icl():
    # The paper's CL-Bench observation: evaluated separately, with a fresh
    # history per task, naive ICL is strong. The isolated control must
    # show a real gain, far above the interleaved gain.
    from routed_cl_lab.experiment import run_isolated

    isolated = run_isolated(seed=0)
    interleaved = run_rollout(seed=0, order="interleaved")
    assert isolated["global_icl"].mean_stateless_gain > 0.10
    assert (
        isolated["global_icl"].mean_stateless_gain
        > 2 * interleaved["global_icl"].stateless_gain
    )


def test_ground_truth_is_an_upper_bound():
    results = run_rollout(seed=0, order="interleaved")
    assert (
        results["ground_truth"].normalized_reward
        >= results["routed_cl"].normalized_reward - 0.05
    )


def test_concurrent_runs_and_staleness_monotone():
    reference = None
    base = run_concurrent(n_agents=2, staleness=0, seed=0)
    stale = run_concurrent(n_agents=2, staleness=16, seed=0)
    assert base.mean_normalized_reward >= -1.0
    # Very stale reads should not beat always-fresh reads.
    assert stale.mean_normalized_reward <= base.mean_normalized_reward + 0.05
