from routed_cl_lab.costs import (
    PRICE_CACHED_PER_M,
    PRICE_INPUT_PER_M,
    PRICE_OUTPUT_PER_M,
    TokenBill,
    base_bill,
    router_bill,
)
from routed_cl_lab.metrics import (
    normalized_reward,
    pareto_frontier,
    stateless_gain,
)


def test_token_bill_cost_math():
    bill = TokenBill(fresh_input=1_000_000, cached_input=1_000_000, output=1_000_000)
    expected = PRICE_INPUT_PER_M + PRICE_CACHED_PER_M + PRICE_OUTPUT_PER_M
    assert abs(bill.cost_usd() - expected) < 1e-9


def test_cached_tokens_are_cheaper_than_fresh():
    fresh = TokenBill(fresh_input=1_000_000)
    cached = TokenBill(cached_input=1_000_000)
    assert cached.cost_usd() < fresh.cost_usd()
    assert abs(cached.cost_usd() - PRICE_CACHED_PER_M) < 1e-9


def test_base_bill_is_same_for_every_instance():
    assert base_bill() == base_bill()
    assert base_bill().fresh_input == 800
    assert base_bill().output == 400


def test_router_bill_has_cached_prefix():
    bill = router_bill()
    assert bill.cached_input == 200
    assert bill.fresh_input == 100
    assert bill.output == 150


def test_normalized_reward_reference_maps_to_zero():
    ref = {"sql": 0.4}
    assert abs(normalized_reward(0, "sql", ref) - (0 - 0.4) / 0.6) < 1e-9
    assert abs(normalized_reward(1, "sql", ref) - 1.0) < 1e-9


def test_stateless_gain_definition():
    assert abs(stateless_gain(0.5, 0.4) - (0.1 / 0.6)) < 1e-9
    assert stateless_gain(0.4, 0.4) == 0.0


def test_pareto_frontier():
    # (cost, reward): cheap+good dominates expensive+bad.
    points = [(1.0, 0.9), (2.0, 0.8), (0.5, 0.5), (1.0, 0.95)]
    frontier = pareto_frontier(points)
    assert frontier == [False, False, True, True]
