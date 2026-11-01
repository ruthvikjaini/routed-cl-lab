import numpy as np

from routed_cl_lab.memory import (
    ICL_CONTEXT_BUDGET_TOKENS,
    ICL_TOKENS_PER_TRAJECTORY,
    DiscreteEventMemory,
    HierarchicalRegressionMemory,
    ICLMemory,
    PersistentEntityMemory,
    ReasoningBankMemory,
    make_system,
)
from routed_cl_lab.stream import make_stream


def _instances(seed=0):
    return make_stream(seed)


def test_icl_retrieve_empty_is_zero():
    mem = ICLMemory()
    inst = _instances()[0]
    ret = mem.retrieve(inst)
    assert ret.relevant_mass == 0.0
    assert ret.irrelevant_mass == 0.0
    assert ret.fresh_tokens == 0


def test_icl_same_family_history_is_mostly_relevant():
    mem = ICLMemory()
    stream = _instances()
    sql = [i for i in stream if i.family == "sql"][:10]
    for inst in sql:
        mem.update(inst, 1)
    probe = [i for i in stream if i.family == "sql"][10]
    ret = mem.retrieve(probe)
    assert ret.relevant_mass > 0.5
    assert ret.irrelevant_mass == 0.0


def test_icl_mixed_history_dilutes_relevant_share():
    mem = ICLMemory()
    stream = _instances()
    for inst in stream[:70]:  # interleaved mix of all families
        mem.update(inst, 1)
    probe = [i for i in stream if i.family == "sql"][20]
    ret = mem.retrieve(probe)
    # Same-family share of a mixed history is small; irrelevant mass dominates.
    assert ret.relevant_mass < 0.3
    assert ret.irrelevant_mass > ret.relevant_mass


def test_icl_budget_cap_evicts_oldest():
    mem = ICLMemory()
    stream = _instances()
    # Feed more trajectories than the budget holds.
    max_trajs = ICL_CONTEXT_BUDGET_TOKENS // ICL_TOKENS_PER_TRAJECTORY
    for inst in stream:
        mem.update(inst, 1)
    assert len(mem.history) == max_trajs
    assert mem.history[0].uid != stream[0].uid  # oldest evicted


def test_icl_fresh_vs_cached_tokens():
    mem = ICLMemory()
    stream = _instances()
    for inst in stream[:5]:
        mem.update(inst, 1)
    first = mem.retrieve(stream[5])
    assert first.fresh_tokens == 5 * ICL_TOKENS_PER_TRAJECTORY
    assert first.cached_tokens == 0
    second = mem.retrieve(stream[6])
    assert second.fresh_tokens == 0
    assert second.cached_tokens == 5 * ICL_TOKENS_PER_TRAJECTORY


def test_reasoning_bank_retrieves_selectively():
    rng = np.random.default_rng(0)
    mem = ReasoningBankMemory(rng)
    stream = _instances()
    for inst in stream[:70]:
        mem.update(inst, 1)
    probe = [i for i in stream if i.family == "sql"][20]
    ret = mem.retrieve(probe)
    # Selective top-3 retrieval: relevant mass should beat the diluted
    # share an ICL history shows for the same mixed input.
    assert ret.relevant_mass > 0.3
    assert ret.fresh_tokens + ret.cached_tokens <= 3 * 60


def test_structured_system_matures_with_observations():
    mem = HierarchicalRegressionMemory()
    stream = _instances()
    sales = [i for i in stream if i.family == "sales"]
    probe = sales[20]
    assert mem.retrieve(probe).relevant_mass == 0.0  # unseen schema
    # Maturity grows with same-schema observations in the cluster,
    # regardless of which key each observation carried.
    for inst in sales[:8]:
        mem.update(inst, 1)
    matured = mem.retrieve(probe).relevant_mass
    assert 0.5 < matured < 1.0
    # More observations -> higher maturity.
    for inst in sales[8:20]:
        mem.update(inst, 1)
    assert mem.retrieve(probe).relevant_mass > matured


def test_structured_system_ignores_other_families():
    mem = HierarchicalRegressionMemory()
    stream = _instances()
    sales = [i for i in stream if i.family == "sales"]
    poker = [i for i in stream if i.family == "poker"]
    for inst in sales[:10]:
        mem.update(inst, 1)
    # A poker instance has no sales key: zero relevant mass, small cost.
    ret = mem.retrieve(poker[0])
    assert ret.relevant_mass == 0.0
    assert ret.fresh_tokens == 40
    # And poker updates do not pollute the sales state.
    assert mem.state_size() == len({i.key for i in sales[:10]})


def test_discrete_event_and_persistent_entity_schemas():
    hde = DiscreteEventMemory()
    pes = PersistentEntityMemory()
    stream = _instances()
    spectrum = [i for i in stream if i.family == "spectrum"]
    cohort = [i for i in stream if i.family == "cohort"]
    for inst in spectrum[:8]:
        hde.update(inst, 1)
    assert hde.retrieve(spectrum[8]).relevant_mass > 0.0
    # Cohort: the probe scores only once its entity key has been seen.
    # Pick the most common entity key so several same-key instances exist.
    from collections import Counter

    key_counts = Counter(i.key for i in cohort)
    common_key, _ = key_counts.most_common(1)[0]
    same_key = [i for i in cohort if i.key == common_key]
    assert len(same_key) >= 3
    probe = same_key[-1]
    assert pes.retrieve(probe).relevant_mass == 0.0  # unseen key
    for inst in same_key[:-1]:
        pes.update(inst, 1)
    assert pes.retrieve(probe).relevant_mass > 0.0
    # Cross-schema: a spectrum instance means nothing to the entity store.
    assert pes.retrieve(spectrum[8]).relevant_mass == 0.0


def test_make_system_names():
    for name in ("icl", "reasoning_bank", "hierarchical_regression",
                 "discrete_event", "persistent_entity"):
        assert make_system(name).name == name
