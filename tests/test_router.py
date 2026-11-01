import numpy as np

from routed_cl_lab.router import Router, ground_truth_cluster
from routed_cl_lab.stream import FAMILY_ORDER, make_stream


def test_router_creates_cluster_for_first_instance():
    router = Router(np.random.default_rng(0))
    stream = make_stream(0)
    cluster = router.route(stream[0], 0)
    assert len(router.clusters) == 1
    assert cluster.system_name in (
        "icl", "reasoning_bank", "hierarchical_regression",
        "discrete_event", "persistent_entity",
    )


def test_router_reuses_cluster_for_same_family():
    router = Router(np.random.default_rng(0), noise=0.0)
    stream = make_stream(0)
    sql = [i for i in stream if i.family == "sql"]
    first = router.route(sql[0], 0)
    # Feed the cluster some same-family members, then route another.
    first.system.update(sql[0], 1)
    first.members.append(sql[0])
    second = router.route(sql[1], 1)
    assert second is first


def test_router_opens_new_cluster_for_distant_family():
    router = Router(np.random.default_rng(0), noise=0.0)
    stream = make_stream(0)
    sql = [i for i in stream if i.family == "sql"][0]
    spectrum = [i for i in stream if i.family == "spectrum"][0]
    c1 = router.route(sql, 0)
    c1.system.update(sql, 1)
    c1.members.append(sql)
    c2 = router.route(spectrum, 1)
    assert c2 is not c1


def test_router_respects_allowed_systems():
    router = Router(np.random.default_rng(0), allowed_systems=["icl"])
    stream = make_stream(0)
    for inst in stream[:50]:
        cluster = router.route(inst, 0)
        assert cluster.system_name == "icl"


def test_router_cluster_cap():
    router = Router(np.random.default_rng(0), noise=0.0, max_clusters=2)
    stream = make_stream(0)
    for step, inst in enumerate(stream[:100]):
        router.route(inst, step)
    assert len(router.clusters) <= 2


def test_router_agreement_rates_are_bounded():
    router = Router(np.random.default_rng(0))
    stream = make_stream(0)
    for step, inst in enumerate(stream):
        cluster = router.route(inst, step)
        cluster.system.update(inst, 1)
        cluster.members.append(inst)
    rates = router.agreement_rates()
    for value in rates.values():
        assert 0.0 <= value <= 1.0
    # Router should agree with ground truth more often than chance
    # (1/7 families, 1/5 systems) on the partition at least.
    assert rates["partition"] > 0.5


def test_ground_truth_assignment_uses_family_and_affinity():
    clusters: dict = {}
    rng = np.random.default_rng(0)
    stream = make_stream(0)
    sql = [i for i in stream if i.family == "sql"][0]
    c = ground_truth_cluster(clusters, sql, rng, 0)
    assert c.system_name == sql.affinity == "icl"
    assert len(clusters) == 1
    # Same family reuses the same cluster.
    sql2 = [i for i in stream if i.family == "sql"][1]
    assert ground_truth_cluster(clusters, sql2, rng, 1) is c
    # A new family opens a new cluster.
    banking = [i for i in stream if i.family == "banking"][0]
    assert ground_truth_cluster(clusters, banking, rng, 2) is not c
    assert len(clusters) == 2
