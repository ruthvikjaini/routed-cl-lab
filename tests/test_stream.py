import numpy as np

from routed_cl_lab.stream import (
    FAMILIES,
    FAMILY_ORDER,
    make_stream,
    stateless_reference,
)


def test_stream_counts_match_paper_structure():
    stream = make_stream(0)
    assert len(stream) == 398
    counts = {fam: 0 for fam in FAMILY_ORDER}
    for inst in stream:
        counts[inst.family] += 1
    analytic = sum(counts[f] for f in FAMILY_ORDER if f != "banking")
    assert analytic == 301
    assert counts["banking"] == 97
    for fam in FAMILY_ORDER:
        assert counts[fam] == FAMILIES[fam]["count"]


def test_interleave_preserves_family_order():
    stream = make_stream(0, order="interleaved")
    seen: dict[str, list[int]] = {f: [] for f in FAMILY_ORDER}
    for inst in stream:
        seen[inst.family].append(inst.uid)
    for fam, uids in seen.items():
        assert uids == sorted(uids), f"{fam} order not preserved"


def test_blocked_order_groups_families():
    stream = make_stream(0, order="blocked")
    families = [inst.family for inst in stream]
    # Each family appears in exactly one contiguous block.
    blocks = []
    for fam in families:
        if not blocks or blocks[-1] != fam:
            blocks.append(fam)
    assert blocks == FAMILY_ORDER


def test_stream_is_deterministic():
    assert [i.uid for i in make_stream(3)] == [i.uid for i in make_stream(3)]
    assert [i.uid for i in make_stream(3)] != [i.uid for i in make_stream(4)]


def test_skill_vectors_are_unit_norm():
    stream = make_stream(0)
    for inst in stream:
        assert abs(np.linalg.norm(inst.skill) - 1.0) < 1e-6


def test_same_family_skills_more_similar_than_cross_family():
    rng = np.random.default_rng(0)
    stream = make_stream(0)
    by_family: dict[str, list] = {f: [] for f in FAMILY_ORDER}
    for inst in stream:
        by_family[inst.family].append(inst.skill)
    sql = np.mean(by_family["sql"], axis=0)
    poker = np.mean(by_family["poker"], axis=0)
    sql_centroid_sim = float(np.dot(sql, np.mean(by_family["sql"], axis=0)))
    cross_sim = float(
        np.dot(sql / np.linalg.norm(sql), poker / np.linalg.norm(poker))
    )
    assert sql_centroid_sim > cross_sim


def test_stateless_reference_in_range_and_deterministic():
    ref_a = stateless_reference()
    ref_b = stateless_reference()
    assert ref_a == ref_b
    for fam, value in ref_a.items():
        assert 0.05 < value < 0.95, f"{fam}: {value}"
