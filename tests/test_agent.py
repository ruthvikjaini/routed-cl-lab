import numpy as np

from routed_cl_lab.agent import base_probability, fidelity, solve
from routed_cl_lab.stream import make_stream


def test_base_probability_decreases_with_difficulty():
    assert base_probability("sql", 0.2) > base_probability("sql", 0.8)


def test_relevant_mass_raises_probability():
    p0 = base_probability("sql", 0.5)
    p1 = base_probability("sql", 0.5, relevant_mass=0.8, system="icl", affinity="icl")
    assert p1 > p0


def test_irrelevant_mass_lowers_probability():
    p0 = base_probability("sql", 0.5, relevant_mass=0.8, system="icl", affinity="icl")
    p1 = base_probability(
        "sql", 0.5, relevant_mass=0.8, irrelevant_mass=0.8,
        system="icl", affinity="icl",
    )
    assert p1 < p0


def test_fidelity_matrix_specialists_beat_generalists_on_home_schema():
    assert fidelity("hierarchical_regression", "hierarchical_regression") > fidelity(
        "icl", "hierarchical_regression"
    )
    assert fidelity("icl", "icl") > fidelity(
        "hierarchical_regression", "icl"
    )


def test_probability_stays_in_range():
    for family in ("sql", "banking"):
        for rel in (0.0, 1.0):
            for irr in (0.0, 1.0):
                p = base_probability(
                    family, 0.5, relevant_mass=rel, irrelevant_mass=irr,
                    system="icl", affinity="icl",
                )
                assert 0.0 < p < 1.0


def test_solve_is_deterministic_given_rng():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    stream = make_stream(0)
    inst = stream[0]
    a = solve(inst.family, inst.difficulty, 0.5, 0.1, "icl", "icl", rng_a)
    b = solve(inst.family, inst.difficulty, 0.5, 0.1, "icl", "icl", rng_b)
    assert a == b
