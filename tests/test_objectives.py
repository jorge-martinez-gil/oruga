"""Objectives: values, cost directionality, and assembly."""

from oruga.encoding import CandidateText
from oruga.objectives import (
    ModificationObjective,
    ReadabilityObjective,
    SemanticDistanceObjective,
    build_objectives,
)


def test_readability_cost_minimizes_grade(provider, complex_text):
    ct = CandidateText(complex_text, provider)
    obj = ReadabilityObjective("fkgl")
    # cost == value for "lower is easier" metrics.
    sol = [0] * ct.n_variables
    assert obj.cost(ct, sol) == obj.value(ct, sol)


def test_flesch_cost_is_negated(provider, complex_text):
    ct = CandidateText(complex_text, provider)
    obj = ReadabilityObjective("flesch")  # higher = easier
    sol = [0] * ct.n_variables
    assert obj.cost(ct, sol) == -obj.value(ct, sol)


def test_modification_modes(provider):
    ct = CandidateText("the committee deliberated", provider)
    sol = [0, 1, 1]
    assert ModificationObjective("active").value(ct, sol) == 2.0
    assert ModificationObjective("changes").value(ct, sol) == 2.0
    assert 0 < ModificationObjective("rate").value(ct, sol) <= 1.0


def test_semantic_distance_bounds(provider):
    ct = CandidateText("the committee deliberated", provider)
    obj = SemanticDistanceObjective(backend="jaccard")
    assert obj.value(ct, [0, 0, 0]) == 0.0          # identical -> distance 0
    assert obj.value(ct, [0, 1, 1]) > 0.0           # changed -> positive distance


def test_build_objectives_counts():
    assert len(build_objectives("fkgl")) == 1
    assert len(build_objectives("fkgl", modification=True)) == 2
    assert len(build_objectives("fkgl", modification=True, semantic=True)) == 3
