"""Optimizers: each should improve readability and respect the contract."""

import pytest

from oruga.optimizers import dominates, get_optimizer, list_optimizers, non_dominated
from oruga.optimizers.base import Individual
from oruga.problem import OrugaProblem

BUILTIN = ["random", "ga", "jaya", "cuckoo", "tlbo"]


def _problem(provider, text, **kw):
    return OrugaProblem.from_text(text, provider, metric="fkgl", **kw)


@pytest.mark.parametrize("algo", BUILTIN)
def test_optimizer_improves_fkgl(algo, provider, complex_text):
    problem = _problem(provider, complex_text)
    baseline = problem.scalar_cost([0] * problem.n_variables)
    kwargs = {"iterations": 25} if algo in ("jaya", "cuckoo", "tlbo") else (
        {"generations": 25} if algo == "ga" else {"evaluations": 300}
    )
    result = get_optimizer(algo, **kwargs).optimize(problem, seed=0)
    assert sum(result.best.costs) <= baseline
    assert result.best_text  # decodes to something
    assert result.n_evaluations > 0


@pytest.mark.parametrize("algo", BUILTIN)
def test_multiobjective_front_is_nondominated(algo, provider, complex_text):
    problem = _problem(provider, complex_text, modification=True, semantic=True)
    kwargs = {"iterations": 20} if algo in ("jaya", "cuckoo", "tlbo") else (
        {"generations": 20} if algo == "ga" else {"evaluations": 200}
    )
    result = get_optimizer(algo, **kwargs).optimize(problem, seed=0)
    assert len(result.front) >= 1
    front = result.front
    for a in front:
        assert not any(dominates(b.costs, a.costs) for b in front if b is not a)


def test_determinism_with_seed(provider, complex_text):
    problem = _problem(provider, complex_text, modification=True)
    r1 = get_optimizer("ga", generations=20).optimize(problem, seed=42)
    r2 = get_optimizer("ga", generations=20).optimize(problem, seed=42)
    assert r1.best.costs == r2.best.costs


def test_dominance_and_front_helpers():
    a = Individual([0], [1.0, 1.0], [1.0, 1.0], "a")
    b = Individual([0], [2.0, 2.0], [2.0, 2.0], "b")  # dominated by a
    c = Individual([0], [0.5, 3.0], [0.5, 3.0], "c")  # trade-off with a
    assert dominates(a.costs, b.costs)
    front = non_dominated([a, b, c])
    texts = {ind.text for ind in front}
    assert "b" not in texts and "a" in texts and "c" in texts


def test_registry_lists_expected_names():
    names = list_optimizers()
    for expected in ["ga", "random", "jaya", "cuckoo", "tlbo", "nsga2", "gde3", "smpso", "pygad"]:
        assert expected in names


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError):
        get_optimizer("does-not-exist")
