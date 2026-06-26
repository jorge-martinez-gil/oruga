"""Optimizer registry.

Maps short names to optimizer classes so a run can be selected by string (from a
config file or the CLI). Built-in optimizers need no extra dependencies; the
others lazily import theirs when instantiated.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Individual, Optimizer, OptimizationResult, dominates, non_dominated
from .builtin import GeneticAlgorithm, RandomSearch
from .metaheuristics import CuckooSearch, Jaya, TLBO
from .adapters import JMetalOptimizer, PyGADOptimizer

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "Individual",
    "dominates",
    "non_dominated",
    "RandomSearch",
    "GeneticAlgorithm",
    "Jaya",
    "CuckooSearch",
    "TLBO",
    "PyGADOptimizer",
    "JMetalOptimizer",
    "get_optimizer",
    "list_optimizers",
    "REGISTRY",
]

REGISTRY: Dict[str, Type[Optimizer]] = {
    "random": RandomSearch,
    "ga": GeneticAlgorithm,
    "jaya": Jaya,
    "cuckoo": CuckooSearch,
    "cs": CuckooSearch,
    "tlbo": TLBO,
    "pygad": PyGADOptimizer,
}


def _jmetal_factory(algorithm: str):
    def factory(**kwargs) -> Optimizer:
        return JMetalOptimizer(algorithm=algorithm, **kwargs)
    return factory


# jMetal algorithms share one adapter parameterized by name.
_JMETAL_ALGOS = {"nsga2": "nsga2", "nsgaii": "nsga2", "gde3": "gde3", "smpso": "smpso", "pso": "smpso"}


def get_optimizer(name: str, **kwargs) -> Optimizer:
    """Instantiate an optimizer by name with optional hyper-parameters."""
    key = name.lower()
    if key in REGISTRY:
        return REGISTRY[key](**kwargs)
    if key in _JMETAL_ALGOS:
        return JMetalOptimizer(algorithm=_JMETAL_ALGOS[key], **kwargs)
    raise ValueError(f"Unknown optimizer '{name}'. Available: {list_optimizers()}")


def list_optimizers() -> list:
    """Return all selectable optimizer names."""
    return sorted(set(REGISTRY) | set(_JMETAL_ALGOS))
