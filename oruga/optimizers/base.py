"""Optimizer contract and shared result types.

Every algorithm -- whether it is a genetic algorithm, a swarm method or a
hand-written metaheuristic -- consumes an :class:`~oruga.problem.OrugaProblem`
and returns an :class:`OptimizationResult`. This is what turned six near-identical
comparison scripts into six small, interchangeable classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..problem import OrugaProblem


@dataclass
class Individual:
    """A candidate solution paired with its evaluations."""

    solution: List[float]
    costs: List[float]
    values: List[float]
    text: str = ""


@dataclass
class OptimizationResult:
    algorithm: str
    problem: OrugaProblem
    best: Individual
    front: List[Individual] = field(default_factory=list)
    history: List[float] = field(default_factory=list)
    n_evaluations: int = 0
    runtime: float = 0.0

    @property
    def best_text(self) -> str:
        return self.best.text

    @property
    def best_values(self) -> List[float]:
        return self.best.values

    def summary(self) -> str:
        labels = self.problem.labels()
        parts = ", ".join(f"{lab}={val:.3f}" for lab, val in zip(labels, self.best.values))
        return (
            f"[{self.algorithm}] {parts} | front={len(self.front)} | "
            f"evals={self.n_evaluations} | {self.runtime:.2f}s"
        )


class Optimizer(ABC):
    """Base class for all ORUGA optimizers."""

    name: str = "optimizer"

    @abstractmethod
    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        ...

    # -- helpers shared by every optimizer --------------------------------
    @staticmethod
    def _evaluate(problem: OrugaProblem, solution: List[float]) -> Individual:
        costs = problem.costs(solution)
        values = problem.values(solution)
        return Individual(solution=list(solution), costs=costs, values=values,
                          text=problem.decode(solution))

    @staticmethod
    def _best_by_scalar(individuals: List[Individual]) -> Individual:
        return min(individuals, key=lambda ind: sum(ind.costs))


def dominates(a: List[float], b: List[float]) -> bool:
    """Pareto dominance for minimization: ``a`` dominates ``b``."""
    not_worse = all(ai <= bi for ai, bi in zip(a, b))
    strictly_better = any(ai < bi for ai, bi in zip(a, b))
    return not_worse and strictly_better


def non_dominated(individuals: List[Individual]) -> List[Individual]:
    """Return the Pareto-optimal subset (by ``costs``), de-duplicated by text."""
    front: List[Individual] = []
    for cand in individuals:
        if any(dominates(other.costs, cand.costs) for other in individuals if other is not cand):
            continue
        front.append(cand)
    # De-duplicate identical decoded texts, keeping the first occurrence.
    seen = set()
    unique: List[Individual] = []
    for ind in front:
        if ind.text in seen:
            continue
        seen.add(ind.text)
        unique.append(ind)
    return unique
