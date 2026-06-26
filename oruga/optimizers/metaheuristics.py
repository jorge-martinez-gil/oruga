"""Population metaheuristics (Jaya, Cuckoo Search, TLBO).

These reproduce the non-evolutionary baselines from the paper's comparison study
in a single, deduplicated form. Each one searches by minimizing the scalarized
objective while archiving every evaluation, so a Pareto front can be reported for
multi-objective problems. They require only NumPy.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional

from ..problem import OrugaProblem
from .base import Individual, Optimizer, OptimizationResult, non_dominated


class _NumpyOptimizer(Optimizer):
    """Shared evaluation/archiving plumbing for NumPy-based metaheuristics."""

    def _setup(self, problem: OrugaProblem, seed: Optional[int]):
        import numpy as np  # lazy, optional core dep

        rng = np.random.default_rng(seed)
        lower, upper = problem.bounds()
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        archive: List[Individual] = []

        def evaluate(vec) -> Individual:
            ind = self._evaluate(problem, list(np.asarray(vec, dtype=float)))
            archive.append(ind)
            return ind

        return np, rng, lower, upper, archive, evaluate

    def _finish(self, problem, archive, history, evals, start, name) -> OptimizationResult:
        runtime = time.perf_counter() - start
        best = min(archive, key=lambda ind: sum(ind.costs))
        front = non_dominated(archive) if problem.is_multiobjective else [best]
        return OptimizationResult(
            algorithm=name, problem=problem, best=best, front=front,
            history=history, n_evaluations=evals, runtime=runtime,
        )


class Jaya(_NumpyOptimizer):
    """Jaya: move toward the best solution and away from the worst."""

    name = "jaya"

    def __init__(self, population_size: int = 20, iterations: int = 50):
        self.population_size = population_size
        self.iterations = iterations

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        np, rng, lower, upper, archive, evaluate = self._setup(problem, seed)
        n = problem.n_variables
        start = time.perf_counter()

        pop = rng.uniform(lower, upper, size=(self.population_size, n))
        inds = [evaluate(row) for row in pop]
        scores = np.array([sum(i.costs) for i in inds])
        history = [float(scores.min())]

        for _ in range(self.iterations):
            best = pop[int(scores.argmin())]
            worst = pop[int(scores.argmax())]
            r1 = rng.random((self.population_size, n))
            r2 = rng.random((self.population_size, n))
            new_pop = pop + r1 * (best - np.abs(pop)) - r2 * (worst - np.abs(pop))
            new_pop = np.clip(new_pop, lower, upper)

            for i in range(self.population_size):
                cand = evaluate(new_pop[i])
                if sum(cand.costs) < scores[i]:
                    pop[i] = new_pop[i]
                    inds[i] = cand
                    scores[i] = sum(cand.costs)
            history.append(float(scores.min()))

        return self._finish(problem, archive, history, len(archive), start, self.name)


class CuckooSearch(_NumpyOptimizer):
    """Cuckoo Search via Levy flights with fraction-pa nest abandonment."""

    name = "cuckoo"

    def __init__(self, population_size: int = 20, iterations: int = 50, pa: float = 0.25, beta: float = 1.5):
        self.population_size = population_size
        self.iterations = iterations
        self.pa = pa
        self.beta = beta

    def _levy(self, np, rng, n):
        beta = self.beta
        sigma = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = rng.normal(0, sigma, n)
        v = rng.normal(0, 1, n)
        return u / np.abs(v) ** (1 / beta)

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        np, rng, lower, upper, archive, evaluate = self._setup(problem, seed)
        n = problem.n_variables
        start = time.perf_counter()

        nests = rng.uniform(lower, upper, size=(self.population_size, n))
        inds = [evaluate(row) for row in nests]
        scores = np.array([sum(i.costs) for i in inds])
        history = [float(scores.min())]

        for _ in range(self.iterations):
            best = nests[int(scores.argmin())]
            for i in range(self.population_size):
                step = self._levy(np, rng, n) * (nests[i] - best)
                cand_vec = np.clip(nests[i] + 0.01 * step, lower, upper)
                cand = evaluate(cand_vec)
                if sum(cand.costs) < scores[i]:
                    nests[i] = cand_vec
                    scores[i] = sum(cand.costs)
            # Abandon a fraction of the worst nests.
            abandon = rng.random((self.population_size, n)) < self.pa
            new_nests = np.clip(
                nests + abandon * rng.uniform(lower, upper, size=(self.population_size, n)) * 0.1,
                lower, upper,
            )
            for i in range(self.population_size):
                cand = evaluate(new_nests[i])
                if sum(cand.costs) < scores[i]:
                    nests[i] = new_nests[i]
                    scores[i] = sum(cand.costs)
            history.append(float(scores.min()))

        return self._finish(problem, archive, history, len(archive), start, self.name)


class TLBO(_NumpyOptimizer):
    """Teaching-Learning-Based Optimization (teacher + learner phases)."""

    name = "tlbo"

    def __init__(self, population_size: int = 20, iterations: int = 50):
        self.population_size = population_size
        self.iterations = iterations

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        np, rng, lower, upper, archive, evaluate = self._setup(problem, seed)
        n = problem.n_variables
        start = time.perf_counter()

        pop = rng.uniform(lower, upper, size=(self.population_size, n))
        inds = [evaluate(row) for row in pop]
        scores = np.array([sum(i.costs) for i in inds])
        history = [float(scores.min())]

        for _ in range(self.iterations):
            # Teacher phase.
            teacher = pop[int(scores.argmin())]
            mean = pop.mean(axis=0)
            for i in range(self.population_size):
                tf = rng.integers(1, 3)  # teaching factor 1 or 2
                new_vec = np.clip(pop[i] + rng.random(n) * (teacher - tf * mean), lower, upper)
                cand = evaluate(new_vec)
                if sum(cand.costs) < scores[i]:
                    pop[i] = new_vec
                    scores[i] = sum(cand.costs)
            # Learner phase.
            for i in range(self.population_size):
                j = int(rng.integers(0, self.population_size))
                if j == i:
                    continue
                if scores[i] < scores[j]:
                    new_vec = pop[i] + rng.random(n) * (pop[i] - pop[j])
                else:
                    new_vec = pop[i] + rng.random(n) * (pop[j] - pop[i])
                new_vec = np.clip(new_vec, lower, upper)
                cand = evaluate(new_vec)
                if sum(cand.costs) < scores[i]:
                    pop[i] = new_vec
                    scores[i] = sum(cand.costs)
            history.append(float(scores.min()))

        return self._finish(problem, archive, history, len(archive), start, self.name)
