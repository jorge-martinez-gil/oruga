"""Dependency-free optimizers.

These pure-Python algorithms let ORUGA run, be tested and be taught without any
heavy optimization library. They are also sensible defaults for small documents.

* :class:`RandomSearch` -- uniform sampling baseline (a fair, honest control).
* :class:`GeneticAlgorithm` -- a real GA (tournament selection, blend crossover,
  Gaussian mutation, elitism) operating on a scalarized objective, so it serves
  both single- and multi-objective problems.
"""

from __future__ import annotations

import random
import time
from typing import List, Optional

from ..problem import OrugaProblem
from .base import Individual, Optimizer, OptimizationResult, non_dominated


class RandomSearch(Optimizer):
    name = "random"

    def __init__(self, evaluations: int = 500):
        self.evaluations = evaluations

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        rng = random.Random(seed)
        lower, upper = problem.bounds()
        n = problem.n_variables
        start = time.perf_counter()

        population: List[Individual] = []
        history: List[float] = []
        best: Optional[Individual] = None

        for _ in range(self.evaluations):
            sol = [rng.uniform(lower[i], upper[i]) for i in range(n)]
            ind = self._evaluate(problem, sol)
            population.append(ind)
            if best is None or sum(ind.costs) < sum(best.costs):
                best = ind
            history.append(sum(best.costs))

        runtime = time.perf_counter() - start
        front = non_dominated(population) if problem.is_multiobjective else [best]  # type: ignore[list-item]
        return OptimizationResult(
            algorithm=self.name, problem=problem, best=best,  # type: ignore[arg-type]
            front=front, history=history, n_evaluations=self.evaluations, runtime=runtime,
        )


class GeneticAlgorithm(Optimizer):
    name = "ga"

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        mutation_scale: float = 1.0,
        tournament_size: int = 3,
        elitism: int = 1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size
        self.elitism = elitism

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        rng = random.Random(seed)
        lower, upper = problem.bounds()
        n = problem.n_variables
        start = time.perf_counter()
        archive: List[Individual] = []
        evals = 0

        def random_solution() -> List[float]:
            return [rng.uniform(lower[i], upper[i]) for i in range(n)]

        def evaluate(sol: List[float]) -> Individual:
            nonlocal evals
            evals += 1
            ind = self._evaluate(problem, sol)
            archive.append(ind)
            return ind

        def fitness(ind: Individual) -> float:
            return sum(ind.costs)

        population = [evaluate(random_solution()) for _ in range(self.population_size)]
        history: List[float] = [min(fitness(i) for i in population)]

        for _ in range(self.generations):
            population.sort(key=fitness)
            next_gen: List[Individual] = population[: self.elitism]

            while len(next_gen) < self.population_size:
                p1 = self._tournament(population, rng, fitness)
                p2 = self._tournament(population, rng, fitness)
                child = self._crossover(p1.solution, p2.solution, rng)
                child = self._mutate(child, lower, upper, rng)
                next_gen.append(evaluate(child))

            population = next_gen
            history.append(min(fitness(i) for i in population))

        runtime = time.perf_counter() - start
        best = min(archive, key=fitness)
        front = non_dominated(archive) if problem.is_multiobjective else [best]
        return OptimizationResult(
            algorithm=self.name, problem=problem, best=best, front=front,
            history=history, n_evaluations=evals, runtime=runtime,
        )

    def _tournament(self, population, rng, fitness) -> Individual:
        competitors = rng.sample(population, min(self.tournament_size, len(population)))
        return min(competitors, key=fitness)

    def _crossover(self, a: List[float], b: List[float], rng) -> List[float]:
        if rng.random() > self.crossover_rate:
            return list(a)
        # Blend crossover (BLX-0.0 -> arithmetic): interpolate per gene.
        return [ai + rng.random() * (bi - ai) for ai, bi in zip(a, b)]

    def _mutate(self, sol: List[float], lower, upper, rng) -> List[float]:
        out = list(sol)
        for i in range(len(out)):
            if rng.random() < self.mutation_rate:
                span = (upper[i] - lower[i]) * 0.1 * self.mutation_scale
                out[i] += rng.gauss(0, span)
                out[i] = max(lower[i], min(upper[i], out[i]))
        return out
