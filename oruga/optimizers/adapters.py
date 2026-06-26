"""Adapters to external optimization libraries used in the paper.

These wrap the *exact* algorithms from the original study so published results
can be reproduced through the unified interface:

* :class:`PyGADOptimizer` -- single-objective GA (``pygad``), as in
  ``oruga_wordnet.py`` & co.
* :class:`JMetalOptimizer` -- NSGA-II, GDE3 and SMPSO (``jmetalpy``), as in the
  multi-objective scripts and the ``comparison/`` study.

Both import their dependency lazily, so the package works without them.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..problem import OrugaProblem
from .base import Individual, Optimizer, OptimizationResult, non_dominated


class PyGADOptimizer(Optimizer):
    """Single-objective genetic algorithm via PyGAD (paper-faithful)."""

    name = "pygad"

    def __init__(self, num_generations: int = 100, sol_per_pop: int = 20,
                 num_parents_mating: int = 10, **ga_kwargs):
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.ga_kwargs = ga_kwargs

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        import pygad  # lazy, optional

        lower, upper = problem.bounds()
        start = time.perf_counter()

        def fitness(*args):
            # Supports both pygad 2.x (solution, idx) and 3.x (ga, solution, idx).
            solution = args[-2]
            return -problem.scalar_cost(solution)

        kwargs = dict(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            sol_per_pop=self.sol_per_pop,
            num_genes=problem.n_variables,
            fitness_func=fitness,
            init_range_low=problem.lower_bound,
            init_range_high=problem.upper_bound,
        )
        if seed is not None:
            kwargs["random_seed"] = seed
        kwargs.update(self.ga_kwargs)

        ga = pygad.GA(**kwargs)
        ga.run()
        solution, _, _ = ga.best_solution()

        best = self._evaluate(problem, list(solution))
        runtime = time.perf_counter() - start
        return OptimizationResult(
            algorithm=self.name, problem=problem, best=best, front=[best],
            history=list(getattr(ga, "best_solutions_fitness", []) or []),
            n_evaluations=self.num_generations * self.sol_per_pop, runtime=runtime,
        )


class JMetalOptimizer(Optimizer):
    """Multi-objective optimizers from jMetalPy: NSGA-II, GDE3, SMPSO."""

    def __init__(self, algorithm: str = "nsga2", max_evaluations: int = 3000,
                 population_size: int = 100, **kwargs):
        self.algorithm = algorithm.lower()
        self.max_evaluations = max_evaluations
        self.population_size = population_size
        self.kwargs = kwargs
        self.name = self.algorithm

    def _build_problem(self, problem: OrugaProblem):
        from jmetal.core.problem import FloatProblem
        from jmetal.core.solution import FloatSolution

        outer = problem
        lower, upper = problem.bounds()

        class _Wrapped(FloatProblem):
            def __init__(self):
                super().__init__()
                self.number_of_objectives = outer.n_objectives
                self.number_of_variables = outer.n_variables
                self.number_of_constraints = 0
                self.obj_directions = [self.MINIMIZE] * outer.n_objectives
                self.obj_labels = outer.labels()
                self.lower_bound = list(lower)
                self.upper_bound = list(upper)
                FloatSolution.lower_bound = self.lower_bound
                FloatSolution.upper_bound = self.upper_bound

            def evaluate(self, solution):
                solution.objectives = outer.costs(solution.variables)
                return solution

            def number_of_objectives_(self):
                return outer.n_objectives

            def get_name(self):
                return "Oruga"

        return _Wrapped()

    def _build_algorithm(self, jproblem):
        from jmetal.operator import PolynomialMutation, SBXCrossover
        from jmetal.util.termination_criterion import StoppingByEvaluations

        mut = PolynomialMutation(probability=1.0 / jproblem.number_of_variables, distribution_index=20)
        term = StoppingByEvaluations(max_evaluations=self.max_evaluations)

        if self.algorithm in ("nsga2", "nsgaii"):
            from jmetal.algorithm.multiobjective import NSGAII
            return NSGAII(
                problem=jproblem,
                population_size=self.kwargs.get("population_size", 20),
                offspring_population_size=self.kwargs.get("offspring_population_size", 30),
                mutation=mut,
                crossover=SBXCrossover(probability=1.0, distribution_index=20),
                termination_criterion=term,
            )
        if self.algorithm == "gde3":
            from jmetal.algorithm.multiobjective.gde3 import GDE3
            return GDE3(
                problem=jproblem,
                population_size=self.population_size,
                cr=self.kwargs.get("cr", 0.5),
                f=self.kwargs.get("f", 0.5),
                termination_criterion=term,
            )
        if self.algorithm in ("smpso", "pso"):
            from jmetal.algorithm.multiobjective.smpso import SMPSO
            from jmetal.util.archive import CrowdingDistanceArchive
            return SMPSO(
                problem=jproblem,
                swarm_size=self.population_size,
                mutation=mut,
                leaders=CrowdingDistanceArchive(self.population_size),
                termination_criterion=term,
            )
        raise ValueError(f"Unknown jMetal algorithm '{self.algorithm}'.")

    def optimize(self, problem: OrugaProblem, seed: Optional[int] = None) -> OptimizationResult:
        from jmetal.util.solution import get_non_dominated_solutions

        if seed is not None:
            import random as _random
            _random.seed(seed)

        start = time.perf_counter()
        jproblem = self._build_problem(problem)
        algorithm = self._build_algorithm(jproblem)
        algorithm.run()
        solutions = get_non_dominated_solutions(algorithm.get_result())

        front: List[Individual] = [self._evaluate(problem, list(s.variables)) for s in solutions]
        front = non_dominated(front)
        best = self._best_by_scalar(front)
        runtime = time.perf_counter() - start
        return OptimizationResult(
            algorithm=self.name, problem=problem, best=best, front=front,
            history=[], n_evaluations=self.max_evaluations, runtime=runtime,
        )
