"""The optimization problem: text + provider + objectives.

:class:`OrugaProblem` is the uniform contract every optimizer consumes. It hides
the encoding and objective machinery behind three methods -- :meth:`bounds`,
:meth:`costs` and :meth:`decode` -- so that a genetic algorithm, a swarm method
or a hand-rolled metaheuristic can all be pointed at the same object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .encoding import CandidateText
from .objectives import Objective, build_objectives
from .synonyms import SynonymProvider, get_provider


@dataclass
class OrugaProblem:
    candidate_text: CandidateText
    objectives: List[Objective]
    lower_bound: float = -4.0
    upper_bound: float = 4.0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_text(
        cls,
        text: str,
        provider: SynonymProvider | str = "dictionary",
        *,
        metric: str = "fkgl",
        readability_backend: str = "builtin",
        modification: bool = False,
        modification_mode: str = "active",
        semantic: bool = False,
        semantic_backend: str = "jaccard",
        min_length: int = 4,
        skip_capitalized: bool = True,
        skip_suffixes: Sequence[str] = (),
        lower_bound: float = -4.0,
        upper_bound: float = 4.0,
        provider_kwargs: dict | None = None,
    ) -> "OrugaProblem":
        if isinstance(provider, str):
            provider = get_provider(provider, **(provider_kwargs or {}))
        ct = CandidateText(
            text=text,
            provider=provider,
            min_length=min_length,
            skip_capitalized=skip_capitalized,
            skip_suffixes=skip_suffixes,
        )
        objectives = build_objectives(
            metric=metric,
            readability_backend=readability_backend,
            modification=modification,
            modification_mode=modification_mode,
            semantic=semantic,
            semantic_backend=semantic_backend,
        )
        return cls(ct, objectives, lower_bound=lower_bound, upper_bound=upper_bound)

    # -- search-space description ------------------------------------------
    @property
    def n_variables(self) -> int:
        return self.candidate_text.n_variables

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def is_multiobjective(self) -> bool:
        return self.n_objectives > 1

    def bounds(self) -> Tuple[List[float], List[float]]:
        n = self.n_variables
        return [self.lower_bound] * n, [self.upper_bound] * n

    # -- evaluation --------------------------------------------------------
    def costs(self, solution: Sequence[float]) -> List[float]:
        """Objective vector to be minimized (used by optimizers)."""
        return [obj.cost(self.candidate_text, solution) for obj in self.objectives]

    def values(self, solution: Sequence[float]) -> List[float]:
        """Interpretable objective vector (used for reporting)."""
        return [obj.value(self.candidate_text, solution) for obj in self.objectives]

    def scalar_cost(self, solution: Sequence[float]) -> float:
        """Sum of costs -- a simple scalarization for single-objective solvers."""
        return float(sum(self.costs(solution)))

    def labels(self) -> List[str]:
        return [obj.label for obj in self.objectives]

    # -- decoding ----------------------------------------------------------
    def decode(self, solution: Sequence[float], *, highlight: bool = False) -> str:
        return self.candidate_text.decode(solution, highlight=highlight)
