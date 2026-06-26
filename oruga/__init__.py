"""ORUGA -- Optimizing Readability Using Genetic (and other) Algorithms.

A unified, config-driven engine for automatic, semantics-preserving readability
optimization and benchmarking. The package consolidates what used to be ~27
near-identical scripts into composable building blocks:

    text + SynonymProvider  ->  CandidateText (encoding)
                            ->  OrugaProblem (objectives)
                            ->  Optimizer    (GA / NSGA-II / GDE3 / SMPSO /
                                              Jaya / Cuckoo / TLBO / ...)
                            ->  OptimizationReport (before/after metrics)

Quickstart
----------
>>> from oruga import optimize
>>> report = optimize(
...     "The committee deliberated extensively regarding the ramifications.",
...     provider="wordnet", optimizer="ga", seed=0,
... )
>>> report.optimized_text          # doctest: +SKIP
>>> report.improvement("fkgl")     # doctest: +SKIP

Cite
----
Martinez-Gil, J. (2024). Optimizing readability using genetic algorithms.
Knowledge-Based Systems, 284, 111273. https://doi.org/10.1016/j.knosys.2023.111273
"""

from __future__ import annotations

__version__ = "2.0.0"

from . import readability
from .config import OrugaConfig
from .corpus import SAMPLE_TEXTS, load_corpus
from .encoding import CandidateText
from .objectives import (
    ModificationObjective,
    Objective,
    ReadabilityObjective,
    SemanticDistanceObjective,
    build_objectives,
)
from .optimizers import (
    OptimizationResult,
    get_optimizer,
    list_optimizers,
)
from .pipeline import OptimizationReport, optimize, optimize_corpus
from .problem import OrugaProblem
from .synonyms import (
    DictionarySynonymProvider,
    SynonymProvider,
    Word2VecSynonymProvider,
    WebSynonymProvider,
    WordNetSynonymProvider,
    get_provider,
)

CITATION = (
    "Martinez-Gil, J. (2024). Optimizing readability using genetic algorithms. "
    "Knowledge-Based Systems, 284, 111273. "
    "https://doi.org/10.1016/j.knosys.2023.111273"
)

__all__ = [
    "__version__",
    "CITATION",
    "optimize",
    "optimize_corpus",
    "OptimizationReport",
    "OrugaConfig",
    "OrugaProblem",
    "CandidateText",
    "Objective",
    "ReadabilityObjective",
    "ModificationObjective",
    "SemanticDistanceObjective",
    "build_objectives",
    "SynonymProvider",
    "DictionarySynonymProvider",
    "WordNetSynonymProvider",
    "Word2VecSynonymProvider",
    "WebSynonymProvider",
    "get_provider",
    "get_optimizer",
    "list_optimizers",
    "OptimizationResult",
    "readability",
    "load_corpus",
    "SAMPLE_TEXTS",
]
