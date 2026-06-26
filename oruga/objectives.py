"""Composable optimization objectives.

The legacy scripts hard-wired one to three objectives directly inside each
``fitness_func`` / jMetal ``evaluate`` method. Here each objective is a small,
reusable object exposing two things:

* :meth:`Objective.value` -- the human-meaningful number (e.g. the FKGL grade,
  the number of replaced words, the semantic distance);
* :meth:`Objective.cost` -- the same quantity transformed so that *lower is
  always better*, which is what every optimizer minimizes.

Single-objective and multi-objective runs are then just different *lists* of
objectives, which is exactly what made the original six-algorithm comparison so
repetitive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from .encoding import CandidateText
from . import readability as rd

__all__ = [
    "Objective",
    "ReadabilityObjective",
    "ModificationObjective",
    "SemanticDistanceObjective",
    "build_objectives",
]


class Objective(ABC):
    name: str = "objective"
    label: str = "objective"

    @abstractmethod
    def value(self, ct: CandidateText, solution: Sequence[float]) -> float:
        """The interpretable value of this objective for ``solution``."""

    def cost(self, ct: CandidateText, solution: Sequence[float]) -> float:
        """The value transformed so that *lower is better* (default: identity)."""
        return self.value(ct, solution)


class ReadabilityObjective(Objective):
    """Readability of the decoded text under a chosen metric."""

    def __init__(self, metric: str = "fkgl", backend: str = "builtin"):
        self.metric = metric
        self.backend = backend
        self.name = metric
        meta = rd.METRICS.get(metric, {"name": metric, "lower_is_easier": True})
        self.label = str(meta["name"])
        self._lower_is_easier = bool(meta["lower_is_easier"])

    def value(self, ct: CandidateText, solution: Sequence[float]) -> float:
        return rd.score(ct.decode(solution), self.metric, backend=self.backend)

    def cost(self, ct: CandidateText, solution: Sequence[float]) -> float:
        v = self.value(ct, solution)
        # If higher means easier (e.g. Flesch Reading Ease), negate so the
        # optimizer's minimization still pushes toward easier text.
        return v if self._lower_is_easier else -v


class ModificationObjective(Objective):
    """How much the text was changed (to be kept small)."""

    name = "modifications"
    label = "Words replaced"

    def __init__(self, mode: str = "active"):
        if mode not in ("active", "changes", "rate"):
            raise ValueError("mode must be 'active', 'changes' or 'rate'")
        self.mode = mode
        self.label = {"active": "Words to replace", "changes": "Words changed", "rate": "Modification rate"}[mode]

    def value(self, ct: CandidateText, solution: Sequence[float]) -> float:
        if self.mode == "active":
            return float(ct.count_active(solution))
        if self.mode == "changes":
            return float(ct.count_changes(solution))
        return ct.modification_rate(solution)


class SemanticDistanceObjective(Objective):
    """Semantic drift between the original and the simplified text.

    ``backend="wmd"`` reproduces the Word Mover's Distance used in the paper
    (needs ``gensim`` + a Word2Vec model). ``backend="jaccard"`` is a
    dependency-free token-overlap proxy useful for tests and quick demos.
    """

    name = "semantic_distance"
    label = "Semantic distance"

    def __init__(self, backend: str = "jaccard", model_name: str = "word2vec-google-news-300"):
        self.backend = backend
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            import gensim.downloader as api  # lazy, optional
            self._model = api.load(self.model_name)
        return self._model

    def value(self, ct: CandidateText, solution: Sequence[float]) -> float:
        target = ct.decode(solution)
        if self.backend == "wmd":
            model = self._load_model()
            return float(model.wmdistance(ct.text, target))
        # Jaccard distance over lowercased word sets.
        a = set(rd.tokenize_words(ct.text.lower()))
        b = set(rd.tokenize_words(target.lower()))
        if not a and not b:
            return 0.0
        return 1.0 - len(a & b) / len(a | b)


def build_objectives(
    metric: str = "fkgl",
    *,
    readability_backend: str = "builtin",
    modification: bool = False,
    modification_mode: str = "active",
    semantic: bool = False,
    semantic_backend: str = "jaccard",
) -> List[Objective]:
    """Assemble a list of objectives from a compact configuration.

    * single-objective readability: ``build_objectives("fkgl")``
    * bi-objective (paper's ORUGA-2): ``build_objectives("fkgl", modification=True)``
    * tri-objective (paper's ORUGA-3 / WMD): add ``semantic=True``
    """
    objectives: List[Objective] = []
    if modification:
        objectives.append(ModificationObjective(mode=modification_mode))
    objectives.append(ReadabilityObjective(metric=metric, backend=readability_backend))
    if semantic:
        objectives.append(SemanticDistanceObjective(backend=semantic_backend))
    return objectives
