"""Dependency-free readability metrics.

This module centralizes the readability formulas that were previously accessed
through ``py-readability-metrics`` in every script. Implementing them in pure
Python removes the documented ``readability`` / ``readability-lxml`` namespace
conflict, makes the metrics deterministic and unit-testable, and lets the
optimizer run without any heavy dependency.

All formulas follow their standard published definitions. For users who need to
reproduce the *exact* numbers reported in the ORUGA paper, the original
``py-readability-metrics`` backend is still available via
:func:`oruga.readability.score` with ``backend="readability-metrics"`` and the
widely used ``textstat`` backend via ``backend="textstat"``.

References
----------
* Kincaid et al. (1975) -- Flesch-Kincaid Grade Level / Reading Ease.
* McLaughlin (1969) -- SMOG.
* Coleman & Liau (1975) -- Coleman-Liau Index.
* Smith & Senter (1967) -- Automated Readability Index.
* Gunning (1952) -- Gunning Fog Index.
* Dale & Chall (1948) -- Dale-Chall (requires the familiar-word list backend).
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List

__all__ = [
    "count_syllables",
    "tokenize_words",
    "split_sentences",
    "TextStatistics",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "smog",
    "coleman_liau",
    "automated_readability_index",
    "gunning_fog",
    "METRICS",
    "score",
    "all_scores",
]

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENT_RE = re.compile(r"[^.!?]+[.!?]*")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")


def tokenize_words(text: str) -> List[str]:
    """Return the alphabetic word tokens in ``text``."""
    return _WORD_RE.findall(text)


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences using terminal punctuation."""
    sentences = [s.strip() for s in _SENT_RE.findall(text)]
    return [s for s in sentences if s]


def count_syllables(word: str) -> int:
    """Estimate the number of syllables in an English ``word``.

    Uses the classic vowel-group heuristic with silent-``e`` correction. It is
    deterministic and dependency-free; for phonetic-dictionary accuracy use the
    ``textstat`` backend.
    """
    word = word.lower().strip()
    if not word:
        return 0
    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)
    # Silent trailing 'e' (but keep words like "the" at >= 1).
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in "aeiouy":
        count += 1
    return max(1, count)


class TextStatistics:
    """Pre-computed surface statistics shared by every readability formula."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.words = tokenize_words(text)
        self.sentences = split_sentences(text)
        self.syllables_per_word = [count_syllables(w) for w in self.words]

        self.n_words = max(1, len(self.words))
        self.n_sentences = max(1, len(self.sentences))
        self.n_syllables = sum(self.syllables_per_word)
        self.n_chars = sum(len(w) for w in self.words)
        self.n_polysyllables = sum(1 for s in self.syllables_per_word if s >= 3)

    @property
    def words_per_sentence(self) -> float:
        return self.n_words / self.n_sentences

    @property
    def syllables_per_word_mean(self) -> float:
        return self.n_syllables / self.n_words

    @property
    def chars_per_word(self) -> float:
        return self.n_chars / self.n_words


def flesch_kincaid_grade(stats: TextStatistics) -> float:
    """Flesch-Kincaid Grade Level (lower = easier)."""
    return (
        0.39 * stats.words_per_sentence
        + 11.8 * stats.syllables_per_word_mean
        - 15.59
    )


def flesch_reading_ease(stats: TextStatistics) -> float:
    """Flesch Reading Ease (higher = easier, 0-100 scale)."""
    return (
        206.835
        - 1.015 * stats.words_per_sentence
        - 84.6 * stats.syllables_per_word_mean
    )


def smog(stats: TextStatistics) -> float:
    """SMOG grade (lower = easier). Designed for >= 30 sentences."""
    return 1.043 * math.sqrt(stats.n_polysyllables * (30 / stats.n_sentences)) + 3.1291


def coleman_liau(stats: TextStatistics) -> float:
    """Coleman-Liau Index (lower = easier)."""
    letters_per_100 = stats.chars_per_word * 100
    sentences_per_100 = (stats.n_sentences / stats.n_words) * 100
    return 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8


def automated_readability_index(stats: TextStatistics) -> float:
    """Automated Readability Index (lower = easier)."""
    return 4.71 * stats.chars_per_word + 0.5 * stats.words_per_sentence - 21.43


def gunning_fog(stats: TextStatistics) -> float:
    """Gunning Fog Index (lower = easier)."""
    complex_ratio = stats.n_polysyllables / stats.n_words
    return 0.4 * (stats.words_per_sentence + 100 * complex_ratio)


# Registry of built-in metrics. ``lower_is_easier`` tells the optimizer which
# direction to push when this metric is the objective.
METRICS: Dict[str, Dict[str, object]] = {
    "fkgl": {"fn": flesch_kincaid_grade, "lower_is_easier": True, "name": "Flesch-Kincaid Grade Level"},
    "flesch": {"fn": flesch_reading_ease, "lower_is_easier": False, "name": "Flesch Reading Ease"},
    "smog": {"fn": smog, "lower_is_easier": True, "name": "SMOG"},
    "coleman_liau": {"fn": coleman_liau, "lower_is_easier": True, "name": "Coleman-Liau Index"},
    "ari": {"fn": automated_readability_index, "lower_is_easier": True, "name": "Automated Readability Index"},
    "gunning_fog": {"fn": gunning_fog, "lower_is_easier": True, "name": "Gunning Fog Index"},
}


def _builtin_score(text: str, metric: str) -> float:
    if metric not in METRICS:
        raise ValueError(
            f"Unknown metric '{metric}'. Available: {sorted(METRICS)}"
        )
    fn: Callable[[TextStatistics], float] = METRICS[metric]["fn"]  # type: ignore[assignment]
    return float(fn(TextStatistics(text)))


def _textstat_score(text: str, metric: str) -> float:
    import textstat  # lazy, optional

    mapping = {
        "fkgl": textstat.flesch_kincaid_grade,
        "flesch": textstat.flesch_reading_ease,
        "smog": textstat.smog_index,
        "coleman_liau": textstat.coleman_liau_index,
        "ari": textstat.automated_readability_index,
        "gunning_fog": textstat.gunning_fog,
        "dale_chall": textstat.dale_chall_readability_score,
    }
    if metric not in mapping:
        raise ValueError(f"Metric '{metric}' not supported by textstat backend.")
    return float(mapping[metric](text))


def _readability_metrics_score(text: str, metric: str) -> float:
    from readability import Readability  # lazy, optional (py-readability-metrics)

    r = Readability(text)
    mapping = {
        "fkgl": lambda: r.flesch_kincaid().score,
        "flesch": lambda: r.flesch().score,
        "smog": lambda: r.smog().score,
        "coleman_liau": lambda: r.coleman_liau().score,
        "ari": lambda: r.ari().score,
        "gunning_fog": lambda: r.gunning_fog().score,
        "dale_chall": lambda: r.dale_chall().score,
    }
    if metric not in mapping:
        raise ValueError(f"Metric '{metric}' not supported by readability-metrics backend.")
    return float(mapping[metric]())


def score(text: str, metric: str = "fkgl", backend: str = "builtin") -> float:
    """Compute a single readability ``metric`` for ``text``.

    Parameters
    ----------
    text:
        Input text.
    metric:
        Metric key, e.g. ``"fkgl"``, ``"flesch"``, ``"smog"``, ``"coleman_liau"``,
        ``"ari"``, ``"gunning_fog"`` (``"dale_chall"`` requires a non-builtin backend).
    backend:
        ``"builtin"`` (default, dependency-free), ``"textstat"`` or
        ``"readability-metrics"`` (the exact backend used in the paper).
    """
    if backend == "builtin":
        return _builtin_score(text, metric)
    if backend == "textstat":
        return _textstat_score(text, metric)
    if backend in ("readability-metrics", "py-readability-metrics"):
        return _readability_metrics_score(text, metric)
    raise ValueError(f"Unknown backend '{backend}'.")


def all_scores(text: str, backend: str = "builtin") -> Dict[str, float]:
    """Return every available built-in metric for ``text``."""
    return {key: score(text, key, backend=backend) for key in METRICS}
