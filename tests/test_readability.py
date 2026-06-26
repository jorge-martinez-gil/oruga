"""Readability metric correctness, including offline cross-validation."""

import math

import pytest

from oruga import readability as rd
from oruga.corpus import load_corpus


def test_fkgl_known_value():
    # "The cat sat on the mat." -> 1 sentence, 6 words, 6 syllables.
    # FKGL = 0.39*(6/1) + 11.8*(6/6) - 15.59 = -1.45
    stats = rd.TextStatistics("The cat sat on the mat.")
    assert stats.n_words == 6
    assert stats.n_sentences == 1
    assert stats.n_syllables == 6
    assert rd.flesch_kincaid_grade(stats) == pytest.approx(-1.45, abs=1e-6)


def test_simple_text_is_easier_than_complex():
    simple = "The dog ran. The sun is hot. We eat food."
    complex_ = (
        "The aforementioned bureaucratic infrastructure necessitates "
        "comprehensive multidisciplinary reconfiguration."
    )
    assert rd.score(simple, "fkgl") < rd.score(complex_, "fkgl")
    # Flesch Reading Ease is inverted: higher = easier.
    assert rd.score(simple, "flesch") > rd.score(complex_, "flesch")


def test_all_scores_keys():
    scores = rd.all_scores("This is a short and simple test sentence for scoring.")
    assert set(scores) == set(rd.METRICS)
    assert all(isinstance(v, float) for v in scores.values())


def test_syllable_counter_floor():
    assert rd.count_syllables("a") == 1
    assert rd.count_syllables("strength") == 1
    assert rd.count_syllables("readability") >= 4


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        rd.score("text", "not_a_metric")


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (sa * sb)


def test_cross_validation_against_pyphen():
    """Built-in FKGL must track an independent syllable reference (pyphen)."""
    pyphen = pytest.importorskip("pyphen")
    dic = pyphen.Pyphen(lang="en_US")
    texts = [t for t in load_corpus() if len(t.split()) > 30]
    assert texts, "corpus should contain long-enough texts"

    def fk_ref(text):
        stats = rd.TextStatistics(text)
        syllables = sum(
            max(1, dic.inserted(w).count("-") + 1) for w in rd.tokenize_words(text)
        )
        return 0.39 * stats.words_per_sentence + 11.8 * (syllables / stats.n_words) - 15.59

    ours = [rd.score(t, "fkgl") for t in texts]
    ref = [fk_ref(t) for t in texts]

    assert _pearson(ours, ref) > 0.9
    mae = sum(abs(a - b) for a, b in zip(ours, ref)) / len(ours)
    assert mae < 2.5, f"mean abs FKGL difference vs pyphen too high: {mae:.2f}"
