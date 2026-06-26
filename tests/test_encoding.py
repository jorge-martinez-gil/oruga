"""Encoding: candidate detection, decoding and change accounting."""

from oruga.encoding import CandidateText


def test_zero_solution_returns_original(provider, complex_text):
    ct = CandidateText(complex_text, provider)
    zero = [0] * ct.n_variables
    # Whitespace is normalized to single spaces by reconstruction.
    assert ct.decode(zero) == " ".join(complex_text.split())


def test_capitalized_and_short_words_skipped(provider):
    ct = CandidateText("The committee met", provider)
    # "The" is capitalized -> not a candidate; "met" len 3 -> too short.
    assert ct.is_candidate[0] is False
    assert ct.is_candidate[2] is False
    # "committee" is a candidate (has synonyms, lowercase, long enough).
    assert ct.is_candidate[1] is True


def test_decode_applies_synonym_choice(provider):
    ct = CandidateText("the committee", provider)
    # gene 1 selects first synonym of "committee" -> "board".
    out = ct.decode([0, 1])
    assert "board" in out


def test_choice_is_clamped(provider):
    ct = CandidateText("the committee", provider)
    # Only 2 synonyms exist; a large gene clamps to the last one.
    assert "group" in ct.decode([0, 9])


def test_punctuation_preserved(provider):
    ct = CandidateText("the committee, again.", provider)
    out = ct.decode([0, 1, 0])
    assert "board," in out
    assert out.endswith(".")


def test_count_active_and_changes(provider):
    ct = CandidateText("the committee deliberated", provider)
    solution = [0, 1, 1]
    assert ct.count_active(solution) == 2
    assert ct.count_changes(solution) == 2
    assert 0.0 < ct.modification_rate(solution) <= 1.0


def test_underscore_normalized():
    from oruga.synonyms import DictionarySynonymProvider
    prov = DictionarySynonymProvider({"city": ["new_york"]})
    ct = CandidateText("the city", prov)
    assert ct.decode([0, 1]) == "the new york"
