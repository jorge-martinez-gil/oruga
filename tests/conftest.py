"""Shared fixtures: a deterministic, dependency-free problem."""

import pytest

from oruga.synonyms import DictionarySynonymProvider

COMPLEX_TEXT = (
    "The committee deliberated extensively regarding the ramifications of the "
    "proposed legislation before reaching a consensus on its implementation."
)

# Synonyms chosen to be simpler (fewer syllables) so optimization clearly helps.
SYNONYMS = {
    "committee": ["board", "group"],
    "deliberated": ["talked", "met"],
    "extensively": ["widely", "a lot"],
    "regarding": ["about", "on"],
    "ramifications": ["results", "effects"],
    "proposed": ["planned", "new"],
    "legislation": ["law", "rule"],
    "before": ["ere"],
    "reaching": ["getting", "making"],
    "consensus": ["accord", "deal"],
    "implementation": ["use", "rollout"],
}


@pytest.fixture
def provider():
    return DictionarySynonymProvider(SYNONYMS)


@pytest.fixture
def complex_text():
    return COMPLEX_TEXT
