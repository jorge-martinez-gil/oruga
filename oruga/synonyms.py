"""Pluggable synonym providers.

Every legacy script reimplemented its own ``Synonym(word, number)`` helper bound
to a single lexical resource. This module replaces all of them with a single
:class:`SynonymProvider` interface and concrete providers for the three sources
used in the paper -- WordNet, Word2Vec and a web thesaurus -- plus an in-memory
:class:`DictionarySynonymProvider` used for deterministic tests and reproducible
demos.

A provider only has to answer one question: *what are the candidate
replacements for this word, in preference order?* The index/clamping logic that
used to live inside every ``Synonym`` function now lives once in
:mod:`oruga.encoding`.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Optional

__all__ = [
    "SynonymProvider",
    "DictionarySynonymProvider",
    "WordNetSynonymProvider",
    "Word2VecSynonymProvider",
    "WebSynonymProvider",
    "get_provider",
    "PROVIDERS",
]


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for item in items:
        if item not in seen:
            seen[item] = None
    return list(seen)


class SynonymProvider(ABC):
    """Abstract source of synonym candidates for a single word."""

    name: str = "abstract"

    @abstractmethod
    def synonyms(self, word: str) -> List[str]:
        """Return ordered candidate replacements for ``word`` (may be empty)."""

    def __call__(self, word: str) -> List[str]:  # convenience
        return self.synonyms(word)


class DictionarySynonymProvider(SynonymProvider):
    """Deterministic, dependency-free provider backed by an in-memory mapping.

    Ideal for unit tests, tutorials and fully reproducible demonstrations where
    network access or large models are undesirable.
    """

    name = "dictionary"

    def __init__(self, mapping: Optional[Dict[str, List[str]]] = None, *, lower: bool = True):
        self.lower = lower
        self.mapping = {}
        for key, values in (mapping or {}).items():
            self.mapping[key.lower() if lower else key] = list(values)

    def synonyms(self, word: str) -> List[str]:
        key = word.lower() if self.lower else word
        return list(self.mapping.get(key, []))


class WordNetSynonymProvider(SynonymProvider):
    """Synonyms from the NLTK WordNet lexical database.

    Requires the optional ``nltk`` dependency and the ``wordnet`` corpus. By
    default the candidate list is de-duplicated and the source word is removed,
    which improves replacement quality relative to the raw legacy behaviour.
    """

    name = "wordnet"

    def __init__(self, *, unique: bool = True, exclude_self: bool = True):
        self.unique = unique
        self.exclude_self = exclude_self
        self._wordnet = None

    def _load(self):
        if self._wordnet is None:
            try:
                from nltk.corpus import wordnet
                wordnet.synsets("test")  # triggers LookupError if corpus missing
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "WordNetSynonymProvider requires nltk. Install with "
                    "`pip install oruga[wordnet]`."
                ) from exc
            except LookupError as exc:  # pragma: no cover - data guard
                raise LookupError(
                    "WordNet corpus not found. Download it with "
                    "`python -m nltk.downloader wordnet omw-1.4`."
                ) from exc
            self._wordnet = wordnet
        return self._wordnet

    def synonyms(self, word: str) -> List[str]:
        wordnet = self._load()
        candidates: List[str] = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                candidates.append(lemma.name())
        if self.exclude_self:
            candidates = [c for c in candidates if c.lower() != word.lower()]
        if self.unique:
            candidates = _dedupe(candidates)
        return candidates


class Word2VecSynonymProvider(SynonymProvider):
    """Synonyms from a Word2Vec / KeyedVectors model (default Google News 300).

    Requires the optional ``gensim`` dependency. The first call downloads the
    model via ``gensim.downloader`` (cached afterwards).
    """

    name = "word2vec"

    def __init__(self, model_name: str = "word2vec-google-news-300", topn: int = 6):
        self.model_name = model_name
        self.topn = topn
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import gensim.downloader as api
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Word2VecSynonymProvider requires gensim. Install with "
                    "`pip install oruga[word2vec]`."
                ) from exc
            self._model = api.load(self.model_name)
        return self._model

    def synonyms(self, word: str) -> List[str]:
        model = self._load()
        try:
            return [w for w, _ in model.most_similar(word, topn=self.topn)]
        except KeyError:
            return []


class WebSynonymProvider(SynonymProvider):
    """Synonyms scraped from the thesaurus.com data endpoint.

    Requires the optional ``requests`` dependency and network access. Use
    responsibly to avoid rate limiting; results are cached per process.
    """

    name = "web"
    ENDPOINT = "https://tuna.thesaurus.com/pageData/{word}"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    @functools.lru_cache(maxsize=4096)
    def _fetch(self, word: str) -> tuple:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "WebSynonymProvider requires requests. Install with "
                "`pip install oruga[web]`."
            ) from exc
        try:
            resp = requests.get(self.ENDPOINT.format(word=word), timeout=self.timeout)
            data = resp.json()["data"]["definitionData"]["definitions"][0]["synonyms"]
            return tuple(item["term"] for item in data)
        except Exception:  # network / parsing errors -> no candidates
            return tuple()

    def synonyms(self, word: str) -> List[str]:
        return list(self._fetch(word))


PROVIDERS = {
    "dictionary": DictionarySynonymProvider,
    "wordnet": WordNetSynonymProvider,
    "word2vec": Word2VecSynonymProvider,
    "web": WebSynonymProvider,
}


def get_provider(name: str, **kwargs) -> SynonymProvider:
    """Instantiate a provider by name (``dictionary``/``wordnet``/``word2vec``/``web``)."""
    key = name.lower()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown synonym provider '{name}'. Available: {sorted(PROVIDERS)}")
    return PROVIDERS[key](**kwargs)
