"""Solution encoding: candidate detection and genotype -> text decoding.

This is the single source of truth for ORUGA's representation. Each script in
the original codebase re-implemented the same three pieces of logic:

1. **Candidate detection** -- scan the tokens and decide which words may be
   replaced (skip capitalised words, very short words, etc.).
2. **The integer coding scheme** -- a solution is a vector with one gene per
   token; gene value ``v`` selects the ``v``-th synonym (``v < 1`` keeps the
   original word).
3. **Reconstruction** -- turn the chosen tokens back into a clean string
   (handle punctuation, underscores, etc.).

:class:`CandidateText` does all three, once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

from .synonyms import SynonymProvider


@dataclass
class CandidateText:
    """Pre-processed text exposing the optimizer's search space.

    Parameters
    ----------
    text:
        The original text to optimize.
    provider:
        A :class:`~oruga.synonyms.SynonymProvider` used to fetch candidates.
    min_length:
        Minimum token length (excluding punctuation) to be replaceable.
    skip_capitalized:
        If ``True``, tokens beginning with an uppercase letter are not replaced
        (a cheap proper-noun guard, matching the original heuristic).
    skip_suffixes:
        Optional tuple of suffixes (e.g. ``("ed",)``) whose tokens are skipped,
        mirroring the Word2Vec variant in the paper.
    """

    text: str
    provider: SynonymProvider
    min_length: int = 4
    skip_capitalized: bool = True
    skip_suffixes: Sequence[str] = ()

    tokens: List[str] = field(init=False)
    base: List[str] = field(init=False)
    punct: List[str] = field(init=False)
    candidates: List[List[str]] = field(init=False)
    is_candidate: List[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = self.text.split()
        self.base = []
        self.punct = []
        self.candidates = []
        self.is_candidate = []

        for token in self.tokens:
            stripped, trailing = _split_trailing_punct(token)
            self.base.append(stripped)
            self.punct.append(trailing)

            if self._eligible(stripped):
                syns = self.provider.synonyms(stripped)
            else:
                syns = []
            self.candidates.append(syns)
            self.is_candidate.append(bool(syns))

    def _eligible(self, word: str) -> bool:
        if len(word) < self.min_length:
            return False
        if self.skip_capitalized and word[:1].isupper():
            return False
        if any(word.endswith(suf) for suf in self.skip_suffixes):
            return False
        return True

    # -- search-space description ------------------------------------------
    @property
    def n_variables(self) -> int:
        """One gene per token (non-candidate genes are simply ignored)."""
        return len(self.tokens)

    @property
    def candidate_indices(self) -> List[int]:
        return [i for i, c in enumerate(self.is_candidate) if c]

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_indices)

    def max_choice(self, index: int) -> int:
        """Number of synonyms available at token ``index`` (0 if none)."""
        return len(self.candidates[index])

    # -- decoding ----------------------------------------------------------
    def decode_tokens(self, solution: Sequence[float], *, highlight: bool = False) -> List[str]:
        """Return the list of output tokens for a ``solution`` vector."""
        out: List[str] = []
        for i, token_base in enumerate(self.base):
            word = token_base
            gene = solution[i] if i < len(solution) else 0
            if self.is_candidate[i] and gene is not None and gene >= 1:
                syns = self.candidates[i]
                choice = min(int(round(gene)), len(syns))
                replacement = syns[choice - 1]
                word = replacement.upper() if highlight else replacement
            out.append(word + self.punct[i])
        return out

    def decode(self, solution: Sequence[float], *, highlight: bool = False) -> str:
        """Return the reconstructed text for a ``solution`` vector."""
        return _join_tokens(self.decode_tokens(solution, highlight=highlight))

    # -- change accounting -------------------------------------------------
    def count_active(self, solution: Sequence[float]) -> int:
        """Genes that *request* a replacement (gene >= 1 at a candidate slot).

        Matches the "words to be replaced" objective from the paper.
        """
        return sum(
            1
            for i, gene in enumerate(solution)
            if i < len(self.is_candidate) and self.is_candidate[i] and gene >= 1
        )

    def count_changes(self, solution: Sequence[float]) -> int:
        """Tokens whose surface form actually differs from the original."""
        decoded = self.decode_tokens(solution)
        return sum(1 for original, new in zip(self.tokens, decoded) if original != new)

    def modification_rate(self, solution: Sequence[float]) -> float:
        """Fraction of candidate tokens that were actually changed."""
        if self.n_candidates == 0:
            return 0.0
        return self.count_changes(solution) / self.n_candidates


def _split_trailing_punct(token: str):
    """Split a whitespace token into (word, trailing punctuation).

    Mirrors the original handling of trailing commas/periods while also
    tolerating other sentence punctuation.
    """
    i = len(token)
    while i > 0 and token[i - 1] in ".,;:!?’'\")]}":
        i -= 1
    return token[:i], token[i:]


def _join_tokens(tokens: List[str]) -> str:
    """Reconstruct a clean string from output tokens.

    Replaces underscores (multi-word WordNet/Word2Vec lemmas) with spaces and
    normalises spacing before punctuation, matching the legacy ``listToString``.
    """
    text = " ".join(str(t) for t in tokens)
    text = text.replace(" ,", ",").replace("_", " ")
    return text.strip()
