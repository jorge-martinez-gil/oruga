"""Benchmark corpus loading.

Loads the Wikipedia sample set shipped in ``texts.txt`` (the dataset used in the
paper) and provides a couple of tiny built-in passages for tests and examples
that must run without any external file.
"""

from __future__ import annotations

import os
from typing import List

__all__ = ["load_corpus", "default_corpus_path", "SAMPLE_TEXTS"]

SAMPLE_TEXTS: List[str] = [
    (
        "The committee deliberated extensively regarding the ramifications of the "
        "proposed legislation before reaching a consensus on its implementation."
    ),
    (
        "Photosynthesis is the process whereby vegetation converts luminous energy "
        "into chemical energy, subsequently sustaining numerous terrestrial organisms."
    ),
]


def default_corpus_path() -> str:
    """Best-effort path to the bundled ``texts.txt`` benchmark file."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "data", "texts.txt"),
        os.path.join(here, "..", "texts.txt"),
        os.path.join(os.getcwd(), "texts.txt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.abspath(candidates[1])


def load_corpus(path: str | None = None) -> List[str]:
    """Return the list of benchmark texts.

    Each non-empty line of the file is one text; surrounding double quotes and
    whitespace are stripped. Falls back to :data:`SAMPLE_TEXTS` if no file is
    found.
    """
    path = path or default_corpus_path()
    if not os.path.isfile(path):
        return list(SAMPLE_TEXTS)
    texts: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            texts.append(line.strip())
    return texts
