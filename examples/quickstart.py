"""Quickstart: optimize one sentence with zero heavy dependencies.

Uses the in-memory DictionarySynonymProvider and the built-in GA, so it runs
anywhere without WordNet, Word2Vec, jMetal or Java.

    python examples/quickstart.py
"""

from oruga import optimize
from oruga.synonyms import DictionarySynonymProvider

text = (
    "The committee deliberated extensively regarding the ramifications of the "
    "proposed legislation before reaching a consensus on its implementation."
)

provider = DictionarySynonymProvider({
    "deliberated": ["talked", "thought"],
    "extensively": ["widely", "fully"],
    "regarding": ["about", "on"],
    "ramifications": ["results", "effects"],
    "proposed": ["planned", "new"],
    "legislation": ["law", "rule"],
    "reaching": ["getting", "making"],
    "consensus": ["agreement", "accord"],
    "implementation": ["use", "rollout"],
})

report = optimize(text, provider=provider, optimizer="ga", seed=0,
                  optimizer_kwargs={"generations": 40})

print("Original :", report.original_text)
print("Optimized:", report.optimized_text)
print(f"FKGL {report.metrics_before['fkgl']:.2f} -> {report.metrics_after['fkgl']:.2f} "
      f"(improvement {report.improvement('fkgl'):+.2f})")
