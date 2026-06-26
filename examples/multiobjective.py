"""Multi-objective optimization and the Pareto front.

Reproduces the spirit of the paper's ORUGA-2 (readability + modification rate)
and ORUGA-3 (+ semantic distance) experiments through the unified API, swapping
algorithms with a single string. Runs dependency-free with the dictionary
provider; swap ``provider="wordnet"`` for real lexical data.

    python examples/multiobjective.py
"""

from oruga import OrugaProblem, get_optimizer
from oruga.synonyms import DictionarySynonymProvider

provider = DictionarySynonymProvider({
    "deliberated": ["talked", "thought"],
    "extensively": ["widely", "fully"],
    "regarding": ["about", "on"],
    "ramifications": ["results", "effects"],
    "proposed": ["planned", "new"],
    "legislation": ["law", "rule"],
    "consensus": ["agreement", "accord"],
    "implementation": ["use", "rollout"],
})

text = (
    "The committee deliberated extensively regarding the ramifications of the "
    "proposed legislation before reaching a consensus on its implementation."
)

# Tri-objective: words-to-replace, FKGL, semantic distance (all minimized).
problem = OrugaProblem.from_text(
    text, provider, metric="fkgl", modification=True, semantic=True,
)

for algo in ["ga", "jaya", "cuckoo", "tlbo"]:
    result = get_optimizer(algo, **({"generations": 30} if algo == "ga" else {"iterations": 30})).optimize(
        problem, seed=0
    )
    print(f"\n=== {algo.upper()} :: {result.summary()} ===")
    labels = problem.labels()
    for ind in sorted(result.front, key=lambda x: x.values[0])[:5]:
        cells = ", ".join(f"{lab}={val:.2f}" for lab, val in zip(labels, ind.values))
        print(f"  [{cells}]  {ind.text}")
