"""Reproduce the paper's experiments through the unified engine.

This mirrors the original single-objective (WordNet + GA) and multi-objective
(NSGA-II / GDE3 with semantic protection) runs, but in a handful of lines
instead of a separate script per combination.

Requirements (install what you need):
    pip install oruga[wordnet]        # WordNet provider
    pip install oruga[pygad]          # paper-faithful GA backend
    pip install oruga[jmetal,word2vec]# NSGA-II / GDE3 + Word Mover's Distance
    python -m nltk.downloader wordnet omw-1.4

    python examples/reproduce_paper.py
"""

from oruga import OrugaConfig, load_corpus, optimize

texts = load_corpus()  # the 10 Wikipedia samples from texts.txt

# --- Single-objective: WordNet + GA, minimize FKGL (paper's ORUGA) ---------
single = OrugaConfig(provider="wordnet", optimizer="pygad", metric="fkgl",
                     optimizer_kwargs={"num_generations": 100, "sol_per_pop": 20})

# --- Multi-objective: WordNet + NSGA-II, FKGL + modification rate (ORUGA-2) -
multi = OrugaConfig(provider="wordnet", optimizer="nsga2", metric="fkgl",
                    modification=True,
                    optimizer_kwargs={"max_evaluations": 3000})

# --- Tri-objective with Word Mover's Distance semantic protection (ORUGA-3) -
wmd = OrugaConfig(provider="wordnet", optimizer="gde3", metric="fkgl",
                  modification=True, semantic=True, semantic_backend="wmd",
                  optimizer_kwargs={"max_evaluations": 3000})

text = texts[0]
for name, cfg in [("single/GA", single), ("multi/NSGA-II", multi), ("tri/GDE3+WMD", wmd)]:
    report = optimize(text, config=cfg, seed=0)
    print(f"\n[{name}] FKGL {report.metrics_before['fkgl']:.2f} -> "
          f"{report.metrics_after['fkgl']:.2f}")
    print("  ", report.optimized_text)
