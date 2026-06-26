# Legacy scripts (original paper implementation)

This folder contains the **original, unmodified scripts** that accompanied the
*Knowledge-Based Systems* (2024) paper. They are preserved verbatim so the
published experiments remain **exactly reproducible**.

> For new work, use the unified `oruga` package in the repository root
> (`pip install -e .`). It consolidates everything below into a single,
> configurable, tested engine. See the top-level [README](../README.md).

## What is here

| File | Synonym source | Algorithm | Objectives |
| :--- | :--- | :--- | :--- |
| `oruga_wordnet.py` | WordNet | PyGAD GA | FKGL |
| `oruga_word2vec.py` | Word2Vec | PyGAD GA | FKGL |
| `oruga_webscraping.py` | Web thesaurus | PyGAD GA | FKGL |
| `oruga2_nsga2.py` | WordNet | NSGA-II | FKGL + modification rate |
| `oruga2_gde3.py` | WordNet | GDE3 | FKGL + modification rate |
| `oruga3_nsga2_wmd.py` | WordNet | NSGA-II | FKGL + modification + WMD |
| `oruga3_gde3_wmd.py` | WordNet | GDE3 | FKGL + modification + WMD |
| `oruga_massive_experiments.py` | mixed | batch | FKGL |
| `oruga_massive_experiments_smog.py` | mixed | batch | SMOG |
| `comparison/{web,word2vec,wordnet}/{cs,gde3,jaya,nsga2,pso,tlbo}.py` | all three | 6 metaheuristics | FKGL + modification + WMD |

## Running the legacy scripts

```bash
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
python legacy/oruga_wordnet.py
```

The same experiments can now be expressed in a few lines with the new engine —
see `examples/reproduce_paper.py`.
