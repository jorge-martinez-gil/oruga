# Contributing to ORUGA

Thanks for your interest in improving ORUGA! The goal of this project is to be
the open reference platform for readability optimization and benchmarking, so
contributions that increase scientific usefulness and reproducibility are
especially valued.

## Development setup

```bash
git clone https://github.com/jorge-martinez-gil/oruga.git
cd oruga
pip install -e ".[dev]"
pytest -q
```

The core engine depends only on NumPy; everything else is an optional extra. The
test suite is designed to run **without any heavy dependency or network access**
(it uses the in-memory `DictionarySynonymProvider`), so `pytest` should pass on a
clean machine.

## High-value contributions

- **New synonym providers** — subclass `oruga.synonyms.SynonymProvider` and add
  it to the `PROVIDERS` registry.
- **New optimizers** — subclass `oruga.optimizers.base.Optimizer` (return an
  `OptimizationResult`) and register it in `oruga/optimizers/__init__.py`.
- **New readability or evaluation metrics** — add to `oruga/readability.py` or a
  new objective in `oruga/objectives.py`.
- **Benchmark datasets** — domain corpora (health, legal, education, news, ...).
- **Baselines** — rule-based, transformer-based or LLM-based simplifiers wrapped
  behind the same interface for fair comparison.

## Guidelines

1. **Add a test** for any new behaviour. Keep tests deterministic (use seeds and
   the dictionary provider where possible).
2. **No fabricated results.** Every benchmark number must be reproducible from
   code in the repository.
3. **Keep the core light.** New third-party dependencies belong in an extra in
   `pyproject.toml`, imported lazily.
4. **Document** new options in the README and relevant docstrings.
5. Run `pytest -q` before opening a pull request.

## Reporting issues

Please include the ORUGA version (`python -c "import oruga; print(oruga.__version__)"`),
your Python version, the command/config used, and the full traceback.
