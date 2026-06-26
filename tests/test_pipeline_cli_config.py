"""Pipeline, config round-trip, and CLI smoke tests."""

import json

import pytest

from oruga import OrugaConfig, optimize
from oruga.cli import main
from oruga.synonyms import DictionarySynonymProvider


def test_pipeline_report_fields(provider, complex_text):
    report = optimize(complex_text, provider=provider, optimizer="ga", seed=0,
                      modification=True, optimizer_kwargs={"generations": 20})
    assert report.optimized_text
    assert "fkgl" in report.metrics_before and "fkgl" in report.metrics_after
    d = report.to_dict()
    assert d["algorithm"] == "ga"
    assert len(d["objective_values"]) == 2  # modification + readability


def test_config_roundtrip_dict():
    cfg = OrugaConfig(provider="wordnet", optimizer="nsga2", modification=True, seed=7)
    again = OrugaConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_config_roundtrip_yaml(tmp_path):
    pytest.importorskip("yaml")
    cfg = OrugaConfig(provider="dictionary", optimizer="ga", semantic=True)
    path = tmp_path / "c.yaml"
    cfg.save(str(path))
    loaded = OrugaConfig.from_file(str(path))
    assert loaded == cfg


def test_cli_metrics_json(capsys):
    rc = main(["metrics", "--text", "This is a simple test sentence.", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "fkgl" in out


def test_cli_list(capsys):
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Optimizers" in out and "ga" in out


def test_cli_optimize_dictionary(capsys):
    # provider defaults to wordnet in the CLI; use random optimizer on a
    # dictionary-friendly text path via the Python API instead for no-deps.
    rc = main([
        "optimize", "--text", "the committee deliberated extensively",
        "--provider", "dictionary", "--optimizer", "random", "--seed", "0",
        "--opt-arg", "evaluations=50", "--json",
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "optimized_text" in out
