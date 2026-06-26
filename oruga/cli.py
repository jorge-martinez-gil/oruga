"""Command-line interface for ORUGA.

Examples
--------
    oruga optimize --text "The committee deliberated extensively." \
        --provider wordnet --optimizer ga --modification --seed 0

    oruga metrics --text "Some text to score."

    oruga list                 # show providers, optimizers and metrics
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__, readability
from .config import OrugaConfig
from .corpus import load_corpus
from .optimizers import list_optimizers
from .pipeline import optimize
from .synonyms import PROVIDERS


def _add_problem_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", default="wordnet", choices=sorted(PROVIDERS),
                   help="Synonym source (default: wordnet).")
    p.add_argument("--optimizer", default="ga",
                   help=f"Optimizer name. One of: {', '.join(list_optimizers())}.")
    p.add_argument("--metric", default="fkgl",
                   help="Readability metric to optimize (default: fkgl).")
    p.add_argument("--readability-backend", default="builtin",
                   choices=["builtin", "textstat", "readability-metrics"])
    p.add_argument("--modification", action="store_true",
                   help="Add the modification-rate objective (bi-objective).")
    p.add_argument("--semantic", action="store_true",
                   help="Add the semantic-distance objective (tri-objective).")
    p.add_argument("--semantic-backend", default="jaccard", choices=["jaccard", "wmd"])
    p.add_argument("--correct-grammar", action="store_true",
                   help="Run LanguageTool over the result (needs Java).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--config", default=None, help="Load an OrugaConfig YAML/JSON file.")
    p.add_argument("--opt-arg", action="append", default=[], metavar="KEY=VALUE",
                   help="Optimizer hyper-parameter, e.g. --opt-arg generations=80.")


def _parse_opt_args(pairs: List[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def _cmd_optimize(args) -> int:
    config = OrugaConfig.from_file(args.config) if args.config else OrugaConfig()
    overrides = dict(
        provider=args.provider,
        optimizer=args.optimizer,
        metric=args.metric,
        readability_backend=args.readability_backend,
        modification=args.modification,
        semantic=args.semantic,
        semantic_backend=args.semantic_backend,
        correct_grammar=args.correct_grammar,
        seed=args.seed,
    )
    opt_kwargs = _parse_opt_args(args.opt_arg)
    if opt_kwargs:
        overrides["optimizer_kwargs"] = {**config.optimizer_kwargs, **opt_kwargs}

    text = _resolve_text(args)
    if text is None:
        print("error: provide --text or --file", file=sys.stderr)
        return 2

    report = optimize(text, config=config, **overrides)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("Original :", report.original_text)
        print("Optimized:", report.optimized_text)
        for label, value in zip(report.result.problem.labels(), report.result.best_values):
            print(f"  {label}: {value:.3f}")
        print(f"  FKGL {report.metrics_before['fkgl']:.2f} -> {report.metrics_after['fkgl']:.2f} "
              f"(improvement {report.improvement('fkgl'):+.2f})")
        print(f"  Pareto front: {len(report.result.front)} solution(s) in {report.result.runtime:.2f}s")
    return 0


def _cmd_metrics(args) -> int:
    text = _resolve_text(args)
    if text is None:
        print("error: provide --text or --file", file=sys.stderr)
        return 2
    scores = readability.all_scores(text, backend=args.readability_backend)
    if args.json:
        print(json.dumps(scores, indent=2))
    else:
        for key, value in scores.items():
            print(f"{readability.METRICS[key]['name']:<32} {value:7.2f}")
    return 0


def _cmd_list(_args) -> int:
    print("Synonym providers:", ", ".join(sorted(PROVIDERS)))
    print("Optimizers       :", ", ".join(list_optimizers()))
    print("Readability metrics:", ", ".join(readability.METRICS))
    return 0


def _resolve_text(args) -> Optional[str]:
    if getattr(args, "text", None):
        return args.text
    if getattr(args, "file", None):
        with open(args.file, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    if getattr(args, "corpus_index", None) is not None:
        return load_corpus()[args.corpus_index]
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oruga", description="Optimize and measure text readability.")
    parser.add_argument("--version", action="version", version=f"oruga {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_opt = sub.add_parser("optimize", help="Optimize a text's readability.")
    p_opt.add_argument("--text", help="Text to optimize.")
    p_opt.add_argument("--file", help="Read text from a file.")
    p_opt.add_argument("--corpus-index", type=int, help="Use the Nth bundled benchmark text.")
    p_opt.add_argument("--json", action="store_true", help="Emit a JSON report.")
    _add_problem_args(p_opt)
    p_opt.set_defaults(func=_cmd_optimize)

    p_met = sub.add_parser("metrics", help="Print readability metrics for a text.")
    p_met.add_argument("--text")
    p_met.add_argument("--file")
    p_met.add_argument("--corpus-index", type=int)
    p_met.add_argument("--json", action="store_true")
    p_met.add_argument("--readability-backend", default="builtin",
                       choices=["builtin", "textstat", "readability-metrics"])
    p_met.set_defaults(func=_cmd_metrics)

    p_list = sub.add_parser("list", help="List providers, optimizers and metrics.")
    p_list.set_defaults(func=_cmd_list)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
