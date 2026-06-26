"""High-level convenience API.

:func:`optimize` is the one-call entry point most users want: give it a string
(and optionally a :class:`~oruga.config.OrugaConfig`) and get back the simplified
text together with a full before/after readability report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import readability as rd
from .config import OrugaConfig
from .grammar import correct_grammar
from .optimizers import OptimizationResult, get_optimizer
from .problem import OrugaProblem

__all__ = ["OptimizationReport", "optimize", "optimize_corpus"]


@dataclass
class OptimizationReport:
    original_text: str
    optimized_text: str
    result: OptimizationResult
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    config: Optional[OrugaConfig] = None

    def improvement(self, metric: str = "fkgl") -> float:
        """Signed change ``before - after`` (positive = easier for grade metrics)."""
        return self.metrics_before.get(metric, 0.0) - self.metrics_after.get(metric, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "optimized_text": self.optimized_text,
            "algorithm": self.result.algorithm,
            "objective_labels": self.result.problem.labels(),
            "objective_values": self.result.best_values,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
            "front_size": len(self.result.front),
            "runtime_seconds": self.result.runtime,
            "n_evaluations": self.result.n_evaluations,
        }


def _config_from(config: Optional[OrugaConfig], overrides: Dict[str, Any]) -> OrugaConfig:
    base = config or OrugaConfig()
    if overrides:
        merged = base.to_dict()
        merged.update({k: v for k, v in overrides.items() if v is not None})
        base = OrugaConfig.from_dict(merged)
    return base


def optimize(text: str, config: Optional[OrugaConfig] = None, **overrides) -> OptimizationReport:
    """Optimize a single ``text`` and return an :class:`OptimizationReport`."""
    cfg = _config_from(config, overrides)

    problem = OrugaProblem.from_text(
        text,
        provider=cfg.provider,
        provider_kwargs=cfg.provider_kwargs,
        metric=cfg.metric,
        readability_backend=cfg.readability_backend,
        modification=cfg.modification,
        modification_mode=cfg.modification_mode,
        semantic=cfg.semantic,
        semantic_backend=cfg.semantic_backend,
        min_length=cfg.min_length,
        skip_capitalized=cfg.skip_capitalized,
        skip_suffixes=tuple(cfg.skip_suffixes),
        lower_bound=cfg.lower_bound,
        upper_bound=cfg.upper_bound,
    )

    optimizer = get_optimizer(cfg.optimizer, **cfg.optimizer_kwargs)
    result = optimizer.optimize(problem, seed=cfg.seed)

    optimized = result.best_text
    if cfg.correct_grammar:
        optimized = correct_grammar(optimized)

    return OptimizationReport(
        original_text=text,
        optimized_text=optimized,
        result=result,
        metrics_before=rd.all_scores(text, backend=cfg.readability_backend),
        metrics_after=rd.all_scores(optimized, backend=cfg.readability_backend),
        config=cfg,
    )


def optimize_corpus(texts: List[str], config: Optional[OrugaConfig] = None, **overrides) -> List[OptimizationReport]:
    """Optimize every text in ``texts`` with the same configuration."""
    return [optimize(t, config=config, **overrides) for t in texts]
