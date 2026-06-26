"""Declarative configuration for a full ORUGA run.

A single :class:`OrugaConfig` captures everything that used to be scattered as
top-level constants across the scripts: the synonym source, the metric, which
objectives are active, the encoding heuristics, the optimizer and its
hyper-parameters. Configs round-trip to YAML/JSON so experiments are
reproducible and shareable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OrugaConfig:
    # --- problem ---------------------------------------------------------
    provider: str = "dictionary"
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)
    metric: str = "fkgl"
    readability_backend: str = "builtin"
    modification: bool = False
    modification_mode: str = "active"
    semantic: bool = False
    semantic_backend: str = "jaccard"

    # --- encoding --------------------------------------------------------
    min_length: int = 4
    skip_capitalized: bool = True
    skip_suffixes: List[str] = field(default_factory=list)
    lower_bound: float = -4.0
    upper_bound: float = 4.0

    # --- optimizer -------------------------------------------------------
    optimizer: str = "ga"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None

    # --- post-processing -------------------------------------------------
    correct_grammar: bool = False

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), indent=2, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrugaConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_file(cls, path: str) -> "OrugaConfig":
        with open(path, "r", encoding="utf-8") as handle:
            if path.endswith((".yaml", ".yml")):
                import yaml  # optional dependency
                data = yaml.safe_load(handle)
            else:
                data = json.load(handle)
        return cls.from_dict(data or {})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            if path.endswith((".yaml", ".yml")):
                import yaml
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
            else:
                handle.write(self.to_json())
