from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"
INSUFFICIENT_DATA = "insufficient data"


@dataclass
class Evidence:
    metric: str
    observed_value: Any
    baseline: Any
    timestamp: str
    source_dataset: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "observed_value": self.observed_value,
            "baseline": self.baseline,
            "timestamp": self.timestamp,
            "source_dataset": self.source_dataset,
        }


@dataclass
class GridSignal:
    signal_name: str
    direction: str
    magnitude: float | None
    timestamp: str
    evidence: list[dict[str, Any]]
    confidence: str


@dataclass
class PriceSignal:
    price_signal: str
    price_change: float | None
    current_price: float | None
    baseline_price: float | None
    day_ahead_price: float | None
    rt_da_spread: float | None
    evidence: list[dict[str, Any]]
    confidence: str


@dataclass
class ReasoningOutput:
    explanation: str
    likely_driver: str
    confidence: str
    evidence: list[dict[str, Any]]
    caveats: list[str] = field(default_factory=list)


@dataclass
class ExposureOutput:
    persona: str
    exposure: str
    possible_actions: list[str]
    linked_signals: list[str]
    disclaimer: str


DISCLAIMER = "This is not financial or trading advice. Human review required."
