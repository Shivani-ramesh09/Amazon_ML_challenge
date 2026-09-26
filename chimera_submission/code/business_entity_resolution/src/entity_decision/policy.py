"""Deterministic singleton-aware decisions on sorted pair scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DecisionPolicy:
    singleton_threshold: float = 0.50
    strong_match_threshold: float = 0.98
    pair_threshold: float = 0.50
    gap_threshold: float = 0.25
    max_match_count: int = 30

    def __post_init__(self):
        if not all(0 <= value <= 1 for value in (self.singleton_threshold,
                  self.strong_match_threshold, self.pair_threshold, self.gap_threshold)):
            raise ValueError("thresholds must be in [0, 1]")
        if self.max_match_count < 1:
            raise ValueError("max_match_count must be positive")

    def decide(self, candidate_ids: list[str], scores: list[float]) -> list[str]:
        if len(candidate_ids) != len(scores):
            raise ValueError("candidate and score lists differ in length")
        if not scores or scores[0] < self.singleton_threshold:
            return []
        second = scores[1] if len(scores) > 1 else 0.0
        if scores[0] >= self.strong_match_threshold and scores[0] - second >= self.gap_threshold:
            return [candidate_ids[0]]
        return [candidate for candidate, score in zip(candidate_ids, scores)
                if score >= self.pair_threshold][:self.max_match_count]

    def as_dict(self) -> dict:
        return asdict(self)
