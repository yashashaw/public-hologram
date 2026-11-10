"""Sensor placement as a two-level minimax problem.

    maximise over sensor sets S:   minimise over routes p:   cost(p | S)

The defender picks sites and the adversary answers with the cheapest route, so
each candidate set costs one full least-cost search to score.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .search import SearchResult, chebyshev_heuristic, search


@dataclass(frozen=True)
class Placement:
    sites: tuple[int, ...]
    value: float


class PlacementProblem:
    def __init__(self, base: np.ndarray, existing: np.ndarray, candidates: list[np.ndarray],
                 target: tuple[int, int], entries: np.ndarray, px_m: float, radar_weight: float):
        self.base = base
        self.existing = existing
        self.candidates = candidates
        self.target = target
        self.entries = entries
        self.px_m = px_m
        self.radar_weight = radar_weight
        # Radars only ever add cost, so the bare surface's minimum bounds every
        # surface this problem will produce and the heuristic is built only once.
        self.heuristic = chebyshev_heuristic(entries, px_m, float(base.min()))
        self._memo: dict[tuple[int, ...], float] = {}
        self.searches = 0

    def surface(self, sites: tuple[int, ...]) -> np.ndarray:
        detection = self.existing.copy()
        for site in sites:
            np.maximum(detection, self.candidates[site], out=detection)
        return self.base + self.radar_weight * detection

    def solve(self, sites: tuple[int, ...]) -> SearchResult:
        return search(self.surface(sites), [self.target], self.entries, self.px_m, self.heuristic)

    def value(self, sites: tuple[int, ...]) -> float:
        key = tuple(sorted(sites))
        if key not in self._memo:
            self.searches += 1
            self._memo[key] = self.solve(key).cost
        return self._memo[key]

    def greedy(self, k: int) -> Placement:
        """Add the single most valuable site, k times."""
        chosen: tuple[int, ...] = ()
        for _ in range(k):
            remaining = [c for c in range(len(self.candidates)) if c not in chosen]
            best = max(remaining, key=lambda c: self.value(chosen + (c,)))
            chosen += (best,)
        return Placement(tuple(sorted(chosen)), self.value(chosen))

    def swap_refine(self, start: Placement) -> Placement:
        """Hill-climb by exchanging one chosen site at a time until none helps."""
        sites, value = start.sites, start.value
        improved = True
        while improved:
            improved = False
            for out in sites:
                kept = tuple(s for s in sites if s != out)
                for new in range(len(self.candidates)):
                    if new in sites:
                        continue
                    trial = tuple(sorted(kept + (new,)))
                    if self.value(trial) > value + 1e-9:
                        sites, value = trial, self.value(trial)
                        improved = True
                        break
                if improved:
                    break
        return Placement(sites, value)

    def exhaustive(self, k: int) -> Placement:
        """Score every k-subset. Only feasible for small instances."""
        best = max(combinations(range(len(self.candidates)), k), key=self.value)
        return Placement(best, self.value(best))
