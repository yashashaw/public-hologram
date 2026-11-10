"""Least-cost search on an 8-connected raster.

One compiled kernel covers three uses:

  flood      reverse Dijkstra from the target, giving every cell's cost to reach it
  Dijkstra   the same search, stopping at the first entry cell settled
  A*         the same again, ordered by an admissible heuristic

Edge cost is the mean of the two cell costs times the step length. That makes
edges symmetric, so a flood outward from the target gives exactly the cost a
route flying inward would pay, and one solve answers every entry point at once.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy import ndimage

_DR = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)
_DC = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)
_STEP = np.array([1.0, 1.0, 1.0, 1.0, 2**0.5, 2**0.5, 2**0.5, 2**0.5])


@njit(cache=True)
def _search(cost, px_m, sources, goals, heuristic, stop_at_goal):
    rows, cols = cost.shape
    flat_cost = cost.ravel()
    g = np.full(rows * cols, np.inf)
    parent = np.full(rows * cols, -1, dtype=np.int64)
    settled = np.zeros(rows * cols, dtype=np.bool_)

    heap = [(heuristic[int(s)], int(s)) for s in sources]
    heapq.heapify(heap)
    for s in sources:
        g[s] = 0.0

    expanded = 0
    while len(heap) > 0:
        _, node = heapq.heappop(heap)
        if settled[node]:
            continue
        settled[node] = True
        expanded += 1
        if stop_at_goal and goals[node]:
            return g, parent, expanded, node

        row, col = node // cols, node % cols
        for k in range(8):
            r, c = row + _DR[k], col + _DC[k]
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            neighbour = r * cols + c
            if settled[neighbour] or not np.isfinite(flat_cost[neighbour]):
                continue
            edge = 0.5 * (flat_cost[node] + flat_cost[neighbour]) * _STEP[k] * px_m
            candidate = g[node] + edge
            if candidate < g[neighbour]:
                g[neighbour] = candidate
                parent[neighbour] = node
                heapq.heappush(heap, (candidate + heuristic[neighbour], neighbour))
    return g, parent, expanded, -1


def _flat(sources: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    return np.array([r * shape[1] + c for r, c in sources], dtype=np.int64)


@dataclass(frozen=True)
class Flood:
    cost_to_go: np.ndarray
    parent: np.ndarray
    expanded: int

    def path_from(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Cells from `cell` to the nearest flood source."""
        cols = self.cost_to_go.shape[1]
        node = cell[0] * cols + cell[1]
        path = []
        while node != -1:
            path.append((node // cols, node % cols))
            node = int(self.parent[node])
        return path


@dataclass(frozen=True)
class SearchResult:
    cost: float
    path: list[tuple[int, int]]  # goal first, source last
    expanded: int


def flood(cost: np.ndarray, sources: list[tuple[int, int]], px_m: float = 1.0) -> Flood:
    """Cost to reach the nearest source from every cell."""
    shape = cost.shape
    empty_goals = np.zeros(shape[0] * shape[1], dtype=np.bool_)
    zero = np.zeros(shape[0] * shape[1])
    g, parent, expanded, _ = _search(
        cost.astype(np.float64), px_m, _flat(sources, shape), empty_goals, zero, False)
    return Flood(g.reshape(shape), parent, expanded)


def chebyshev_heuristic(goals: np.ndarray, px_m: float, min_cost: float) -> np.ndarray:
    """Admissible lower bound on the remaining cost to any goal cell.

    Reaching a goal takes at least its chessboard distance in steps, each at
    least one cell long and at least `min_cost` per metre. Unlike a bound based
    on distance to the grid edge, this stays valid for goals anywhere in the
    grid. `min_cost` must not exceed the cheapest passable cell.
    """
    steps = ndimage.distance_transform_cdt(~goals, metric="chessboard")
    return (steps * px_m * min_cost).astype(np.float64).ravel()


def search(cost: np.ndarray, sources: list[tuple[int, int]], goals: np.ndarray,
           px_m: float = 1.0, heuristic: np.ndarray | None = None) -> SearchResult:
    """Cheapest route between any source and any goal cell.

    With no heuristic this is Dijkstra with early exit.
    """
    shape = cost.shape
    if heuristic is None:
        heuristic = np.zeros(shape[0] * shape[1])
    g, parent, expanded, goal = _search(
        cost.astype(np.float64), px_m, _flat(sources, shape), goals.ravel(), heuristic, True)
    if goal < 0:
        return SearchResult(float("inf"), [], expanded)
    field = Flood(g.reshape(shape), parent, expanded)
    return SearchResult(float(g[goal]), field.path_from((goal // shape[1], goal % shape[1])), expanded)
