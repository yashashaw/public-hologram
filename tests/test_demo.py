import heapq
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hologram_demo.placement import PlacementProblem  # noqa: E402
from hologram_demo.search import chebyshev_heuristic, flood, search  # noqa: E402
from hologram_demo.viewshed import Radar, line_of_sight  # noqa: E402

MOVES = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def reference_dijkstra(cost, source):
    """Plain-Python implementation, sharing no code with the compiled kernel."""
    rows, cols = cost.shape
    best = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > best[(r, c)]:
            continue
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and np.isfinite(cost[nr, nc]):
                nd = d + 0.5 * (cost[r, c] + cost[nr, nc]) * math.hypot(dr, dc)
                if nd < best.get((nr, nc), math.inf):
                    best[(nr, nc)] = nd
                    heapq.heappush(heap, (nd, (nr, nc)))
    out = np.full(cost.shape, np.inf)
    for cell, d in best.items():
        out[cell] = d
    return out


def random_surface(seed, size=40):
    rng = np.random.default_rng(seed)
    cost = rng.uniform(0.1, 3.0, (size, size))
    cost[rng.random((size, size)) < 0.15] = np.inf
    cost[size // 2, size // 2] = 1.0
    return cost


@pytest.mark.parametrize("seed", range(8))
def test_flood_matches_independent_dijkstra(seed):
    cost = random_surface(seed)
    source = (20, 20)
    np.testing.assert_allclose(flood(cost, [source]).cost_to_go, reference_dijkstra(cost, source))


@pytest.mark.parametrize("seed", range(20))
def test_astar_is_optimal_for_goals_anywhere(seed):
    # Goals are scattered through the interior on purpose: a heuristic that
    # assumes goals lie on the grid edge overestimates here and returns bad paths.
    rng = np.random.default_rng(1000 + seed)
    cost = random_surface(seed)
    goals = (rng.random(cost.shape) < 0.01) & np.isfinite(cost)
    goals[20, 20] = False
    if not goals.any():
        goals[5, 5] = True
        cost[5, 5] = 1.0

    finite_min = float(cost[np.isfinite(cost)].min())
    plain = search(cost, [(20, 20)], goals)
    astar = search(cost, [(20, 20)], goals, heuristic=chebyshev_heuristic(goals, 1.0, finite_min))
    assert astar.cost == pytest.approx(plain.cost)
    assert astar.expanded <= plain.expanded


def test_reverse_flood_equals_forward_route_cost():
    cost = random_surface(3)
    field = flood(cost, [(20, 20)])
    cell = tuple(np.argwhere(np.isfinite(field.cost_to_go) & (field.cost_to_go > 5))[0])
    assert reference_dijkstra(cost, cell)[20, 20] == pytest.approx(field.cost_to_go[cell])


def test_traced_path_sums_to_reported_cost():
    cost = random_surface(5)
    goals = np.zeros(cost.shape, dtype=bool)
    goals[0, :] = np.isfinite(cost[0, :])
    result = search(cost, [(20, 20)], goals)
    total = 0.0
    for (r0, c0), (r1, c1) in zip(result.path[:-1], result.path[1:]):
        total += 0.5 * (cost[r0, c0] + cost[r1, c1]) * math.hypot(r1 - r0, c1 - c0)
    assert total == pytest.approx(result.cost)


def test_flat_ground_is_fully_visible_in_range():
    z = np.full((101, 101), 100.0)
    radar = Radar(max_range_m=4_000.0, mast_height_m=20.0, target_height_m=20.0)
    visible = line_of_sight(z, (50, 50), 100.0, radar)
    rows, cols = np.indices(z.shape)
    in_range = np.hypot(rows - 50, cols - 50) <= 40
    assert visible[in_range].all()
    assert not visible[~in_range].any()


def test_wall_casts_a_shadow():
    z = np.full((101, 101), 100.0)
    z[:, 60] = 400.0
    radar = Radar(max_range_m=4_000.0, mast_height_m=20.0, target_height_m=20.0)
    visible = line_of_sight(z, (50, 50), 100.0, radar)
    assert visible[50, 55]
    assert not visible[50, 75]


def test_earth_curvature_hides_distant_low_targets():
    z = np.zeros((3, 801))
    radar = Radar(max_range_m=80_000.0, mast_height_m=10.0, target_height_m=10.0)
    visible = line_of_sight(z, (1, 0), 100.0, radar)
    assert visible[1, 100]  # 10 km, inside the combined radio horizon
    assert not visible[1, 700]  # 70 km, far beyond it


def make_problem():
    size = 60
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 1.0, (size, size))
    entries = np.zeros((size, size), dtype=bool)
    entries[:, 0] = True
    candidates = []
    for centre in [(10, 30), (20, 30), (30, 30), (40, 30), (50, 30), (30, 15)]:
        rows, cols = np.indices((size, size))
        candidates.append((np.hypot(rows - centre[0], cols - centre[1]) < 12).astype(float))
    return PlacementProblem(base, np.zeros((size, size)), candidates, (30, 55), entries, 1.0, 4.0)


def test_adding_a_sensor_never_helps_the_adversary():
    problem = make_problem()
    assert problem.value((0,)) >= problem.value(()) - 1e-9
    assert problem.value((0, 1)) >= problem.value((0,)) - 1e-9


def test_strategies_are_ordered_and_bounded_by_exhaustive():
    problem = make_problem()
    greedy = problem.greedy(3)
    refined = problem.swap_refine(greedy)
    exact = problem.exhaustive(3)
    assert greedy.value <= refined.value + 1e-9
    assert refined.value <= exact.value + 1e-9
