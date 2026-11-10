"""Graded radar detection over terrain.

A ray is traced from the antenna to every cell in range. A low-flying target is
visible only if no intervening terrain rises above the line of sight, allowing
for earth curvature with the standard 4/3-earth refraction model. Visible cells
then get a detection value that fades with range.

Per-cell ray tracing is O(R^3), which is fine for a small demo grid. Large
rasters need a sweep-based viewshed instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

EFFECTIVE_EARTH_RADIUS_M = 6_371_000.0 * 4.0 / 3.0


@dataclass(frozen=True)
class Radar:
    """Illustrative sensor parameters, not those of any real system."""

    max_range_m: float = 12_000.0
    mast_height_m: float = 15.0
    target_height_m: float = 15.0


@njit(cache=True)
def _bilinear(z: np.ndarray, row: float, col: float) -> float:
    r0 = int(row)
    c0 = int(col)
    r1 = min(r0 + 1, z.shape[0] - 1)
    c1 = min(c0 + 1, z.shape[1] - 1)
    dr = row - r0
    dc = col - c0
    return (
        z[r0, c0] * (1 - dr) * (1 - dc)
        + z[r1, c0] * dr * (1 - dc)
        + z[r0, c1] * (1 - dr) * dc
        + z[r1, c1] * dr * dc
    )


@njit(cache=True)
def _line_of_sight(z, row0, col0, px_m, radius, mast, target_h, earth_r):
    rows, cols = z.shape
    visible = np.zeros((rows, cols), dtype=np.bool_)
    visible[row0, col0] = True
    antenna = z[row0, col0] + mast

    for row in range(max(0, row0 - radius), min(rows, row0 + radius + 1)):
        for col in range(max(0, col0 - radius), min(cols, col0 + radius + 1)):
            cells = np.hypot(row - row0, col - col0)
            if cells == 0.0 or cells > radius:
                continue
            distance = cells * px_m
            target_z = z[row, col] + target_h - distance * distance / (2.0 * earth_r)
            sight_slope = (target_z - antenna) / distance

            blocked = False
            steps = int(np.ceil(cells))
            for i in range(1, steps):
                t = i / steps
                along = t * distance
                ground = _bilinear(z, row0 + t * (row - row0), col0 + t * (col - col0))
                ground -= along * along / (2.0 * earth_r)
                if (ground - antenna) / along > sight_slope:
                    blocked = True
                    break
            visible[row, col] = not blocked
    return visible


def line_of_sight(elevation: np.ndarray, site: tuple[int, int], px_m: float, radar: Radar) -> np.ndarray:
    """Boolean mask of cells where a target at altitude is visible from `site`."""
    radius = int(radar.max_range_m / px_m)
    return _line_of_sight(
        elevation.astype(np.float64), site[0], site[1], px_m, radius,
        radar.mast_height_m, radar.target_height_m, EFFECTIVE_EARTH_RADIUS_M,
    )


def detection_field(elevation: np.ndarray, site: tuple[int, int], px_m: float, radar: Radar) -> np.ndarray:
    """Detection likelihood 0..1: line of sight, fading quadratically with range."""
    rows, cols = np.indices(elevation.shape)
    range_m = np.hypot(rows - site[0], cols - site[1]) * px_m
    fade = np.clip(1.0 - (range_m / radar.max_range_m) ** 2, 0.0, 1.0)
    return fade * line_of_sight(elevation, site, px_m, radar)
