"""Synthetic scene generator.

Everything here comes from a random seed. The demo never reads a real elevation
model, map layer or sensor location, so nothing in it describes a real place.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Scene:
    elevation: np.ndarray  # metres
    canopy: np.ndarray  # 0..1, forest concealment
    noise_exposure: np.ndarray  # 0..1, proximity to people who could hear a drone
    entries: np.ndarray  # bool mask of cells where a route may start
    target: tuple[int, int]  # (row, col) of the defended asset
    px_m: float  # cell size in metres


def _fractal_noise(rng: np.random.Generator, size: int, beta: float) -> np.ndarray:
    """Spectral synthesis: random phases with a 1/f^beta power spectrum, in 0..1."""
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    freq = np.hypot(fy, fx)
    freq[0, 0] = 1.0
    amplitude = freq ** (-beta / 2.0)
    amplitude[0, 0] = 0.0
    phases = np.exp(2j * np.pi * rng.random((size, size)))
    field = np.fft.ifft2(amplitude * phases).real
    return (field - field.min()) / (field.max() - field.min())


def _ridge(size: int, column: float, gap_row: float, rng: np.random.Generator) -> np.ndarray:
    """A north-south ridge with one low pass through it, 0..1."""
    rows = np.arange(size) / size
    meander = 0.04 * np.sin(2 * np.pi * (rows * 1.3 + rng.random()))
    distance = (np.arange(size)[None, :] / size - (column + meander)[:, None]) * size
    profile = np.exp(-0.5 * (distance / 13.0) ** 2)
    gap = 1.0 - 0.85 * np.exp(-0.5 * ((rows - gap_row) / 0.07) ** 2)
    return profile * gap[:, None]


def make_scene(size: int = 320, px_m: float = 100.0, seed: int = 7) -> Scene:
    """Build a square scene. Routes enter along the west edge, the asset is east."""
    rng = np.random.default_rng(seed)

    elevation = 100.0 + 45.0 * _fractal_noise(rng, size, beta=3.2)
    elevation += 130.0 * _ridge(size, column=0.42, gap_row=0.28, rng=rng)
    elevation += 120.0 * _ridge(size, column=0.68, gap_row=0.74, rng=rng)

    canopy_field = ndimage.gaussian_filter(_fractal_noise(rng, size, beta=2.2), 2.0)
    canopy_field = (canopy_field - canopy_field.min()) / np.ptp(canopy_field)
    canopy = np.clip((canopy_field - 0.48) * 7.0, 0.0, 1.0)

    towns = np.zeros((size, size))
    for _ in range(6):
        row, col = rng.integers(10, size - 10, size=2)
        towns[row, col] = rng.uniform(0.4, 1.0)
    noise_exposure = ndimage.gaussian_filter(towns, 16.0)
    noise_exposure /= noise_exposure.max()

    entries = np.zeros((size, size), dtype=bool)
    entries[:, 0] = True
    target = (size // 2, int(size * 0.92))

    return Scene(elevation, canopy, noise_exposure, entries, target, px_m)


def base_cost(scene: Scene) -> np.ndarray:
    """Terrain-only cost per metre, before any radar is considered.

    The weights are illustrative round numbers for this synthetic scene.
    """
    return 0.10 + 0.40 * scene.noise_exposure + 0.20 * (1.0 - scene.canopy)


def block_high_points(elevation: np.ndarray, blocks: int, margin: int = 8) -> list[tuple[int, int]]:
    """Candidate sensor sites: the highest cell in each block of a coarse grid."""
    size = elevation.shape[0]
    edges = np.linspace(margin, size - margin, blocks + 1).astype(int)
    sites = []
    for r0, r1 in zip(edges[:-1], edges[1:]):
        for c0, c1 in zip(edges[:-1], edges[1:]):
            window = elevation[r0:r1, c0:c1]
            row, col = np.unravel_index(int(np.argmax(window)), window.shape)
            sites.append((r0 + int(row), c0 + int(col)))
    return sites
