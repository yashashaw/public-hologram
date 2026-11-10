"""Run the whole pipeline on a synthetic scene and write figures and metrics to docs/."""
from __future__ import annotations

import argparse
import json
import math
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from hologram_demo.placement import PlacementProblem  # noqa: E402
from hologram_demo.search import chebyshev_heuristic, flood, search  # noqa: E402
from hologram_demo.terrain import base_cost, block_high_points, make_scene  # noqa: E402
from hologram_demo.viewshed import Radar, detection_field  # noqa: E402

DOCS = Path(__file__).parent / "docs"
RADAR_WEIGHT = 2.0
SENSORS_TO_PLACE = 3
FIXED_RADARS = [(0.14, 0.24), (0.86, 0.2), (0.5, 0.28), (0.18, 0.82), (0.84, 0.84), (0.48, 0.55)]  # fractions of the grid
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
ROUTE = "#eb6834"
HEAT = "Blues"


def strongest_entries(field, entries, count, separation):
    """Cheapest entry cells, keeping only ones well apart along the border."""
    cells = sorted(map(tuple, np.argwhere(entries)), key=lambda rc: field.cost_to_go[rc])
    chosen = []
    for cell in cells:
        if np.isfinite(field.cost_to_go[cell]) and all(abs(cell[0] - c[0]) >= separation for c in chosen):
            chosen.append(cell)
        if len(chosen) == count:
            break
    return [(int(r), int(c)) for r, c in chosen]


def panel(ax, title):
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_route(ax, path, **kwargs):
    rows, cols = zip(*path)
    ax.plot(cols, rows, color=ROUTE, linewidth=kwargs.pop("linewidth", 2), solid_capstyle="round", **kwargs)


def draw_target(ax, target):
    ax.plot(target[1], target[0], marker="*", markersize=15, color=ROUTE,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)


def label_entry_and_asset(ax, scene, entry_row=0.62, asset_text="Defended asset"):
    """Mark where routes begin (left edge) and what they are trying to reach."""
    size = scene.elevation.shape[0]
    box = dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2)
    ax.plot([-0.5, -0.5], [-0.5, size - 0.5], color=INK, linewidth=5, clip_on=False,
            solid_capstyle="butt", zorder=6)
    ax.text(9, size * entry_row, "Routes enter along this edge", rotation=90, ha="left", va="center",
            fontsize=8.5, color=INK, bbox=box, zorder=6)
    ax.text(scene.target[1] + 4, scene.target[0] + 17, asset_text, ha="right", va="center",
            fontsize=8.5, color=INK, bbox=box, zorder=6)


def draw_entry_points(ax, routes, label=None):
    rows, cols = zip(*[route[0] for route in routes])
    ax.plot(cols, rows, linestyle="none", marker="o", markersize=9, color=ROUTE, clip_on=False,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=7, label=label)


def draw_sites(ax, sites, marker, label=None):
    rows, cols = zip(*sites)
    ax.plot(cols, rows, linestyle="none", marker=marker, markersize=9, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label=label)


def figure_scene(scene, hillshade, detection, cost, fixed):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), facecolor=SURFACE)
    axes[0].imshow(hillshade, cmap="gray", vmin=0, vmax=1)
    draw_sites(axes[0], fixed, "^")
    panel(axes[0], "Terrain and fixed radars")
    axes[1].imshow(detection, cmap=HEAT, vmin=0, vmax=1)
    panel(axes[1], "Radar detection (dark = likely)")
    axes[2].imshow(scene.canopy, cmap=HEAT, vmin=0, vmax=1)
    panel(axes[2], "Forest concealment (dark = dense)")
    axes[3].imshow(cost, cmap=HEAT)
    draw_target(axes[3], scene.target)
    label_entry_and_asset(axes[3], scene)
    panel(axes[3], "Combined cost (dark = exposed)")
    fig.tight_layout()
    fig.savefig(DOCS / "01_scene.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def figure_flood(scene, field, routes):
    fig, ax = plt.subplots(figsize=(6.4, 6.0), facecolor=SURFACE)
    ax.imshow(field.cost_to_go, cmap=HEAT)
    ax.contour(field.cost_to_go, levels=14, colors=SURFACE, linewidths=0.6, alpha=0.55)
    for index, route in enumerate(routes):
        draw_route(ax, route, label="Cheapest route" if index == 0 else None)
    draw_entry_points(ax, routes, label="Entry point")
    draw_target(ax, scene.target)
    label_entry_and_asset(ax, scene, entry_row=0.3, asset_text="Flood starts here (defended asset)")
    ax.legend(loc="upper right", frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="none")
    panel(ax, "Reverse Dijkstra: one flood prices every entry")
    ax.set_xlabel("Dark = costly to reach the asset. White lines join cells of equal cost.",
                  fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(DOCS / "02_flood.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def figure_minimax(scene, problem, fixed, candidate_sites, chosen, before, after):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.7), facecolor=SURFACE)
    surfaces = [problem.surface(()), problem.surface(chosen.sites)]
    vmax = max(s.max() for s in surfaces)
    titles = [f"Before: cheapest route costs {before.cost:,.0f}",
              f"After {len(chosen.sites)} placed radars: {after.cost:,.0f} "
              f"(+{100 * (after.cost / before.cost - 1):.0f}%)"]
    for ax, surface, result, title in zip(axes, surfaces, [before, after], titles):
        ax.imshow(surface, cmap=HEAT, vmin=0, vmax=vmax)
        draw_route(ax, result.path, linewidth=2.5, label="Cheapest route")
        draw_entry_points(ax, [result.path], label="Entry point")
        draw_sites(ax, fixed, "^", "Fixed radar")
        draw_target(ax, scene.target)
        label_entry_and_asset(ax, scene)
        panel(ax, title)
    draw_sites(axes[1], [candidate_sites[i] for i in chosen.sites], "o", "Placed radar")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(DOCS / "03_minimax.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def animation_flood(scene, hillshade, field, route):
    finite = field.cost_to_go[np.isfinite(field.cost_to_go)]
    limit = finite.max()
    cmap = plt.get_cmap(HEAT)
    base_rgb = (np.dstack([hillshade] * 3) * 255).astype(np.uint8)
    scale = 480 / scene.elevation.shape[0]
    font = ImageFont.load_default(size=14)
    star_x, star_y = scene.target[1] * scale, scene.target[0] * scale

    def annotate(frame):
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, 5, 479], fill=INK)
        draw.text((12, 8), "Left edge: routes enter here", font=font, fill=INK,
                  stroke_width=3, stroke_fill=SURFACE)
        draw.ellipse([star_x - 7, star_y - 7, star_x + 7, star_y + 7], fill=ROUTE, outline=SURFACE, width=2)
        draw.text((star_x + 10, star_y + 26), "Flood starts here", font=font, fill=INK, anchor="rm",
                  stroke_width=3, stroke_fill=SURFACE)

    frames = []
    for fraction in np.linspace(0.03, 1.0, 34):
        reached = field.cost_to_go <= fraction * limit
        colour = (cmap(0.25 + 0.7 * np.clip(field.cost_to_go / limit, 0, 1))[..., :3] * 255).astype(np.uint8)
        rgb = base_rgb.copy()
        rgb[reached] = (0.25 * base_rgb[reached] + 0.75 * colour[reached]).astype(np.uint8)
        frame = Image.fromarray(rgb).resize((480, 480), Image.BILINEAR)
        annotate(frame)
        frames.append(frame)
    final = frames[-1].copy()
    ImageDraw.Draw(final).line([(c * scale, r * scale) for r, c in route], fill=ROUTE, width=3)
    annotate(final)
    frames.extend([final] * 12)
    palette_frames = [f.quantize(colors=96, dither=Image.Dither.NONE) for f in frames]
    palette_frames[0].save(DOCS / "flood.gif", save_all=True, append_images=palette_frames[1:],
                           duration=90, loop=0, optimize=True)


def write_results_page(metrics):
    """Local page that walks a reader through the figures and numbers from this run."""
    search = metrics["search"]
    placement = metrics["placement"]
    radar = metrics["radar"]
    baseline = metrics["baseline_route_cost"]
    optimum = placement["exhaustive"]["value"]
    side_km = metrics["grid_side_cells"] * metrics["cell_size_m"] / 1000
    subsets = math.comb(metrics["candidate_sites"], metrics["sensors_placed"])

    search_rows = "".join(
        f"<tr><td>{name}</td><td>{what}</td><td>{search[key]['expanded']:,}</td><td>{search[key]['ms']:.0f} ms</td></tr>"
        for name, what, key in [
            ("Full reverse flood", "Cheapest route from every cell", "full_flood"),
            ("Dijkstra with early exit", "Cheapest entry only", "dijkstra_early_exit"),
            ("A* with early exit", "Cheapest entry only", "astar")])
    placement_rows = "".join(
        f"<tr><td>{name}</td><td>{how}</td><td>{placement[key]['value']:,.0f}</td>"
        f"<td>+{100 * (placement[key]['value'] / baseline - 1):.0f}%</td>"
        f"<td>{100 * (1 - placement[key]['value'] / optimum):.2f}%</td></tr>"
        for name, how, key in [
            ("Greedy", "Adds the single best radar, three times", "greedy"),
            ("Greedy + swap", "Then exchanges one radar at a time while that helps", "greedy_plus_swap"),
            ("Exhaustive", f"Scores all {subsets:,} possible sets", "exhaustive")])
    saved = 100 * (1 - search["astar"]["expanded"] / search["dijkstra_early_exit"]["expanded"])

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hologram demo results</title>
<style>
body {{ margin: 0 auto; max-width: 1000px; padding: 24px 16px 64px; background: {SURFACE}; color: {INK};
       font: 16px/1.55 system-ui, sans-serif; }}
h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 48px; }} h3 {{ margin: 20px 0 2px; font-size: 16px; }}
p, li, dd {{ color: {INK_2}; }} img {{ max-width: 100%; height: auto; display: block; margin: 16px 0; }}
.table-wrap {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
td, th {{ padding: 6px 8px; border-bottom: 1px solid #e4e3df; text-align: left; vertical-align: top; }}
dl {{ display: grid; grid-template-columns: 170px 1fr; gap: 6px 16px; }} dt {{ font-weight: 600; }} dd {{ margin: 0; }}
code {{ font-size: 14px; }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr; }} dd {{ margin-bottom: 8px; }} }}
</style></head><body>
<h1>Hologram demo results</h1>
<p>A walkthrough of one run on invented terrain. The question: if a drone is trying not to be detected on its way
to a defended asset, which route does it fly, and where should the defender add radars to make even that route costly?</p>

<h2>How to read the maps</h2>
<p>Every map shows the same {side_km:.0f} km &times; {side_km:.0f} km square, divided into
{metrics['grid_cells']:,} cells of {metrics['cell_size_m']:.0f} m. Nothing on them describes a real place.</p>
<dl>
<dt>Left edge (black bar)</dt><dd>The starting points. A route may enter at any of the {metrics['entry_cells']} cells on this edge.</dd>
<dt>Orange star</dt><dd>The defended asset, which every route is trying to reach.</dd>
<dt>Orange line</dt><dd>A cheapest route. The orange dot on the left edge is where it enters.</dd>
<dt>Black triangle</dt><dd>A radar that already exists.</dd>
<dt>Black circle</dt><dd>A radar the optimizer chose to add.</dd>
<dt>Blue shading</dt><dd>Darker always means more: more detection, more forest, or more cost, as each map's title says.</dd>
</dl>

<h2>1. The inputs</h2>
<img src="01_scene.png" alt="Four maps: terrain, radar detection, forest concealment and combined cost">
<p>From left to right:</p>
<h3>Terrain and fixed radars</h3>
<p>A shaded-relief view of the generated elevation. Two ridges run north to south, each with one low pass, and the
triangles are the {metrics['fixed_radars']} existing radars.</p>
<h3>Radar detection</h3>
<p>How likely a drone flying {radar['target_height_m']:.0f} m above the ground is to be seen. A ray is traced from each
antenna to every cell within {radar['max_range_m'] / 1000:.0f} km, and the cell only counts if terrain and earth curvature
don't block the line. Detection then fades with range. The sharp white areas are <strong>terrain shadows</strong> behind the ridges.</p>
<h3>Forest concealment</h3>
<p>Generated forest cover, which makes a cell cheaper to fly through.</p>
<h3>Combined cost</h3>
<p>The single surface that the search runs on. Each cell's cost per metre is
<code>0.10 + 0.40 &times; noise exposure + 0.20 &times; (1 &minus; forest) + {metrics['radar_weight']:.1f} &times; detection</code>.
Noise exposure rises near generated towns. Radar has by far the largest weight because being detected is the worst outcome.
A route's total cost is this value summed along its length, so it is a unitless exposure score.</p>

<h2>2. The flood: reverse Dijkstra</h2>
<img src="flood.gif" alt="Animation of a cost wave spreading from the defended asset across the terrain">
<p>This animation is a <strong>reverse Dijkstra search</strong>. It is "reverse" because it starts at the destination
(the defended asset, near the right edge) and works outward, instead of starting from the entry points.</p>
<p>The search keeps a priority queue of cells and always finalizes the unfinished cell that is cheapest to reach. It
then checks that cell's 8 neighbours and records which neighbour gave each one its best cost. Grey has not been reached
yet, and blue has, with darker blue costing more. The front races through shadows and crawls through radar coverage,
because each frame shows a higher cost limit and cheap cells fit under it from much farther away.</p>
<p>The payoff is that <strong>one flood answers all {metrics['entry_cells']} entry points</strong>. Moving between two cells
costs the same in either direction, so the cost outward from the asset is exactly what a drone flying inward would pay.
The orange line in the last frames is the cheapest route of all.</p>

<h2>3. The finished flood and its routes</h2>
<img src="02_flood.png" alt="Map of cost to reach the asset with cheapest routes from several entry points">
<p>Each cell now shows the cost of the cheapest route from there to the asset, so it is lightest around the star. The
white contour lines join cells of equal cost, like height lines on a hiking map.</p>
<p>To draw a route, the code starts at an entry cell and follows the recorded neighbours back to the star. The orange
lines are the cheapest routes from {metrics['routes_drawn']} well-separated entries. They merge as they near the star
because once two routes reach the same cell, the cheapest way onward is identical. Notice that they pass through the
ridge shadows from the first section instead of flying straight.</p>

<h2>4. Comparing the search methods</h2>
<div class="table-wrap"><table><tr><th>Method</th><th>What it answers</th><th>Cells expanded</th><th>Time</th></tr>{search_rows}</table></div>
<p>"Cells expanded" is how many cells each search had to finalize. Early exit stops at the first entry cell finalized,
which is guaranteed to be the cheapest. A* also steers toward the entry edge using a lower bound on the remaining cost.</p>
<p>On this map A* expanded {saved:.1f}% fewer cells than plain Dijkstra. The bound has to assume the rest of the route is as
cheap as the cheapest cell anywhere, and radar makes real costs much higher than that. The full flood therefore costs about
the same and answers far more, which is why it is the workhorse.</p>

<h2>5. The minimax optimizer: where to add radars</h2>
<img src="03_minimax.png" alt="Two maps showing the cheapest route before and after three radars are added">
<p>The optimizer is a <strong>minimax</strong>. The defender chooses radar sites to maximize the cost of the adversary's
cheapest route, and the adversary responds with the route that minimizes that cost:</p>
<p><code>maximize over radar sets S &nbsp; minimize over routes p &nbsp; cost(p | S)</code></p>
<p>The left map is the starting situation, where the cheapest route slips through a ridge shadow and costs {baseline:,.0f}.
The right map shows the best {metrics['sensors_placed']} radars out of {metrics['candidate_sites']} candidate sites, which are the
highest points in a coarse grid. The shadows have mostly closed, and the adversary's best remaining route is
forced to cross covered ground.</p>
<div class="table-wrap"><table><tr><th>Strategy</th><th>Method</th><th>Cheapest-route cost after placement</th><th>Increase</th><th>Gap to optimum</th></tr>{placement_rows}</table></div>
<p>Every candidate set is scored by solving the adversary's route exactly. Greedy and swap refinement search the defender's
choices approximately, while exhaustive search checks every set and provides the true optimum for comparison. Exhaustive
search only works because this example is tiny, as the number of possible sets explodes as sites are added.</p>

<h2>Limits of this demo</h2>
<ul>
<li>The terrain, weights and radar settings ({radar['max_range_m'] / 1000:.0f} km range, {radar['mast_height_m']:.0f} m mast) are invented to make the algorithms visible. They say nothing about any real location or sensor.</li>
<li>The map stops at its edge, so routes can run along the boundary in a way a real one couldn't.</li>
<li>This is one scene from one random seed, so the numbers illustrate the method and are not a benchmark.</li>
</ul>
</body></html>
"""
    path = DOCS / "results.html"
    path.write_text(page, encoding="utf-8")
    return path


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000.0


def main(open_results=True):
    DOCS.mkdir(exist_ok=True)
    scene = make_scene()
    radar = Radar()
    size = scene.elevation.shape[0]
    base = base_cost(scene)

    candidate_sites = block_high_points(scene.elevation, blocks=4)
    fixed = [(int(size * r), int(size * c)) for r, c in FIXED_RADARS]

    existing = np.zeros_like(base)
    for site in fixed:
        np.maximum(existing, detection_field(scene.elevation, site, scene.px_m, radar), out=existing)
    candidates, viewshed_ms = timed(
        lambda: [detection_field(scene.elevation, s, scene.px_m, radar) for s in candidate_sites])

    problem = PlacementProblem(base, existing, candidates, scene.target, scene.entries, scene.px_m, RADAR_WEIGHT)
    cost = problem.surface(())

    heuristic = chebyshev_heuristic(scene.entries, scene.px_m, float(base.min()))
    flood(cost, [scene.target], scene.px_m)  # warm the JIT so the timings below are fair
    field, flood_ms = timed(lambda: flood(cost, [scene.target], scene.px_m))
    dijkstra, dijkstra_ms = timed(lambda: search(cost, [scene.target], scene.entries, scene.px_m))
    astar, astar_ms = timed(lambda: search(cost, [scene.target], scene.entries, scene.px_m, heuristic))
    assert abs(field.cost_to_go[scene.entries].min() - astar.cost) < 1e-6

    entries = strongest_entries(field, scene.entries, count=5, separation=size // 8)
    routes = [field.path_from(e) for e in entries]

    greedy, greedy_ms = timed(lambda: problem.greedy(SENSORS_TO_PLACE))
    greedy_searches = problem.searches
    refined, refine_ms = timed(lambda: problem.swap_refine(greedy))
    refine_searches = problem.searches - greedy_searches
    exact, exact_ms = timed(lambda: problem.exhaustive(SENSORS_TO_PLACE))
    exact_searches = problem.searches

    hillshade = LightSource(azdeg=315, altdeg=40).hillshade(
        scene.elevation, vert_exag=3, dx=scene.px_m, dy=scene.px_m)
    after = problem.solve(exact.sites)

    figure_scene(scene, hillshade, existing, cost, fixed)
    figure_flood(scene, field, routes)
    figure_minimax(scene, problem, fixed, candidate_sites, exact, astar, after)
    animation_flood(scene, hillshade, field, astar.path)

    metrics = {
        "grid_cells": size * size,
        "grid_side_cells": size,
        "cell_size_m": scene.px_m,
        "entry_cells": int(scene.entries.sum()),
        "routes_drawn": len(routes),
        "fixed_radars": len(fixed),
        "radar_weight": RADAR_WEIGHT,
        "radar": asdict(radar),
        "candidate_sites": len(candidates),
        "sensors_placed": SENSORS_TO_PLACE,
        "viewshed_ms_each": round(viewshed_ms / len(candidates), 1),
        "search": {
            "full_flood": {"expanded": field.expanded, "ms": round(flood_ms, 1)},
            "dijkstra_early_exit": {"expanded": dijkstra.expanded, "ms": round(dijkstra_ms, 1)},
            "astar": {"expanded": astar.expanded, "ms": round(astar_ms, 1)},
        },
        "baseline_route_cost": round(astar.cost, 1),
        "placement": {
            "greedy": {"value": round(greedy.value, 1), "sites": greedy.sites,
                       "searches": greedy_searches, "ms": round(greedy_ms)},
            "greedy_plus_swap": {"value": round(refined.value, 1), "sites": refined.sites,
                                 "new_searches": refine_searches, "ms": round(refine_ms)},
            "exhaustive": {"value": round(exact.value, 1), "sites": exact.sites,
                           "total_distinct_searches": exact_searches, "ms": round(exact_ms)},
        },
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    results = write_results_page(metrics).resolve()
    print(f"\nResults page: {results}")
    if open_results:
        webbrowser.open(results.as_uri())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="don't open the results page in a browser")
    main(open_results=not parser.parse_args().no_open)
