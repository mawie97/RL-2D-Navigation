# generator_hybrid.py
from __future__ import annotations

import random
from typing import List, Tuple, Literal

from generator_solver import SymbolicScenarioGenerator

Coord = Tuple[int, int]
Grid = List[List[bool]]
DistanceLabel = Literal["short", "mid", "long"]
HybridScenario = Literal["deadend", "corridor"]

FULL_H = 15
FULL_W = 15
BLOCK_H = 5
BLOCK_W = 5
TARGET_DEFAULT: Coord = (FULL_H // 2, FULL_W // 2)


def distance_band_for_label(label: DistanceLabel) -> tuple[int, int]:
    # Keep exactly what you currently use
    if label == "short":
        return (3, 6)
    if label == "mid":
        return (7, 10)
    if label == "long":
        return (11, 14)
    raise ValueError(label)


def candidate_offsets_for_at(
    agent_rel: Coord,
    target: Coord,
    band: tuple[int, int],
) -> List[Coord]:
    """All offsets (off_r, off_c) that place agent-target manhattan distance within band."""
    lo, hi = band
    offsets: List[Coord] = []
    for off_r in range(0, FULL_H - BLOCK_H + 1):
        for off_c in range(0, FULL_W - BLOCK_W + 1):
            ag = (off_r + agent_rel[0], off_c + agent_rel[1])
            d = abs(ag[0] - target[0]) + abs(ag[1] - target[1])
            if lo <= d <= hi:
                offsets.append((off_r, off_c))
    return offsets


def _count_walls(grid: Grid) -> int:
    return sum(1 for row in grid for v in row if v)


def generate(
    *,
    scenario: HybridScenario,
    depth: int,
    distance: DistanceLabel,
    seed: int,
    obstacles_max: int,
    target: Coord = TARGET_DEFAULT,
) -> tuple[Grid, Coord, Coord]:
    """
    Hybrid generation:
    1) Generate symbolic block (6x6) via Z3
    2) Choose agent rel-cell (deadend endpoint OR corridor midpoint)
    3) Find offset so agent-target distance is in required band
    4) Embed block into 15x15 grid, return (grid, agent, target)
    """
    rng = random.Random(seed)

    use_deadend = (scenario == "deadend")
    use_corridor = (scenario == "corridor")

    max_block_walls = max(1, obstacles_max)
    min_block_walls = 1

    block_bool, chosen_dead, deadend_path_block, corridor_path_block = SymbolicScenarioGenerator.generate_grid(
        H=BLOCK_H,
        W=BLOCK_W,
        deadend=use_deadend,
        corridor=use_corridor,
        min_deadend_depth=depth,
        min_corridorLength=depth,
        corridor_endpoint_min_free_degree=3,
        z3_seed=rng.randint(0, 1_000_000),
        exact_walls=None,
        min_walls=min_block_walls,
        max_walls=max_block_walls,
        spawn=(1, 1),
    )

    # agent position within block (free on the path)
    if use_deadend:
        if not deadend_path_block:
            raise RuntimeError("deadend_path_block empty")
        agent_rel = deadend_path_block[-1]
    else:
        if not corridor_path_block:
            raise RuntimeError("corridor_path_block empty")
        agent_rel = corridor_path_block[len(corridor_path_block) // 2]

    band = distance_band_for_label(distance)
    offsets = candidate_offsets_for_at(agent_rel=agent_rel, target=target, band=band)
    rng.shuffle(offsets)

    fixed_target = target

    for off_r, off_c in offsets:
        grid_full: Grid = [[False for _ in range(FULL_W)] for _ in range(FULL_H)]

        # embed block walls
        for r in range(BLOCK_H):
            for c in range(BLOCK_W):
                if block_bool[r][c]:
                    grid_full[off_r + r][off_c + c] = True

        # force target free
        tr, tc = fixed_target
        grid_full[tr][tc] = False

        # compute agent global and force free
        agent_global = (off_r + agent_rel[0], off_c + agent_rel[1])
        ar, ac = agent_global
        grid_full[ar][ac] = False

        if _count_walls(grid_full) <= obstacles_max:
            return grid_full, agent_global, fixed_target

    raise RuntimeError("No valid hybrid placement found (distance band too tight or constraints too strict).")
