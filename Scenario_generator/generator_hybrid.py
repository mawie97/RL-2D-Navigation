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
BLOCK_H = 6
BLOCK_W = 6
TARGET_DEFAULT: Coord = (FULL_H // 2, FULL_W // 2)


def distance_band_for_label(label: DistanceLabel) -> tuple[int, int]:
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
    Hybrid generation with retries:
    - Generate symbolic block
    - Try embed offsets satisfying distance band
    - If embed fails (no offsets / agent lands on wall everywhere), regenerate block and retry
    """
    rng = random.Random(seed)

    use_deadend = (scenario == "deadend")
    use_corridor = (scenario == "corridor")
    if not (use_deadend ^ use_corridor):
        raise ValueError(f"scenario must be 'deadend' or 'corridor', got: {scenario}")

    band_min, band_max = distance_band_for_label(distance)

    min_block_walls = 1
    max_block_walls = max(1, obstacles_max)

    MAX_BLOCK_RETRIES = 100

    for attempt in range(MAX_BLOCK_RETRIES):
        # New symbolic instance each attempt
        z3_seed = rng.randint(0, 1_000_000)

        block_bool, chosen_dead, deadend_path_block, corridor_path_block = SymbolicScenarioGenerator.generate_grid(
            H=BLOCK_H,
            W=BLOCK_W,
            deadend=use_deadend,
            corridor=use_corridor,
            min_deadend_depth=depth,
            min_corridorLength=depth,
            corridor_endpoint_min_free_degree=3,
            z3_seed=z3_seed,
            exact_walls=None,
            min_walls=min_block_walls,
            max_walls=max_block_walls,
            spawn=(1, 1),
        )

        # Agent position within block
        if use_deadend:
            if not deadend_path_block:
                continue
            agent_rel_block = deadend_path_block[-1]  # deadend endpoint
        else:
            if not corridor_path_block:
                continue
            agent_rel_block = corridor_path_block[len(corridor_path_block) // 2]  # corridor midpoint

        # Find offsets satisfying distance band
        candidate_offsets = candidate_offsets_for_at(
            agent_rel=agent_rel_block,
            target=target,
            band=(band_min, band_max),
        )


        if not candidate_offsets:
            continue

        rng.shuffle(candidate_offsets)

        tr, tc = target

        for off_r, off_c in candidate_offsets:
            # Start with free grid
            grid_full: Grid = [[False for _ in range(FULL_W)] for _ in range(FULL_H)]

            # Embed symbolic block walls
            for r in range(BLOCK_H):
                for c in range(BLOCK_W):
                    if block_bool[r][c]:
                        grid_full[off_r + r][off_c + c] = True

            # Force target free
            grid_full[tr][tc] = False

            # Agent global coords
            agent_global = (off_r + agent_rel_block[0], off_c + agent_rel_block[1])

            # Agent must be free
            if grid_full[agent_global[0]][agent_global[1]]:
                continue

            # Wall count sanity (should already hold, but keep it)
            if _count_walls(grid_full) > obstacles_max:
                continue

            return grid_full, agent_global, target

        # Had offsets, but none worked -> try a new symbolic block
        continue

    raise RuntimeError(
        f"Hybrid generation failed after {MAX_BLOCK_RETRIES} symbolic retries "
        f"(scenario={scenario}, depth={depth}, distance={distance}, seed={seed})."
    )
