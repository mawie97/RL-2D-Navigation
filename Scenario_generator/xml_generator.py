# xml_generator.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

from generator_bresenham import BresenhamStandardGenerator
import generator_hybrid
from xml_writer import write_xml_from_base

Coord = Tuple[int, int]
DistanceLabel = Literal["short", "mid", "long"]

# ============================
# Config
# ============================

FULL_H = 15
FULL_W = 15
TARGET_DEFAULT: Coord = (FULL_H // 2, FULL_W // 2)

DIST_ORDER = {"short": "1short", "mid": "2mid", "long": "3long"}

EXPERIMENT_ROOT = os.path.join("scenarios", "test")
OUT_LVL_1_4 = os.path.join(EXPERIMENT_ROOT, "lvl_1_4")
OUT_LVL_1_5 = os.path.join(EXPERIMENT_ROOT, "lvl_1_5")
OUT_LVL_5 = os.path.join(EXPERIMENT_ROOT, "lvl_5")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def distance_band_for_label(label: DistanceLabel) -> tuple[int, int]:
    # Keep exactly as your current runner uses
    if label == "short":
        return (3, 6)
    if label == "mid":
        return (7, 10)
    if label == "long":
        return (11, 14)
    raise ValueError(label)


@dataclass(frozen=True)
class ScenarioMeta:
    level: int
    scenario: str
    obstacles: int
    distance: DistanceLabel
    seed: int
    depth: Optional[int]
    path: str


# ============================
# Level 1–4: Standard (Bresenham)
# ============================

def generate_standard_scenario(
    *,
    level: int,
    obstacles: int,
    distance: DistanceLabel,
    seed: int,
    out_root: str,
    bresenham_radius: int = 1,
) -> ScenarioMeta:
    rng = random.Random(seed)
    target = TARGET_DEFAULT

    band_min, band_max = distance_band_for_label(distance)

    # pick an agent cell satisfying the band
    candidates: List[Coord] = []
    for r in range(FULL_H):
        for c in range(FULL_W):
            if (r, c) == target:
                continue
            d = abs(r - target[0]) + abs(c - target[1])
            if band_min <= d <= band_max:
                candidates.append((r, c))
    if not candidates:
        raise RuntimeError(f"No agent candidates for distance={distance}")

    agent = rng.choice(candidates)

    gen = BresenhamStandardGenerator(FULL_H, FULL_W)
    grid = gen.generate(
        agent=agent,
        target=target,
        exact_walls=obstacles,
        neighbor_radius=bresenham_radius,
        min_walls=None,
        max_walls=None,
        rng=rng,
    )

    ensure_dir(out_root)
    dist_tag = DIST_ORDER[distance]
    filename = f"lvl{level}_standard_obs{obstacles}_d{dist_tag}_seed{seed}.xml"
    out_path = os.path.join(out_root, filename)

    # Pass rng so jitter is deterministic per seed (optional but recommended)
    write_xml_from_base(grid, agent, target, out_path, rng=rng)

    return ScenarioMeta(level, "standard", obstacles, distance, seed, None, out_path)


# ============================
# Level 5: Hybrid (deadend/corridor)
# ============================

def generate_hybrid_scenario(
    *,
    scenario: Literal["deadend", "corridor"],
    depth: int,
    distance: DistanceLabel,
    seed: int,
    obstacles: int,
    out_root: str,
) -> ScenarioMeta:
    rng = random.Random(seed)

    grid, agent, target = generator_hybrid.generate(
        scenario=scenario,
        depth=depth,
        distance=distance,
        seed=seed,
        obstacles_max=obstacles,
        target=TARGET_DEFAULT,
    )

    ensure_dir(out_root)
    filename = f"lvl5_{scenario}_obs{obstacles}_d{distance}_depth{depth}_seed{seed}.xml"
    out_path = os.path.join(out_root, filename)

    write_xml_from_base(grid, agent, target, out_path, rng=rng)

    return ScenarioMeta(5, scenario, obstacles, distance, seed, depth, out_path)


# ============================
# Experiment drivers (same outputs as before)
# ============================

def generate_experiment_lvl_1_4(seed_start: int) -> int:
    ensure_dir(OUT_LVL_1_4)
    seed = seed_start

    all_dist = ("short", "mid", "long")
    mid_long = ("mid", "long")

    # L1: 10 files, obstacles=0, 3 short 3 mid 4 long
    for dist in (["short"] * 3 + ["mid"] * 3 + ["long"] * 4):
        generate_standard_scenario(level=1, obstacles=0, distance=dist, seed=seed, out_root=OUT_LVL_1_4)
        seed += 1

    # L2: 10 files, obstacles=1
    for dist in (["short"] * 3 + ["mid"] * 3 + ["long"] * 4):
        generate_standard_scenario(level=2, obstacles=1, distance=dist, seed=seed, out_root=OUT_LVL_1_4)
        seed += 1

    # L3: 10 files, obstacles in [2..5], distance random short/mid/long
    rng = random.Random(seed_start + 12345)
    for _ in range(10):
        obs = rng.randint(2, 5)
        dist = rng.choice(all_dist)
        generate_standard_scenario(level=3, obstacles=obs, distance=dist, seed=seed, out_root=OUT_LVL_1_4)
        seed += 1

    # L4: 10 files, obstacles in [6..9], distance random mid/long, bresenham_radius=2
    for _ in range(10):
        obs = rng.randint(6, 9)
        dist = rng.choice(mid_long)
        generate_standard_scenario(level=4, obstacles=obs, distance=dist, seed=seed, out_root=OUT_LVL_1_4, bresenham_radius=2)
        seed += 1

    return seed


def generate_experiment_lvl_1_5(seed_start: int) -> int:
    ensure_dir(OUT_LVL_1_5)
    seed = seed_start

    # Reuse the same lvl1-4 recipe but output into lvl_1_5 folder
    # (keeps your current structure)
    all_dist = ("short", "mid", "long")
    mid_long = ("mid", "long")

    for dist in (["short"] * 3 + ["mid"] * 3 + ["long"] * 4):
        generate_standard_scenario(level=1, obstacles=0, distance=dist, seed=seed, out_root=OUT_LVL_1_5)
        seed += 1

    for dist in (["short"] * 3 + ["mid"] * 3 + ["long"] * 4):
        generate_standard_scenario(level=2, obstacles=1, distance=dist, seed=seed, out_root=OUT_LVL_1_5)
        seed += 1

    rng = random.Random(seed_start + 54321)
    for _ in range(10):
        obs = rng.randint(2, 5)
        dist = rng.choice(all_dist)
        generate_standard_scenario(level=3, obstacles=obs, distance=dist, seed=seed, out_root=OUT_LVL_1_5)
        seed += 1

    for _ in range(10):
        obs = rng.randint(6, 9)
        dist = rng.choice(mid_long)
        generate_standard_scenario(level=4, obstacles=obs, distance=dist, seed=seed, out_root=OUT_LVL_1_5, bresenham_radius=2)
        seed += 1

    # Add 10 lvl5: 5 deadend + 5 corridor
    kinds = ["deadend"] * 5 + ["corridor"] * 5
    rng.shuffle(kinds)

    for kind in kinds:
        depth = rng.choice([1, 2, 3])
        dist = rng.choice(["mid", "long"])
        generate_hybrid_scenario(
            scenario=kind,
            depth=depth,
            distance=dist,
            seed=seed,
            obstacles=10,
            out_root=OUT_LVL_1_5,
        )
        seed += 1

    return seed


def generate_experiment_lvl_5(seed_start: int) -> int:
    ensure_dir(OUT_LVL_5)
    seed = seed_start

    rng = random.Random(seed_start)
    kinds = ["deadend"] * 10 + ["corridor"] * 10
    rng.shuffle(kinds)

    for kind in kinds:
        depth = rng.choice([1, 2, 3])
        dist = rng.choice(["mid", "long"])
        generate_hybrid_scenario(
            scenario=kind,
            depth=depth,
            distance=dist,
            seed=seed,
            obstacles=10,
            out_root=OUT_LVL_5,
        )
        seed += 1

    return seed


def main() -> None:
    seed = 1
    seed = generate_experiment_lvl_1_4(seed)
    seed = generate_experiment_lvl_1_5(seed)
    seed = generate_experiment_lvl_5(seed)

    print("Done.")
    print(f"Generated in:\n  {OUT_LVL_1_4}\n  {OUT_LVL_1_5}\n  {OUT_LVL_5}")


if __name__ == "__main__":
    main()
