import os
import random
from typing import Tuple

import mujoco
from mujoco import viewer

from generator_solver import SymbolicScenarioGenerator, GridSpec, SpawnSpec
from generator_proc import ProceduralScenarioGenerator, GridSpec as ProcGridSpec

Coord = Tuple[int, int]


def print_grid(grid):
    for row in grid:
        print("".join("#" if cell else "." for cell in row))


def build_scenario_filepath(
    base_root: str,
    scenario_type: str,   # "standard", "deadend", "corridor"
    H: int,
    W: int,
    seed: int,
    exact_walls=None,
    min_walls=None,
    max_walls=None,
    deadend_depth=None,
    corridor_length=None,
    prefix: str = "solver",
    ext: str = ".xml",
):
    import os

    dir_path = os.path.join(base_root, scenario_type)
    os.makedirs(dir_path, exist_ok=True)

    parts = [f"{prefix}_{H}x{W}"]

    if exact_walls is not None:
        parts.append(f"wExact{exact_walls}")
    else:
        if min_walls is not None:
            parts.append(f"wMin{min_walls}")
        if max_walls is not None:
            parts.append(f"wMax{max_walls}")

    if deadend_depth is not None:
        parts.append(f"deadD{deadend_depth}")
    if corridor_length is not None:
        parts.append(f"corrL{corridor_length}")

    parts.append(f"seed{seed}")

    filename = "_".join(parts) + ext
    return os.path.join(dir_path, filename)


if __name__ == "__main__":
    seed = 7
    rng = random.Random(seed)

    H, W = 10, 10

    use_deadend = False
    use_corridor = True

    min_deadend_depth = 4
    min_corridorLength = 4
    exact_walls = 40
    min_walls = None
    max_walls = None

    sym_grid = GridSpec(
        H=H,
        W=W,
        deadend=use_deadend,
        corridor=use_corridor,
    )
    spawn = SpawnSpec(start=(1, 1))
    sym = SymbolicScenarioGenerator(sym_grid, spawn)

    grid, chosen_dead, chosen_depth, deadend_path, corridor_path = sym.generate_grid(
        min_deadend_depth=min_deadend_depth,
        min_corridorLength=min_corridorLength,
        z3_seed=rng.randint(0, 1_000_000),
        exact_walls=exact_walls,
        min_walls=min_walls,
        max_walls=max_walls,
    )

    print("=== SOLVER-GENERATED GRID ===")
    print_grid(grid)

    if use_deadend:
        print("\nDeadend endpoint:", chosen_dead, "depth:", chosen_depth)
        print("Deadend path:", deadend_path)

    if use_corridor:
        print("\nCorridor path:", corridor_path)

    if use_corridor:
        scenario_type = "corridor"
    elif use_deadend:
        scenario_type = "deadend"
    else:
        scenario_type = "standard"

    base_root = "scenarios"
    path = build_scenario_filepath(
        base_root=base_root,
        scenario_type=scenario_type,
        H=H,
        W=W,
        seed=seed,
        exact_walls=exact_walls,
        min_walls=min_walls,
        max_walls=max_walls,
        deadend_depth=min_deadend_depth if use_deadend else None,
        corridor_length=min_corridorLength if use_corridor else None,
        prefix="solver",
    )

    model_name = os.path.splitext(os.path.basename(path))[0]

    proc = ProceduralScenarioGenerator(ProcGridSpec(H=H, W=W))
    xml = proc.grid_to_mujoco_xml_base_compatible(grid, model_name=model_name)
    proc.write_xml(xml, path)
    print(f"Saved {path}")

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    viewer.launch(model, data)