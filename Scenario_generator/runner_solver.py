import os
import random
import argparse

import mujoco
from mujoco import viewer

from generator_solver import SymbolicScenarioGenerator
from generator_proc import ProceduralScenarioGenerator, GridSpec as ProcGridSpec


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure symbolic grid generator (Z3)"
    )

    p.add_argument("--H", type=int, default=10)
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--scenario",
        choices=["standard", "deadend", "corridor"],
        default="corridor",
    )

    p.add_argument("--min-deadend-depth", type=int, default=4)
    p.add_argument("--min-corridor-length", type=int, default=4)

    p.add_argument("--exact-walls", type=int, default=None)
    p.add_argument("--min-walls", type=int, default=None)
    p.add_argument("--max-walls", type=int, default=None)

    p.add_argument("--target-r", type=int, default=None,
               help="Row of fixed target (global coords)")
    p.add_argument("--target-c", type=int, default=None,
               help="Col of fixed target (global coords)")

    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="If set, do not launch MuJoCo viewer",
    )

    return p.parse_args()


def solver_filename(
    *,
    H, W,
    exact_walls, min_walls, max_walls,
    deadend_depth, corridor_length,
    seed,
    ext=".xml",
):
    parts = [f"{H}x{W}"]

    if exact_walls is not None:
        parts.append(f"FWx{exact_walls}")
    else:
        if min_walls is not None:
            parts.append(f"FWmin{min_walls}")
        if max_walls is not None:
            parts.append(f"FWmax{max_walls}")

    if deadend_depth is not None:
        parts.append(f"DEp{deadend_depth}")
    if corridor_length is not None:
        parts.append(f"COl{corridor_length}")

    parts.append(f"seed{seed}")

    return "_".join(parts) + ext


def main():
    args = parse_args()

    seed = args.seed
    rng = random.Random(seed)

    H, W = args.H, args.W

    use_deadend = (args.scenario == "deadend")
    use_corridor = (args.scenario == "corridor")

    min_deadend_depth = args.min_deadend_depth
    min_corridorLength = args.min_corridor_length

    exact_walls = args.exact_walls
    min_walls = args.min_walls
    max_walls = args.max_walls

    grid, chosen_dead, chosen_depth, deadend_path, corridor_path = (
        SymbolicScenarioGenerator.generate_grid(
            H=H,
            W=W,
            deadend=use_deadend,
            corridor=use_corridor,
            min_deadend_depth=min_deadend_depth,
            min_corridorLength=min_corridorLength,
            z3_seed=rng.randint(0, 1_000_000),
            exact_walls=exact_walls,
            min_walls=min_walls,
            max_walls=max_walls,
            spawn=(1, 1),
        )
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
    scenario_type = scenario_type  # already defined

    filename = solver_filename(
        H=H, W=W,
        exact_walls=exact_walls,
        min_walls=min_walls,
        max_walls=max_walls,
        deadend_depth=min_deadend_depth if use_deadend else None,
        corridor_length=min_corridorLength if use_corridor else None,
        seed=seed,
    )

    outdir = os.path.join(base_root, "solver", scenario_type)
    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, filename)
    model_name = os.path.splitext(os.path.basename(path))[0]

    proc = ProceduralScenarioGenerator(ProcGridSpec(H=H, W=W))
    xml = proc.grid_to_mujoco_xml_base_compatible(grid, model_name=model_name)
    proc.write_xml(xml, path)
    print(f"Saved {path}")

    if not args.no_viewer:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
