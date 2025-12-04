import os
import random
import argparse

import mujoco
from mujoco import viewer

from generator_proc import ProceduralScenarioGenerator, GridSpec


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure procedural grid generator with BFS guarantees"
    )

    p.add_argument("--H", type=int, default=20)
    p.add_argument("--W", type=int, default=20)
    p.add_argument("--root-r", type=int, default=1)
    p.add_argument("--root-c", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--min-walls", type=int, default=80)
    p.add_argument("--max-walls", type=int, default=100)

    p.add_argument("--min-corridor", type=int, default=12)
    p.add_argument("--min-deadends", type=int, default=1)
    p.add_argument("--min-deadend-depth", type=int, default=4)

    p.add_argument("--max-tries", type=int, default=200)

    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="If set, do not launch MuJoCo viewer",
    )

    return p.parse_args()


def main():
    args = parse_args()

    H, W = args.H, args.W
    root = (args.root_r, args.root_c)
    rng = random.Random(args.seed)

    gen = ProceduralScenarioGenerator(GridSpec(H=H, W=W))

    grid = gen.generate_with_requirements(
        root=root,
        min_walls=args.min_walls,
        max_walls=args.max_walls,
        min_corridor=args.min_corridor,
        min_deadends=args.min_deadends,
        min_deadend_depth=args.min_deadend_depth,
        rng=rng,
        max_tries=args.max_tries,
    )

    print("Preview:")
    for r in range(H):
        print("".join("#" if grid[r][c] else "." for c in range(W)))

    xml = gen.grid_to_mujoco_xml_base_compatible(
        grid, model_name=f"connected_{H}x{W}"
    )
    path = f"scenarios/connected_{H}x{W}.xml"
    gen.write_xml(xml, path)
    print(f"Saved {path}")

    if not args.no_viewer:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
