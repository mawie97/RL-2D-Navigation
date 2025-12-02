import os
import random
import argparse
from typing import List, Tuple, Optional
from collections import deque

import mujoco
from mujoco import viewer
from z3 import Int, Solver, And, Or, Distinct, sat

from generator_solver import SymbolicScenarioGenerator, GridSpec as SymGridSpec, SpawnSpec
from generator_proc import ProceduralScenarioGenerator, GridSpec as ProcGridSpec

Coord = Tuple[int, int]


# ------------------------------------------------------------
# Filename builder (unchanged logic, now fed from CLI)
# ------------------------------------------------------------
def build_scenario_filepath(
    base_root: str,
    scenario_type: str,   # "standard", "deadend", "corridor"
    H: int,
    W: int,
    seed: int,
    # wall constraints (for the FULL grid)
    exact_walls: Optional[int] = None,
    min_walls: Optional[int] = None,
    max_walls: Optional[int] = None,
    # structural stuff
    deadend_depth: Optional[int] = None,
    corridor_length: Optional[int] = None,
    # distance constraints
    aa_min: Optional[int] = None,
    aa_max: Optional[int] = None,
    aa_exact: Optional[int] = None,
    tt_min: Optional[int] = None,
    tt_max: Optional[int] = None,
    tt_exact: Optional[int] = None,
    at_min: Optional[int] = None,
    at_max: Optional[int] = None,
    at_exact: Optional[int] = None,
    prefix: str = "hybrid",
    ext: str = ".xml",
) -> str:
    dir_path = os.path.join(base_root, scenario_type)
    os.makedirs(dir_path, exist_ok=True)

    parts = [f"{prefix}_{H}x{W}"]

    # walls
    if exact_walls is not None:
        parts.append(f"wExact{exact_walls}")
    else:
        if min_walls is not None:
            parts.append(f"wMin{min_walls}")
        if max_walls is not None:
            parts.append(f"wMax{max_walls}")

    # deadend / corridor
    if deadend_depth is not None:
        parts.append(f"deadD{deadend_depth}")
    if corridor_length is not None:
        parts.append(f"corrL{corridor_length}")

    def add_dist(tag: str, dmin, dmax, deq):
        if deq is not None:
            parts.append(f"{tag}Eq{deq}")
        else:
            if dmin is not None:
                parts.append(f"{tag}Min{dmin}")
            if dmax is not None:
                parts.append(f"{tag}Max{dmax}")

    add_dist("aa", aa_min, aa_max, aa_exact)
    add_dist("tt", tt_min, tt_max, tt_exact)
    add_dist("at", at_min, at_max, at_exact)

    parts.append(f"seed{seed}")

    filename = "_".join(parts) + ext
    return os.path.join(dir_path, filename)


# ------------------------------------------------------------
# Embedding symbolic block into larger grid
# ------------------------------------------------------------
def embed_block(
    big_H: int,
    big_W: int,
    block: List[List[bool]],
    offset: Coord,
) -> List[List[bool]]:
    Hs, Ws = len(block), len(block[0])
    br0, bc0 = offset
    assert br0 + Hs <= big_H and bc0 + Ws <= big_W, "offset out of range"

    grid = [[True for _ in range(big_W)] for _ in range(big_H)]
    for r in range(Hs):
        for c in range(Ws):
            grid[br0 + r][bc0 + c] = block[r][c]
    return grid


# ------------------------------------------------------------
# Z3 placement of agents/targets on a fixed grid
# (unchanged, just called from main with CLI params)
# ------------------------------------------------------------
def place_entities_with_solver(
    grid: List[List[bool]],
    num_agents: int,
    num_targets: int,
    min_agent_agent: Optional[int] = None,
    max_agent_agent: Optional[int] = None,
    exact_agent_agent: Optional[int] = None,
    min_agent_target: Optional[int] = None,
    max_agent_target: Optional[int] = None,
    exact_agent_target: Optional[int] = None,
    min_target_target: Optional[int] = None,
    max_target_target: Optional[int] = None,
    exact_target_target: Optional[int] = None,
    pinned_target_cells: Optional[List[Coord]] = None,
    pinned_agent_cells: Optional[List[Coord]] = None,
    z3_seed: Optional[int] = None,
) -> Tuple[List[Coord], List[Coord]]:
    H, W = len(grid), len(grid[0])

    free_cells: List[Coord] = [
        (r, c) for r in range(H) for c in range(W) if not grid[r][c]
    ]
    n = len(free_cells)
    pinned_agent_cells = pinned_agent_cells or []
    pinned_target_cells = pinned_target_cells or []

    if n < num_agents + num_targets:
        raise RuntimeError("Not enough free cells for agents + targets")

    def bfs_from_idx(idx: int) -> List[int]:
        sr, sc = free_cells[idx]
        dgrid = [[-1] * W for _ in range(H)]
        if grid[sr][sc]:
            return [-1] * n

        q = deque()
        q.append((sr, sc))
        dgrid[sr][sc] = 0

        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and not grid[nr][nc]
                    and dgrid[nr][nc] == -1
                ):
                    dgrid[nr][nc] = dgrid[r][c] + 1
                    q.append((nr, nc))

        dist = [-1] * n
        for i, (r, c) in enumerate(free_cells):
            dist[i] = dgrid[r][c]
        return dist

    dist_matrix: List[List[int]] = [bfs_from_idx(i) for i in range(n)]

    agents = [Int(f"A{i}") for i in range(num_agents)]
    targets = [Int(f"T{j}") for j in range(num_targets)]
    all_vars = agents + targets

    s = Solver()
    if z3_seed is not None:
        s.set("random_seed", z3_seed)

    for v in all_vars:
        s.add(And(v >= 0, v < n))

    if len(all_vars) > 1:
        s.add(Distinct(*all_vars))

        # Pin agents to specific free cells, if requested
    for i, cell in enumerate(pinned_agent_cells):
        if i >= num_agents:
            raise ValueError("More pinned_agent_cells than num_agents")
        if cell not in free_cells:
            raise ValueError(f"Pinned agent cell {cell} is not a free cell")
        idx = free_cells.index(cell)
        s.add(agents[i] == idx)

    for j, cell in enumerate(pinned_target_cells):
        if j >= num_targets:
            raise ValueError("More pinned_target_cells than num_targets")
        if cell not in free_cells:
            raise ValueError(f"Pinned target cell {cell} is not a free cell")
        idx = free_cells.index(cell)
        s.add(targets[j] == idx)

    def add_pair_dist_constraint(
        v1,
        v2,
        min_d: Optional[int],
        max_d: Optional[int],
        exact_d: Optional[int],
    ):
        if min_d is None and max_d is None and exact_d is None:
            return

        for i in range(n):
            for j in range(n):
                d = dist_matrix[i][j]
                if d == -1:
                    s.add(Or(v1 != i, v2 != j))
                    continue

                if exact_d is not None:
                    if d != exact_d:
                        s.add(Or(v1 != i, v2 != j))
                else:
                    if min_d is not None and d < min_d:
                        s.add(Or(v1 != i, v2 != j))
                    if max_d is not None and d > max_d:
                        s.add(Or(v1 != i, v2 != j))

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            add_pair_dist_constraint(
                agents[i], agents[j],
                min_agent_agent, max_agent_agent, exact_agent_agent
            )

    for i in range(num_targets):
        for j in range(i + 1, num_targets):
            add_pair_dist_constraint(
                targets[i], targets[j],
                min_target_target, max_target_target, exact_target_target
            )

    for ai in range(num_agents):
        for tj in range(num_targets):
            add_pair_dist_constraint(
                agents[ai], targets[tj],
                min_agent_target, max_agent_target, exact_agent_target
            )

    if s.check() != sat:
        raise RuntimeError("No entity placement satisfies distance constraints")

    m = s.model()
    agent_cells = [free_cells[m[a].as_long()] for a in agents]
    target_cells = [free_cells[m[t].as_long()] for t in targets]
    return agent_cells, target_cells


# ------------------------------------------------------------
# CLI parsing
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Hybrid symbolic+procedural scenario generator"
    )

    # Grid sizes
    p.add_argument("--full-H", type=int, default=20)
    p.add_argument("--full-W", type=int, default=20)
    p.add_argument("--block-H", type=int, default=10)
    p.add_argument("--block-W", type=int, default=10)

    # Block offset inside full grid (top-left corner of symbolic block)
    p.add_argument(
        "--offset-r",
        type=int,
        default=0,
        help="Row offset of block inside full grid (top-left row index)",
    )
    p.add_argument(
        "--offset-c",
        type=int,
        default=0,
        help="Column offset of block inside full grid (top-left col index)",
    )

    # Scenario type
    p.add_argument(
        "--scenario",
        choices=["standard", "deadend", "corridor"],
        default="deadend",
        help="Which symbolic structure to enforce in the block",
    )

    # Free-cell budgets
    p.add_argument(
        "--block-free",
        type=int,
        required=True,
        help="Exact number of FREE cells inside the symbolic block",
    )
    p.add_argument(
        "--total-free",
        type=int,
        required=True,
        help="Exact number of FREE cells in the FULL grid",
    )

    # Seeds
    p.add_argument("--seed", type=int, default=7, help="Master random seed")

    # Structure parameters
    p.add_argument(
        "--deadend-depth",
        type=int,
        default=3,
        help="Deadend chain length in edges (block-level)",
    )
    p.add_argument(
        "--corridor-length",
        type=int,
        default=3,
        help="Corridor chain length in edges (block-level)",
    )

    # Entity counts
    p.add_argument("--num-agents", type=int, default=1)
    p.add_argument("--num-targets", type=int, default=1)

    # Whether to pin an agent onto the symbolic structure
    p.add_argument(
        "--pin-agent-on-structure",
        action="store_true",
        help="If set, pin agent 0 to the deadend endpoint or corridor midpoint",
    )


    # Distance constraints
    def add_dist(prefix: str):
        p.add_argument(f"--{prefix}-min", type=int, default=None)
        p.add_argument(f"--{prefix}-max", type=int, default=None)
        p.add_argument(f"--{prefix}-exact", type=int, default=None)

    add_dist("aa")  # agent-agent
    add_dist("tt")  # target-target
    add_dist("at")  # agent-target

    # Viewer toggle
    p.add_argument(
        "--no-viewer",
        action="store_true",
        help="If set, do not launch MuJoCo viewer",
    )

    return p.parse_args()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    args = parse_args()
    master_seed = args.seed
    rng = random.Random(master_seed)

    # ---------- sizes & sanity checks ----------
    full_H, full_W = args.full_H, args.full_W
    block_H, block_W = args.block_H, args.block_W

    block_cells = block_H * block_W
    full_cells = full_H * full_W

    if args.block_free < 1 or args.block_free > block_cells:
        raise ValueError(f"block-free must be in [1, {block_cells}]")
    if args.total_free < args.block_free or args.total_free > full_cells:
        raise ValueError(
            f"total-free must be in [{args.block_free}, {full_cells}]"
        )

    # Convert FREE counts to WALL counts
    exact_block_walls = block_cells - args.block_free
    exact_full_walls = full_cells - args.total_free

    # ---------- scenario flags ----------
    use_deadend = (args.scenario == "deadend")
    use_corridor = (args.scenario == "corridor")

    min_deadend_depth = args.deadend_depth
    min_corridorLength = args.corridor_length

    # ---------- symbolic block config ----------
    sym_grid = SymGridSpec(
        H=block_H,
        W=block_W,
        deadend=use_deadend,
        corridor=use_corridor,
    )
    spawn = SpawnSpec(start=(1, 1))
    sym_gen = SymbolicScenarioGenerator(sym_grid, spawn)

    z3_seed = rng.randint(0, 1_000_000)
    block_bool, chosen_dead, chosen_depth, deadend_path_block, corridor_path_block = (
        sym_gen.generate_grid(
            min_deadend_depth=min_deadend_depth,
            min_corridorLength=min_corridorLength,
            z3_seed=z3_seed,
            # single source of truth: walls in block from CLI
            exact_walls=exact_block_walls,
            min_walls=None,
            max_walls=None,
        )
    )

    # ---------- embed symbolic block into full grid ----------
    off_r = args.offset_r
    off_c = args.offset_c
    offset = (off_r, off_c)

    # sanity check: block must fit inside full grid at this offset
    if off_r < 0 or off_c < 0 or off_r + block_H > full_H or off_c + block_W > full_W:
        raise ValueError(
            f"Block offset ({off_r}, {off_c}) with size {block_H}x{block_W} "
            f"does not fit inside full grid {full_H}x{full_W}"
        )

    base_grid = embed_block(full_H, full_W, block_bool, offset)

    root_small = spawn.start
    root_global = (offset[0] + root_small[0],
                   offset[1] + root_small[1])
    
    deadend_path_global: List[Coord] = []
    dead_global: Optional[Coord] = None
    if use_deadend and chosen_dead is not None: # If the deadend flag = True and 
        dead_global = (offset[0] + chosen_dead[0], offset[1] + chosen_dead[1])
        deadend_path_global = [
            (offset[0] + r, offset[1] + c)
            for (r, c) in deadend_path_block
        ]

    corridor_path_global: List[Coord] = []
    mid_corr_global: Optional[Coord] = None
    if use_corridor:
        if not corridor_path_block:
            raise RuntimeError(
                "Corridor path is empty – solver corridor constraints too weak/disabled"
            )
        corridor_path_global = [
            (offset[0] + r, offset[1] + c)
            for (r, c) in corridor_path_block
        ]
        L_corr = len(corridor_path_block)
        mid_idx_corr = L_corr // 2
        mid_corr_block = corridor_path_block[mid_idx_corr]
        mid_corr_global = (offset[0] + mid_corr_block[0],
                           offset[1] + mid_corr_block[1])

    frozen = set(
        (offset[0] + r, offset[1] + c)
        for r in range(block_H)
        for c in range(block_W)
    )

    # ---------- procedural fill config ----------
    # We want EXACT total walls = exact_full_walls over the FULL grid.
    proc_min_walls = exact_full_walls
    proc_max_walls = exact_full_walls

    proc_gen = ProceduralScenarioGenerator(ProcGridSpec(H=full_H, W=full_W))
    grid_full = proc_gen.generate_with_requirements(
        root=root_global,
        min_walls=proc_min_walls,
        max_walls=proc_max_walls,
        min_corridor=None,
        min_deadends=None,
        min_deadend_depth=min_deadend_depth + 1,  # same default as before
        rng=rng,
        max_tries=200,
        base_grid=base_grid,
        frozen=frozen,
    )

    print("Hybrid grid (# = wall, . = free):")
    for r in range(full_H):
        print("".join("#" if grid_full[r][c] else "." for c in range(full_W)))

    if use_deadend and dead_global is not None:
        print("Dead-end endpoint (global):", dead_global)
        print("Dead-end path (global):", deadend_path_global)
        print("Deadend length:", min_deadend_depth)

    if use_corridor:
        print("Corridor path (global):")
        for step in corridor_path_global:
            print("  ", step)
        print("Middle of corridor (global):", mid_corr_global)

    # ---------- distance constraints for entities ----------
    num_agents = args.num_agents
    num_targets = args.num_targets

    aa_min, aa_max, aa_exact = args.aa_min, args.aa_max, args.aa_exact
    at_min, at_max, at_exact = args.at_min, args.at_max, args.at_exact
    tt_min, tt_max, tt_exact = args.tt_min, args.tt_max, args.tt_exact

    pinned_agents: List[Coord] = []

    if args.pin_agent_on_structure:
        # Only if the user *wants* the agent pinned on the structure
        if use_deadend and dead_global is not None:
            pinned_agents.append(dead_global)
        if use_corridor and mid_corr_global is not None:
            pinned_agents.append(mid_corr_global)
    # else: leave pinned_agents empty → no forced placement

    agent_cells, target_cells = place_entities_with_solver(
        grid_full,
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        min_agent_agent=aa_min,
        max_agent_agent=aa_max,
        exact_agent_agent=aa_exact,
        min_agent_target=at_min,
        max_agent_target=at_max,
        exact_agent_target=at_exact,
        min_target_target=tt_min,
        max_target_target=tt_max,
        exact_target_target=tt_exact,
        pinned_agent_cells=pinned_agents if pinned_agents else None,
        pinned_target_cells=None,
        z3_seed=rng.randint(0, 1_000_000),
    )

    print("Agents at:", agent_cells)
    print("Targets at:", target_cells)

    # ---------- decide scenario type ----------
    scenario_type = args.scenario

    base_root = "scenarios"

    path = build_scenario_filepath(
        base_root=base_root,
        scenario_type=scenario_type,
        H=full_H,
        W=full_W,
        seed=master_seed,
        exact_walls=exact_full_walls,
        min_walls=None,
        max_walls=None,
        deadend_depth=min_deadend_depth if use_deadend else None,
        corridor_length=min_corridorLength if use_corridor else None,
        aa_min=aa_min, aa_max=aa_max, aa_exact=aa_exact,
        tt_min=tt_min, tt_max=tt_max, tt_exact=tt_exact,
        at_min=at_min, at_max=at_max, at_exact=at_exact,
        prefix="hybrid",
    )

    model_name = os.path.splitext(os.path.basename(path))[0]
    xml = proc_gen.grid_to_mujoco_xml_base_compatible(
        grid_full,
        model_name=model_name,
        agent_cells=agent_cells,
        target_cells=target_cells,
    )

    proc_gen.write_xml(xml, path)
    print(f"Saved {path}")

    if not args.no_viewer:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
