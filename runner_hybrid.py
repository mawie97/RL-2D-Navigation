import os
import random
import argparse
import math
from typing import List, Tuple, Optional

import mujoco
from mujoco import viewer
from z3 import Int, Solver, And, Or, Distinct, sat

from generator_solver import SymbolicScenarioGenerator
from generator_proc import ProceduralScenarioGenerator, GridSpec as ProcGridSpec
from generator_bresenham import BresenhamStandardGenerator

Coord = Tuple[int, int]

def at_constraints_satisfied(
    d: Optional[float],
    at_min: Optional[int],
    at_max: Optional[int],
    at_exact: Optional[int],
) -> bool:
    if d is None:
        return False
    if at_exact is not None:
        return d == at_exact
    if at_min is not None and d < at_min:
        return False
    if at_max is not None and d > at_max:
        return False
    return True

# Helper to find possible offsets in deadend/corridor scenarios
def candidate_offsets_for_at(
    *,
    block_H: int,
    block_W: int,
    full_H: int,
    full_W: int,
    fixed_target: Coord,
    agent_rel_block: Coord,       # (ra, ca) in block coordinates
    at_min: Optional[int] = None,
    at_max: Optional[int] = None,
    at_exact: Optional[int] = None,
) -> list[Coord]:
    """
    Return candidate (off_r, off_c) offsets where the *straight-line* distance
    between agent (on the structure) and fixed_target is compatible with the
    requested AT constraints.

    grid coords for agent: (off_r + ra, off_c + ca)
    """

    (tr, tc) = fixed_target
    (ra, ca) = agent_rel_block

    # interpret constraints
    if at_exact is not None:
        desired_min = at_exact
        desired_max = at_exact
    else:
        desired_min = at_min if at_min is not None else 0
        desired_max = at_max if at_max is not None else float("inf")

    candidates: list[Coord] = []

    # brute force all offsets where block fits inside full grid
    for off_r in range(full_H - block_H + 1):
        for off_c in range(full_W - block_W + 1):
            ar = off_r + ra
            ac = off_c + ca

            # quick sanity: must be inside grid by construction, so skip checks

            # use straight-line (euclidean) distance as heuristic
            dr = tr - ar
            dc = tc - ac
            d_line = math.sqrt(dr * dr + dc * dc)

            if desired_min <= d_line <= desired_max:
                candidates.append((off_r, off_c))

    return candidates

# ------------------------------------------------------------
# Filename builder (unchanged logic, now fed from CLI)
# ------------------------------------------------------------
def scenario_filename(
    *,
    H: int,
    W: int,
    block_walls: int,
    full_exact_walls: Optional[int],
    full_min_walls: Optional[int],
    full_max_walls: Optional[int],
    deadend_depth: Optional[int],
    corridor_length: Optional[int],
    num_agents: int,
    num_targets: int,
    aa_min: Optional[int],
    aa_max: Optional[int],
    aa_exact: Optional[int],
    tt_min: Optional[int],
    tt_max: Optional[int],
    tt_exact: Optional[int],
    at_min: Optional[int],
    at_max: Optional[int],
    at_exact: Optional[int],
    offset_r: int,
    offset_c: int,
    seed: int,
    ext: str = ".xml",
) -> str:

    parts = [f"{H}x{W}"]

    # block walls (always present)
    parts.append(f"BW{block_walls}")

    # full grid walls
    if full_exact_walls is not None:
        parts.append(f"FWx{full_exact_walls}")
    else:
        if full_min_walls is not None:
            parts.append(f"FWmin{full_min_walls}")
        if full_max_walls is not None:
            parts.append(f"FWmax{full_max_walls}")

    # deadend / corridor
    if deadend_depth is not None:
        parts.append(f"DEp{deadend_depth}")
    if corridor_length is not None:
        parts.append(f"COl{corridor_length}")

    # agents & targets
    parts.append(f"AG{num_agents}")
    parts.append(f"TG{num_targets}")

    # distance constraints
    def add_dist(tag, mn, mx, eq):
        if eq is not None:
            parts.append(f"{tag}eq{eq}")
        else:
            if mn is not None:
                parts.append(f"{tag}min{mn}")
            if mx is not None:
                parts.append(f"{tag}max{mx}")

    add_dist("AA", aa_min, aa_max, aa_exact)
    add_dist("TT", tt_min, tt_max, tt_exact)
    add_dist("AT", at_min, at_max, at_exact)

    # offset
    if offset_r != 0 or offset_c != 0:
        parts.append(f"Off{offset_r}-{offset_c}")

    # seed
    parts.append(f"seed{seed}")

    return "_".join(parts) + ext



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
    
    # Deadend position
    p.add_argument(
        "--deadend-agent-pos",
        choices=["start", "middle", "end"],
        default="end",  # or whatever you like
        help="Where to place the agent on a deadend path when pinning",
    )

    # Wall budgets
    p.add_argument(
        "--block-walls",
        type=int,
        default=None,
        help="Exact number of WALL cells inside the symbolic block",
    )

    p.add_argument(
        "--full-exact-walls",
        type=int,
        default=None,
        help="Exact number of WALL cells in the FULL grid "
             "(if set, overrides full-min-walls/full-max-walls)",
    )
    p.add_argument(
        "--full-min-walls",
        type=int,
        default=None,
        help="Minimum number of WALL cells in the FULL grid",
    )
    p.add_argument(
        "--full-max-walls",
        type=int,
        default=None,
        help="Maximum number of WALL cells in the FULL grid",
    )

    # Seeds
    p.add_argument("--seed", type=int, default=0, help="Master random seed")

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

    p.add_argument("--target-r", type=int, default=None,
               help="Row of fixed target (global coords)")
    p.add_argument("--target-c", type=int, default=None,
               help="Col of fixed target (global coords)")
    
    p.add_argument(
        "--agent-r",
        type=int,
        default=None,
        help="Row of agent start (only used in standard scenarios)",
    )
    p.add_argument(
        "--agent-c",
        type=int,
        default=None,
        help="Column of agent start (only used in standard scenarios)",
    )



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

    # ---------- scenario flags ----------
    use_deadend = (args.scenario == "deadend")
    use_corridor = (args.scenario == "corridor")
    scenario_type = args.scenario

    # ----- block walls (symbolic) -----
    # Only relevant when we actually use a symbolic block
    if scenario_type == "standard":
        # no symbolic block → just pick something for filename
        exact_block_walls = 0
    else:
        if args.block_walls is None:
            raise ValueError(
                "For deadend/corridor scenarios you must provide --block-walls."
            )
        if args.block_walls < 0 or args.block_walls > block_cells:
            raise ValueError(f"block-walls must be in [0, {block_cells}]")
        exact_block_walls = args.block_walls

    # ----- fixed target -----
    if args.target_r is not None and args.target_c is not None:
        fixed_target = (args.target_r, args.target_c)
    else:
        # default near bottom-right
        fixed_target = (full_H - 2, full_W - 2)

    # ----- full-grid walls (procedural) -----
    full_exact_walls: Optional[int] = None
    proc_min_walls: Optional[int] = None
    proc_max_walls: Optional[int] = None

    if args.full_exact_walls is not None:
        if not (0 <= args.full_exact_walls <= full_cells):
            raise ValueError(f"full-exact-walls must be in [0, {full_cells}]")
        full_exact_walls = args.full_exact_walls
        proc_min_walls = proc_max_walls = full_exact_walls
    elif args.full_min_walls is not None or args.full_max_walls is not None:
        if (args.full_min_walls is not None and
                args.full_max_walls is not None and
                args.full_min_walls > args.full_max_walls):
            raise ValueError("full-min-walls cannot be greater than full-max-walls")
        proc_min_walls = args.full_min_walls if args.full_min_walls is not None else 0
        proc_max_walls = args.full_max_walls if args.full_max_walls is not None else full_cells
    else:
        raise ValueError(
            "You must specify at least one of "
            "--full-exact-walls, --full-min-walls or --full-max-walls."
        )

    min_deadend_depth = args.deadend_depth
    min_corridorLength = args.corridor_length

    # ---------- distance constraints for entities ----------
    aa_min, aa_max, aa_exact = args.aa_min, args.aa_max, args.aa_exact
    at_min, at_max, at_exact = args.at_min, args.at_max, args.at_exact
    tt_min, tt_max, tt_exact = args.tt_min, args.tt_max, args.tt_exact

    # common offset (used in deadend/corridor, and filename)
    off_r = args.offset_r
    off_c = args.offset_c

    # single procedural generator instance
    proc_gen = ProceduralScenarioGenerator(ProcGridSpec(H=full_H, W=full_W))

    # Set defaults so they exist in all branches
    deadend_path_global: List[Coord] = []
    corridor_path_global: List[Coord] = []
    dead_global: Optional[Coord] = None
    mid_corr_global: Optional[Coord] = None


    # ============================
    # 1) STANDARD (Bresenham) CASE
    # ============================
    if scenario_type == "standard":
        # For now, assume 1 agent and 1 target. Enforce it.
        if args.num_agents != 1 or args.num_targets != 1:
            raise ValueError("Standard scenarios currently only support 1 agent and 1 target")

        # ----- target position (fixed) -----
        target = fixed_target

        # Do we have AT constraints?
        at_constraints = (
            at_min is not None
            or at_max is not None
            or at_exact is not None
        )

        if at_constraints:
            # Don't allow contradictory specification
            if args.agent_r is not None or args.agent_c is not None:
                raise ValueError(
                    "In standard scenarios, do not specify --agent-r/--agent-c "
                    "when using AT distance constraints. The agent position "
                    "will be derived automatically from the distance band."
                )

            # Find all candidate agent cells that satisfy the Euclidean AT band
            candidates: list[Coord] = []
            tr, tc = target
            for r in range(full_H):
                for c in range(full_W):
                    if (r, c) == target:
                        continue  # can't sit on the target

                    d_line = math.hypot(tr - r, tc - c)
                    if at_constraints_satisfied(d_line, at_min, at_max, at_exact):
                        candidates.append((r, c))

            if not candidates:
                raise RuntimeError(
                    "No agent position in the grid satisfies the requested "
                    "agent-target distance band for the given AT constraints."
                )

            # Pick one uniformly at random
            agent = rng.choice(candidates)

        else:
            # No AT constraints → use explicit agent position or default
            if args.agent_r is not None and args.agent_c is not None:
                agent = (args.agent_r, args.agent_c)
            else:
                # default if user doesn't specify
                agent = (1, 1)

            # basic sanity checks
            ar, ac = agent
            if not (0 <= ar < full_H and 0 <= ac < full_W):
                raise ValueError(
                    f"Agent position {agent} is out of bounds for grid {full_H}x{full_W}"
                )
            if agent == target:
                raise ValueError(
                    f"Agent position {agent} coincides with target position {target}"
                )

        # Map full-grid wall config for the Bresenham generator
        if args.full_exact_walls is not None:
            exact_walls = args.full_exact_walls
            min_walls = None
            max_walls = None
        else:
            exact_walls = None
            min_walls = args.full_min_walls
            max_walls = args.full_max_walls

        gen = BresenhamStandardGenerator(full_H, full_W)
        grid_full = gen.generate(
            agent=agent,
            target=target,
            exact_walls=exact_walls,
            min_walls=min_walls,
            max_walls=max_walls,
            line_wall_fraction=0.7,
            rng=rng,
        )

        agent_cells = [agent]
        target_cells = [target]

    # ===================================
    # 2) DEADEND / CORRIDOR (HYBRID) CASE
    # ===================================
    else:
        # Do we have AT constraints?
        at_constraints = (
            at_min is not None
            or at_max is not None
            or at_exact is not None
        )

        # We only support AT constraints in this special case:
        # - agent pinned on the symbolic structure
        # - exactly 1 agent and 1 target
        can_use_geom_AT = (
            at_constraints
            and args.pin_agent_on_structure
            and args.num_agents == 1
            and args.num_targets == 1
        )

        if at_constraints and not can_use_geom_AT:
            raise ValueError(
                "AT distance constraints for deadend/corridor are only "
                "supported when pin-agent-on-structure is set and "
                "num_agents = num_targets = 1."
            )

        # ---- 2A: generate ONE symbolic block ----
        block_bool, chosen_dead, chosen_depth, \
            deadend_path_block, corridor_path_block = SymbolicScenarioGenerator.generate_grid(
                H=block_H,
                W=block_W,
                deadend=use_deadend,
                corridor=use_corridor,
                min_deadend_depth=min_deadend_depth,
                min_corridorLength=min_corridorLength,
                z3_seed=rng.randint(0, 1_000_000),
                exact_walls=exact_block_walls,
                min_walls=None,
                max_walls=None,
                spawn=(1, 1),
            )

        # choose agent position in BLOCK coordinates
        if use_deadend:
            if not deadend_path_block:
                raise RuntimeError("Deadend scenario, but solver returned empty deadend_path_block.")
            pos_mode = args.deadend_agent_pos  # "start"/"middle"/"end"
            if pos_mode == "start":
                agent_rel_block = deadend_path_block[0]
            elif pos_mode == "middle":
                mid_idx = len(deadend_path_block) // 2
                agent_rel_block = deadend_path_block[mid_idx]
            else:
                agent_rel_block = deadend_path_block[-1]

        elif use_corridor:
            if not corridor_path_block:
                raise RuntimeError("Corridor scenario, but solver returned empty corridor_path_block.")
            Lc = len(corridor_path_block)
            mid_idx = Lc // 2
            agent_rel_block = corridor_path_block[mid_idx]
        else:
            # shouldn't happen in this branch
            raise RuntimeError("Internal error: hybrid branch without deadend or corridor.")

        # ---- 2B: decide offsets ----
        if can_use_geom_AT:
            # Use Euclidean distance band to filter offsets
            candidate_offsets = candidate_offsets_for_at(
                block_H=block_H,
                block_W=block_W,
                full_H=full_H,
                full_W=full_W,
                fixed_target=fixed_target,
                agent_rel_block=agent_rel_block,
                at_min=at_min,
                at_max=at_max,
                at_exact=at_exact,
            )

            if not candidate_offsets:
                raise RuntimeError(
                    "No placement of this symbolic block can achieve the requested "
                    "Euclidean agent-target distance band."
                )

            rng.shuffle(candidate_offsets)  # randomize search order a bit
        else:
            # No AT constraints → just use the user-provided offset
            candidate_offsets = [(off_r, off_c)]

        success = False
        last_error = None

        # ---- 2C: try offsets until proc generation works ----
        for off_r_try, off_c_try in candidate_offsets:
            offset = (off_r_try, off_c_try)

            # sanity check: block must fit inside full grid
            if off_r_try < 0 or off_c_try < 0 or \
               off_r_try + block_H > full_H or off_c_try + block_W > full_W:
                continue

            base_grid = embed_block(full_H, full_W, block_bool, offset)

            root_small = (1, 1)
            root_global = (off_r_try + root_small[0],
                           off_c_try + root_small[1])

            # build global deadend / corridor metadata
            deadend_path_global = []
            dead_global = None
            if use_deadend and deadend_path_block:
                deadend_path_global = [
                    (off_r_try + r, off_c_try + c)
                    for (r, c) in deadend_path_block
                ]
                dead_global = deadend_path_global[-1]

            corridor_path_global = []
            mid_corr_global = None
            if use_corridor and corridor_path_block:
                corridor_path_global = [
                    (off_r_try + r, off_c_try + c)
                    for (r, c) in corridor_path_block
                ]
                mid_idx_corr = len(corridor_path_global) // 2
                mid_corr_global = corridor_path_global[mid_idx_corr]

            frozen = {
                (off_r_try + r, off_c_try + c)
                for r in range(block_H)
                for c in range(block_W)
            }

            try:
                grid_full = proc_gen.generate_with_requirements(
                    root=root_global,
                    min_walls=proc_min_walls,
                    max_walls=proc_max_walls,
                    min_corridor=None,
                    min_deadends=None,
                    min_deadend_depth=min_deadend_depth + 1,
                    rng=rng,
                    max_tries=200,
                    base_grid=base_grid,
                    frozen=frozen,
                )
            except Exception as e:
                last_error = e
                continue  # try next offset

            # compute agent global pos
            agent_pos = (
                off_r_try + agent_rel_block[0],
                off_c_try + agent_rel_block[1],
            )

            # make sure agent and target are on FREE cells
            ar, ac = agent_pos
            tr, tc = fixed_target
            if grid_full[ar][ac] or grid_full[tr][tc]:
                # one of them ended up inside a wall → reject this offset
                continue

            # if we’re here, we’re happy:
            agent_cells = [agent_pos]
            target_cells = [fixed_target]
            # and also update off_r/off_c used later for filename
            off_r = off_r_try
            off_c = off_c_try
            success = True
            break

        if not success:
            raise RuntimeError(
                "Could not embed block & generate full grid for any candidate offset. "
                f"Last error: {last_error}"
            )


    # ============================
    # PRINT GRID & STRUCTURE INFO
    # ============================
    print("Grid (# = wall, . = free):")
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


    # At this point, agent_cells / target_cells are set for ALL scenarios
    print("Agents at:", agent_cells)
    print("Targets at:", target_cells)

    # ---------- decide scenario type for directory ----------
    if use_corridor:
        scenario_dir = "corridor"
    elif use_deadend:
        scenario_dir = "deadend"
    else:
        scenario_dir = "standard"

    # For filename: if we used an exact wall budget, record that.
    fname_exact = full_exact_walls
    fname_min = proc_min_walls if fname_exact is None else None
    fname_max = proc_max_walls if fname_exact is None else None

    filename = scenario_filename(
        H=full_H,
        W=full_W,
        block_walls=exact_block_walls,
        full_exact_walls=full_exact_walls,
        full_min_walls=fname_min,
        full_max_walls=fname_max,
        deadend_depth=min_deadend_depth if use_deadend else None,
        corridor_length=min_corridorLength if use_corridor else None,
        num_agents=args.num_agents,
        num_targets=args.num_targets,
        aa_min=aa_min, aa_max=aa_max, aa_exact=aa_exact,
        tt_min=tt_min, tt_max=tt_max, tt_exact=tt_exact,
        at_min=at_min, at_max=at_max, at_exact=at_exact,
        offset_r=off_r,
        offset_c=off_c,
        seed=master_seed,
    )

    outdir = os.path.join("scenarios", "hybrid", scenario_dir)
    os.makedirs(outdir, exist_ok=True)

    path = os.path.join(outdir, filename)
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
