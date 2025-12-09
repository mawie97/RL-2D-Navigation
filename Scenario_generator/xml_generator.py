import os
import math
import random
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

from generator_bresenham import BresenhamStandardGenerator
from generator_proc import ProceduralScenarioGenerator, GridSpec
from generator_solver import SymbolicScenarioGenerator

Coord = Tuple[int, int]

# ============================
# Config
# ============================

FULL_H = 15
FULL_W = 15

# Block size for deadend/corridor
BLOCK_H = 5
BLOCK_W = 5

# Single output folder
OUT_ROOT = os.path.join("scenarios", "all")

# Path to base XML template
BASE_XML_PATH = os.path.join(os.path.dirname(__file__), "base_layout.xml")

# Same arena / height as grid_to_mujoco_xml_base_compatible 
ARENA_HALF_EXTENT = 15.0
OBSTACLE_HEIGHT = 0.25  # used as z for obstacles + half-height
AGENT_HEIGHT = 0.25     # vertical half-size (same as in generator_proc)


# Fixed target position in grid coords (row, col)
TARGET_DEFAULT: Coord = (FULL_H - 2, FULL_W - 2)


@dataclass
class ScenarioMeta:
    level: int
    scenario: str          # "standard" | "deadend" | "corridor"
    obstacles: int
    distance: int
    seed: int
    depth: Optional[int]   # for deadend/corridor only
    path: str              # path to XML file


# ============================
# Distance helpers
# ============================

def distance_band_for_label(label: int) -> Tuple[int, int]:
    """
    Map the 'distance' label (5, 10, 15) to a Manhattan distance band.
    We treat them as short/mid/long categories:

    5  -> short  -> 3-5
    10 -> mid    -> 6-10
    15 -> long   -> 11-15
    """
    if label == 5:
        return 3, 5
    if label == 10:
        return 6, 10
    if label == 15:
        return 11, 15
    raise ValueError(f"Unsupported distance label: {label}")

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


def candidate_offsets_for_at(
    *,
    block_H: int,
    block_W: int,
    full_H: int,
    full_W: int,
    fixed_target: Coord,
    agent_rel_block: Coord,
    at_min: Optional[int] = None,
    at_max: Optional[int] = None,
    at_exact: Optional[int] = None,
) -> List[Coord]:
    """
    Return candidate (off_r, off_c) offsets where the Manhattan distance
    between the agent (inside the block) and target matches the constraints.
    """
    (tr, tc) = fixed_target
    (ra, ca) = agent_rel_block

    if at_exact is not None:
        desired_min = at_exact
        desired_max = at_exact
    else:
        desired_min = at_min if at_min is not None else 0
        desired_max = at_max if at_max is not None else float("inf")

    candidates: List[Coord] = []

    for off_r in range(full_H - block_H + 1):
        for off_c in range(full_W - block_W + 1):
            ar = off_r + ra
            ac = off_c + ca

            # *** Manhattan distance ***
            d_line = abs(tr - ar) + abs(tc - ac)

            if desired_min <= d_line <= desired_max:
                candidates.append((off_r, off_c))

    return candidates


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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================
# XML construction using base_layout.xml
# ============================

def build_xml_from_base(
    grid: List[List[bool]],
    agent_cell: Coord,
    target_cell: Coord,
    model_name: str,
) -> str:
    """
    Load base_layout.xml and inject:
    - obstacle bodies named obstacle1, obstacle2, ... with geoms obstacleN_geom
    - agent body position + size
    - goal body position
    - distance sensors from agent_geom to each obstacleN_geom

    Everything else (assets, plugins, contacts, etc.) stays as in base_layout.
    """
    if not os.path.exists(BASE_XML_PATH):
        raise FileNotFoundError(f"BASE_XML_PATH not found: {BASE_XML_PATH}")

    with open(BASE_XML_PATH, "r", encoding="utf-8") as f:
        xml = f.read()

    # Update model name
    xml = xml.replace('model="simple_navigation"', f'model="{model_name}"', 1)

    H = len(grid)
    W = len(grid[0])
    A = ARENA_HALF_EXTENT
    height = OBSTACLE_HEIGHT
    rng = random.Random()

    cell_w = (2.0 * A) / W
    cell_h = (2.0 * A) / H
    half_w = cell_w * 0.4
    half_h = cell_h * 0.4
    max_jitter_x = 0.5 * cell_w - half_w
    max_jitter_y = 0.5 * cell_h - half_h

    def cell_to_xy(r: int, c: int):
        x = -A + (c + 0.5) * cell_w
        y = A - (r + 0.5) * cell_h
        return x, y
    
    def addJitter(pos: tuple[float, float]):
        x, y = pos
        x_jitter = rng.uniform(-max_jitter_x, max_jitter_x)
        y_jitter = rng.uniform(-max_jitter_y, max_jitter_y)
        return x + x_jitter, y + y_jitter
        

    # --- Obstacles (True = wall) ---
    obstacles_xml: List[str] = []
    obstacle_geom_names: List[str] = []

    obstacle_idx = 0
    for r in range(H):
        for c in range(W):
            if not grid[r][c]:
                continue
            obstacle_idx += 1
            body_name = f"obstacle{obstacle_idx}"
            geom_name = f"{body_name}_geom"
            obstacle_geom_names.append(geom_name)

            x, y = addJitter(cell_to_xy(r, c))
            # structure similar to example n3_dist1_w0.xml:
            # body "obstacleN" with geom "obstacleN_geom"
            obstacles_xml.append(
                f'<body name="{body_name}" pos="{x:.4f} {y:.4f} {height:.3f}">'
                f'  <geom name="{geom_name}" type="box" '
                f'size="{half_w:.4f} {half_h:.4f} {height:.3f}" '
                f'rgba="0 1 0 1"/>'
                f'</body>'
            )

    obstacles_text = ""
    if obstacles_xml:
        obstacles_text = "\n        " + "\n        ".join(obstacles_xml) + "\n"

    # Inject obstacles at <!-- OBSTACLES -->
    if "<!-- OBSTACLES -->" not in xml:
        raise RuntimeError("Comment <!-- OBSTACLES --> not found in base_layout.xml")
    xml = xml.replace("<!-- OBSTACLES -->", obstacles_text)

    # --- Agent & goal positions (grid -> world) ---
    ar, ac = agent_cell
    tr, tc = target_cell

    ax, ay = cell_to_xy(ar, ac)
    tx, ty = cell_to_xy(tr, tc)

    # Replace GOAL body block
    goal_body_new = (
        f'        <body name="goal" pos="{tx:.4f} {ty:.4f} 0">\n'
        f'            <geom name="goal_geom" type="plane" size="0.5 0.5 1" '
        f'pos="0 0 0" rgba="1 0 0 1"/>\n'
        f'        </body>'
    )

    start = xml.find('<body name="goal"')
    if start == -1:
        raise RuntimeError('Could not find <body name="goal" ...> in base_layout.xml')
    end = xml.find('</body>', start)
    if end == -1:
        raise RuntimeError('Goal </body> not found in base_layout.xml')
    end += len('</body>')
    xml = xml[:start] + goal_body_new + xml[end:]

    # Replace AGENT body block
    agent_body_new = (
        f'        <body name="agent" pos="{ax:.4f} {ay:.4f} {AGENT_HEIGHT:.3f}">\n'
        f'            <joint name="j1" type="slide" axis ="1 0 0 "/>\n'
        f'            <joint name="j2" type="slide" axis ="0 1 0 "/>\n'
        f'            <geom name="agent_geom" type="box" '
        f'size="0.25 0.25 {AGENT_HEIGHT:.3f}" rgba="1 1 0 1"/>\n'
        f'        </body>'
    )

    start = xml.find('<body name="agent"')
    if start == -1:
        raise RuntimeError('Could not find <body name="agent" ...> in base_layout.xml')
    end = xml.find('</body>', start)
    if end == -1:
        raise RuntimeError('Agent </body> not found in base_layout.xml')
    end += len('</body>')
    xml = xml[:start] + agent_body_new + xml[end:]

    # --- Sensors: distance from agent_geom to each obstacleN_geom ---
    if obstacle_geom_names:
        sensor_lines = []
        for i, geom_name in enumerate(obstacle_geom_names, start=1):
            sensor_lines.append(
                f'        <distance name="dist{i}" '
                f'geom1="agent_geom" geom2="{geom_name}" cutoff="1.5"/>'
            )
        sensors_text = "\n" + "\n".join(sensor_lines) + "\n"

        if "<sensor" in xml:
            # inject inside existing <sensor> ... </sensor>
            s_start = xml.find("<sensor")
            s_end = xml.find("</sensor>", s_start)
            if s_end == -1:
                raise RuntimeError("Malformed <sensor> block in base_layout.xml")
            insert_pos = s_end
            xml = xml[:insert_pos] + sensors_text + xml[insert_pos:]
        else:
            # no sensor block: create one before closing </mujoco>
            insert_pos = xml.rfind("</mujoco>")
            if insert_pos == -1:
                raise RuntimeError("No </mujoco> closing tag found in XML")
            sensor_block = "    <sensor>\n" + sensors_text + "    </sensor>\n"
            xml = xml[:insert_pos] + sensor_block + xml[insert_pos:]

    return xml


def write_xml_from_base(
    grid: List[List[bool]],
    agent: Coord,
    target: Coord,
    out_path: str,
) -> None:
    model_name = os.path.splitext(os.path.basename(out_path))[0]
    xml = build_xml_from_base(grid, agent, target, model_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)


# ============================
# Level 1–4: STANDARD scenarios
# ============================

def generate_standard_scenario(
    level: int,
    obstacles: int,
    distance: int,
    seed: int,
) -> ScenarioMeta:
    rng = random.Random(seed)
    target = TARGET_DEFAULT

    # Use Manhattan band instead of exact Euclidean
    band_min, band_max = distance_band_for_label(distance)

    candidates: List[Coord] = []
    tr, tc = target
    for r in range(FULL_H):
        for c in range(FULL_W):
            if (r, c) == target:
                continue
            # Manhattan distance on the grid
            d_line = abs(tr - r) + abs(tc - c)
            if at_constraints_satisfied(d_line, band_min, band_max, None):
                candidates.append((r, c))

    if not candidates:
        raise RuntimeError(
            f"No agent position with Manhattan distance in [{band_min},{band_max}] "
            f"for grid {FULL_H}x{FULL_W}"
        )

    agent = rng.choice(candidates)

    gen = BresenhamStandardGenerator(FULL_H, FULL_W)
    grid = gen.generate(
        agent=agent,
        target=target,
        exact_walls=obstacles,
        min_walls=None,
        max_walls=None,
        rng=rng,
    )

    ensure_dir(OUT_ROOT)
    filename = f"lvl{level}_standard_obs{obstacles}_d{distance}_seed{seed}.xml"
    out_path = os.path.join(OUT_ROOT, filename)

    write_xml_from_base(grid, agent, target, out_path)

    return ScenarioMeta(
        level=level,
        scenario="standard",
        obstacles=obstacles,
        distance=distance,
        seed=seed,
        depth=None,
        path=out_path,
    )

# ============================
# Level 5: DEADEND / CORRIDOR 
# ============================

def generate_structured_scenario(
    scenario: str,          # "deadend" or "corridor"
    depth: int,
    distance: int,
    seed: int,
    obstacles: int = 10,
) -> ScenarioMeta:
    assert scenario in ("deadend", "corridor")

    rng = random.Random(seed)
    use_deadend = (scenario == "deadend")
    use_corridor = (scenario == "corridor")

    # 1) symbolic block: let it use between 1 and `obstacles` walls
    max_block_walls = max(1, obstacles)
    min_block_walls = 1

    block_bool, chosen_dead, chosen_depth, \
        deadend_path_block, corridor_path_block = SymbolicScenarioGenerator.generate_grid(
            H=BLOCK_H,
            W=BLOCK_W,
            deadend=use_deadend,
            corridor=use_corridor,
            min_deadend_depth=depth,
            min_corridorLength=depth,
            corridor_endpoint_min_free_degree=3,
            z3_seed=rng.randint(0, 1_000_000),
            exact_walls=None,           # variable wall count in [min_block_walls, max_block_walls]
            min_walls=min_block_walls,
            max_walls=max_block_walls,
            spawn=(1, 1),
        )

    # agent position within block (guaranteed FREE on the path)
    if use_deadend:
        if not deadend_path_block:
            raise RuntimeError("Deadend scenario but deadend_path_block is empty.")
        agent_rel_block = deadend_path_block[-1]  # endpoint of deadend
    else:
        if not corridor_path_block:
            raise RuntimeError("Corridor scenario but corridor_path_block is empty.")
        mid_idx = len(corridor_path_block) // 2
        agent_rel_block = corridor_path_block[mid_idx]  # midpoint of corridor

    fixed_target = TARGET_DEFAULT

    # 2) offsets based on Manhattan distance band (short/mid/long)
    band_min, band_max = distance_band_for_label(distance)

    candidate_offsets = candidate_offsets_for_at(
        block_H=BLOCK_H,
        block_W=BLOCK_W,
        full_H=FULL_H,
        full_W=FULL_W,
        fixed_target=fixed_target,
        agent_rel_block=agent_rel_block,
        at_min=band_min,
        at_max=band_max,
        at_exact=None,
    )

    if not candidate_offsets:
        raise RuntimeError(
            f"No offsets for {scenario} with depth={depth}, "
            f"Manhattan distance in [{band_min},{band_max}] in {FULL_H}x{FULL_W}"
        )

    rng.shuffle(candidate_offsets)

    success = False
    grid_full: Optional[List[List[bool]]] = None
    agent_global: Optional[Coord] = None

    tr, tc = fixed_target

    for off_r, off_c in candidate_offsets:
        # Start with a fully free 15x15 grid
        grid_candidate = [[False for _ in range(FULL_W)] for _ in range(FULL_H)]

        # Embed the symbolic 5x5 block: all structural walls live here
        for r in range(BLOCK_H):
            for c in range(BLOCK_W):
                if block_bool[r][c]:
                    grid_candidate[off_r + r][off_c + c] = True

        # Force target cell FREE
        grid_candidate[tr][tc] = False

        # Agent global coords
        ag = (off_r + agent_rel_block[0], off_c + agent_rel_block[1])

        # Agent must be free
        if grid_candidate[ag[0]][ag[1]]:
            # This offset makes agent land on a wall -> try another
            continue

        # If we got here, we have a valid embedding
        grid_full = grid_candidate
        agent_global = ag
        success = True
        break

    if not success or grid_full is None or agent_global is None:
        raise RuntimeError(
            f"Could not embed {scenario} block for depth={depth}, "
            f"distance={distance}, seed={seed}."
        )

    # sanity: global wall count must not exceed requested obstacles
    wall_count = sum(
        1 for r in range(FULL_H) for c in range(FULL_W) if grid_full[r][c]
    )
    if wall_count > obstacles:
        raise RuntimeError(
            f"Wall count mismatch for {scenario}: expected <= {obstacles}, got {wall_count}"
        )

    ensure_dir(OUT_ROOT)
    filename = (
        f"lvl5_{scenario}_obs{obstacles}_d{distance}_depth{depth}_seed{seed}.xml"
    )
    out_path = os.path.join(OUT_ROOT, filename)

    write_xml_from_base(grid_full, agent_global, fixed_target, out_path)

    return ScenarioMeta(
        level=5,
        scenario=scenario,
        obstacles=obstacles,
        distance=distance,
        seed=seed,
        depth=depth,
        path=out_path,
    )



# ============================
# Master driver
# ============================

def main() -> None:
    all_meta: List[ScenarioMeta] = []

    # We'll use only short + mid distances (labels 5 and 10)
    all_dist_labels = (5, 10, 15)
    short_mid_dist_labels = (5,10)

    # Level 1: obstacles=0, ~10 scenarios
    # 2 distances * 5 seeds = 10
    print("Generating Level 1 (standard, obs=0)...")
    for dist in all_dist_labels:
        for seed in (0, 1, 2):
            all_meta.append(generate_standard_scenario(1, 0, dist, seed))

    # Level 2: obstacles=1, ~10 scenarios
    # 2 distances * 5 seeds = 10
    print("Generating Level 2 (standard, obs=1)...")
    for dist in all_dist_labels:
        for seed in (0, 1, 2):
            all_meta.append(generate_standard_scenario(2, 1, dist, seed))

    # Level 3: obstacles=(2,3,5), 1 seed
    # 3 obs * 3 distances * 1 seed = 9
    print("Generating Level 3 (standard, obs=2,3,5)...")
    for obs in (2, 3, 5):
        for dist in all_dist_labels:
            seed = 0
            all_meta.append(generate_standard_scenario(3, obs, dist, seed))

    # Level 4: obstacles=(6,8,10), 2 seeds
    # 3 obs * 2 distances * 2 seeds = 12
    print("Generating Level 4 (standard, obs=6,8,10)...")
    for obs in (6, 8, 10):
        for dist in short_mid_dist_labels:
            for seed in (0, 1):
                all_meta.append(generate_standard_scenario(4, obs, dist, seed))

    # Level 5: deadend/corridor, depth=(1,2,3),
    # distances=(short,mid)=5,10, 1 seed
    # 2 scenarios * 3 depths * 2 distances * 1 seed = 12
    print("Generating Level 5 (deadend & corridor, obs=15)...")
    for scenario in ("deadend", "corridor"):
        for depth in (2, 3, 4):
            for dist in short_mid_dist_labels:
                seed = 0
                all_meta.append(
                    generate_structured_scenario(
                        scenario=scenario,
                        depth=depth,
                        distance=dist,
                        seed=seed,
                        obstacles=15,
                    )
                )

    expected = 51  # 9 + 9 + 9 + 12 + 12
    if len(all_meta) != expected:
        raise RuntimeError(f"Expected {expected} scenarios in L1–5, got {len(all_meta)}")

    print(
        f"Done. Levels 1-5: {len(all_meta)} scenarios, "
        f"All XMLs in: {OUT_ROOT}"
    )


if __name__ == "__main__":
    main()