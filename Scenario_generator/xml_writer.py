# xml_writer.py
from __future__ import annotations

import os
import random
from typing import List, Tuple

Coord = Tuple[int, int]

# Path to base XML template
BASE_XML_PATH = os.path.join(os.path.dirname(__file__), "base_layout.xml")

# Same arena / height as your existing setup
ARENA_HALF_EXTENT = 15.0
OBSTACLE_HEIGHT = 0.25
AGENT_HEIGHT = 0.25


def build_xml_from_base(
    grid: List[List[bool]],
    agent_cell: Coord,
    target_cell: Coord,
    model_name: str,
    *,
    rng: random.Random | None = None,
) -> str:
    """
    Load base_layout.xml and inject:
    - obstacle bodies named obstacle1, obstacle2, ... with geoms obstacleN_geom
    - agent body position + size
    - goal body position
    - distance sensors from agent_geom to each obstacleN_geom
    """
    if not os.path.exists(BASE_XML_PATH):
        raise FileNotFoundError(f"BASE_XML_PATH not found: {BASE_XML_PATH}")

    if rng is None:
        # If you want full determinism, pass rng from the runner/generator.
        rng = random.Random()

    with open(BASE_XML_PATH, "r", encoding="utf-8") as f:
        xml = f.read()

    # Update model name
    xml = xml.replace('model="simple_navigation"', f'model="{model_name}"', 1)

    H = len(grid)
    W = len(grid[0])
    A = ARENA_HALF_EXTENT
    height = OBSTACLE_HEIGHT

    cell_w = (2.0 * A) / W
    cell_h = (2.0 * A) / H
    half_w = cell_w * 0.45
    half_h = cell_h * 0.45
    max_jitter_x = 0.5 * cell_w - half_w
    max_jitter_y = 0.5 * cell_h - half_h

    def cell_to_xy(r: int, c: int) -> tuple[float, float]:
        x = -A + (c + 0.5) * cell_w
        y = A - (r + 0.5) * cell_h
        return x, y

    def add_jitter(xy: tuple[float, float]) -> tuple[float, float]:
        x, y = xy
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

            x, y = add_jitter(cell_to_xy(r, c))
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

    if "<!-- OBSTACLES -->" not in xml:
        raise RuntimeError("Comment <!-- OBSTACLES --> not found in base_layout.xml")
    xml = xml.replace("<!-- OBSTACLES -->", obstacles_text)

    # --- Agent & goal positions ---
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
    end = xml.find("</body>", start)
    if end == -1:
        raise RuntimeError('Goal </body> not found in base_layout.xml')
    end += len("</body>")
    xml = xml[:start] + goal_body_new + xml[end:]

    # Replace AGENT body block
    agent_body_new = (
        f'        <body name="agent" pos="{ax:.4f} {ay:.4f} {AGENT_HEIGHT:.3f}">\n'
        f'            <joint name="j1" type="slide" axis ="1 0 0 "/>\n'
        f'            <joint name="j2" type="slide" axis ="0 1 0 "/>\n'
        f'            <geom name="agent_geom" type="cylinder" '
        f'size="0.25 0.25 {AGENT_HEIGHT:.3f}" rgba="1 1 0 1"/>\n'
        f'        </body>'
    )
    start = xml.find('<body name="agent"')
    if start == -1:
        raise RuntimeError('Could not find <body name="agent" ...> in base_layout.xml')
    end = xml.find("</body>", start)
    if end == -1:
        raise RuntimeError('Agent </body> not found in base_layout.xml')
    end += len("</body>")
    xml = xml[:start] + agent_body_new + xml[end:]

    # --- Sensors ---
    if obstacle_geom_names:
        sensor_lines = []
        for i, geom_name in enumerate(obstacle_geom_names, start=1):
            sensor_lines.append(
                f'        <distance name="dist{i}" '
                f'geom1="agent_geom" geom2="{geom_name}" cutoff="1.5"/>'
            )
        sensors_text = "\n" + "\n".join(sensor_lines) + "\n"

        if "<sensor" in xml:
            s_start = xml.find("<sensor")
            s_end = xml.find("</sensor>", s_start)
            if s_end == -1:
                raise RuntimeError("Malformed <sensor> block in base_layout.xml")
            xml = xml[:s_end] + sensors_text + xml[s_end:]
        else:
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
    *,
    rng: random.Random | None = None,
) -> None:
    model_name = os.path.splitext(os.path.basename(out_path))[0]
    xml = build_xml_from_base(grid, agent, target, model_name, rng=rng)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
