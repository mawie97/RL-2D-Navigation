import os
import random
import shutil
import xml.etree.ElementTree as ET
import numpy as np

# =========================
# Config
# =========================
SEED = 123
random.seed(SEED)

BASE_TEMPLATE_PATH = "../../layouts/base/base_layout.xml"
OUTPUT_DIR = "../../layouts/eval/"

GOAL_NAME = "goal"
AGENT_NAME = "agent"

OBSTACLE_SIZE = (0.3, 0.3, 0.3)
SENSOR_CUTOFF = 1.5

GRID_CELL_SIZE = 2.0
GRID_WIDTH = 20
GRID_HEIGHT = 20
GRID_X_START = -20.0
GRID_Y_START = -20.0

# =========================
# Utility Functions
# =========================
def clear_output_dir(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Failed to delete {path}. Reason: {e}")
    else:
        os.makedirs(directory, exist_ok=True)

def grid_index_to_position(ix, iy):
    x = GRID_X_START + (ix + 0.5) * GRID_CELL_SIZE
    y = GRID_Y_START + (iy + 0.5) * GRID_CELL_SIZE
    return x, y

def grid_index_to_position_with_jitter(ix, iy, jitter=0.5):
    cx, cy = grid_index_to_position(ix, iy)
    return cx + random.uniform(-jitter, jitter), cy + random.uniform(-jitter, jitter)

def position_to_grid_index(x, y):
    ix = int((x - GRID_X_START) // GRID_CELL_SIZE)
    iy = int((y - GRID_Y_START) // GRID_CELL_SIZE)
    if 0 <= ix < GRID_WIDTH and 0 <= iy < GRID_HEIGHT:
        return ix, iy
    raise ValueError("Position out of grid bounds")

def cells_at_manhattan_distance(x, y, d):
    candidates = []
    for dx in range(-d, d + 1):
        dy = d - abs(dx)
        for ny in (y + dy, y - dy):
            nx = x + dx
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                candidates.append((nx, ny))
    return list(set(candidates))

def get_agent_goal_grid_positions(xml_text):
    root = ET.fromstring(xml_text)
    agent_pos = None
    goal_pos = None
    for body in root.findall(".//body"):
        name = body.get("name", "")
        pos = body.get("pos", "")
        if not pos:
            continue
        pos = tuple(map(float, pos.strip().split()))
        if name == AGENT_NAME:
            agent_pos = pos
        elif name == GOAL_NAME:
            goal_pos = pos
        if agent_pos and goal_pos:
            break
    if agent_pos is None or goal_pos is None:
        raise ValueError("Agent or Goal position not found")
    agent_cell = position_to_grid_index(agent_pos[0], agent_pos[1])
    goal_cell = position_to_grid_index(goal_pos[0], goal_pos[1])
    return agent_cell, goal_cell

def get_occupied_cells(pos, size):
    """Get set of grid cells occupied by an object of given size."""
    half_w, half_h = size[0], size[1]
    x_min, x_max = pos[0] - half_w, pos[0] + half_w
    y_min, y_max = pos[1] - half_h, pos[1] + half_h
    ix_min = int((x_min - GRID_X_START) / GRID_CELL_SIZE)
    ix_max = int((x_max - GRID_X_START) / GRID_CELL_SIZE)
    iy_min = int((y_min - GRID_Y_START) / GRID_CELL_SIZE)
    iy_max = int((y_max - GRID_Y_START) / GRID_CELL_SIZE)
    return set((ix, iy) for ix in range(ix_min, ix_max + 1)
            for iy in range(iy_min, iy_max + 1))

def get_agent_goal_info(xml_text):
    root = ET.fromstring(xml_text)

    agent_pos = None
    goal_pos = None

    for body in root.findall(".//body"):
        name = body.get("name", "")
        pos = body.get("pos", "")
        if not pos:
            continue
        pos = tuple(map(float, pos.strip().split()))

        # Look for geom inside this body to get size
        geom = body.find(".//geom")
        size_str = geom.get("size") if geom is not None else ""
        size = tuple(map(float, size_str.strip().split())) if size_str else None

        if name == AGENT_NAME and size is not None:
            agent_pos = pos
            agent_size = size
        elif name == GOAL_NAME and size is not None:
            goal_pos = pos
            goal_size = size

        if agent_pos and goal_pos and agent_size and goal_size:
            break

    if agent_pos is None or goal_pos is None or agent_size is None or goal_size is None:
        raise ValueError("Could not find agent or goal position or size in XML")

    return agent_pos, agent_size, goal_pos, goal_size

def neighbors_4(cell):
    x, y = cell
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def neighbors_2_layers(cell):
    first_layer = neighbors_4(cell)
    second_layer = []
    for c in first_layer:
        second_layer.extend(neighbors_4(c))
    return set(first_layer + second_layer)

def rectangle_cells(c1, c2):
    x_min = min(c1[0], c2[0])
    x_max = max(c1[0], c2[0])
    y_min = min(c1[1], c2[1])
    y_max = max(c1[1], c2[1])
    
    rectangle = {(x, y) for x in range(x_min, x_max+1) for y in range(y_min, y_max+1)}
    corners = {(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)}

    exclude_corners = corners - {c1, c2}
    
    exclude_cells = set()
    for corner in exclude_corners:
        # Add the corner cell itself
        exclude_cells.add(corner)
        
        # Add neighbors up to 2 layers, clipped inside rectangle
        neighbors = neighbors_4(corner)
        neighbors = {n for n in neighbors if n in rectangle}
        exclude_cells.update(neighbors)
    
    allowed_cells = rectangle - exclude_cells
    print (f".  Exclude corners: {exclude_corners}, Exclude cells: {exclude_cells}, Allowed cells: {allowed_cells}")
    print(" ")
    return allowed_cells


def bresenham_line(x0, y0, x1, y1):
    """Return list of grid cells on a Bresenham line from (x0,y0) to (x1,y1)."""
    cells = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            cells.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        while y != y1:
            cells.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    cells.append((x1, y1))
    return cells

def corridor_around_bresenham_line(x0, y0, x1, y1, width):
    line_cells = bresenham_line(x0, y0, x1, y1)
    # print(f"Bresenham line cells: {line_cells}")

    corridor = set(line_cells)

    # For each cell in line, find direction vector to next cell (or previous if last)
    for i, (cx, cy) in enumerate(line_cells):
        if i < len(line_cells) - 1:
            nx, ny = line_cells[i + 1]
            dx, dy = nx - cx, ny - cy
        elif i > 0:
            px, py = line_cells[i - 1]
            dx, dy = cx - px, cy - py
        else:
            dx, dy = 0, 0  # single point line, no direction

        # Get perpendicular directions
        perp_dirs = [(-dy, dx), (dy, -dx)]

        for (pdx, pdy) in perp_dirs:
            for dist in range(1, width + 1):
                neighbor = (cx + pdx * dist, cy + pdy * dist)
                if 0 <= neighbor[0] < GRID_WIDTH and 0 <= neighbor[1] < GRID_HEIGHT:
                    corridor.add(neighbor)

    return corridor


def move_obstacles_with_manhattan_distance(obstacle_cells, manhattan_dist, forbidden_cells, rectangle):
    new_positions = []
    for (x, y) in obstacle_cells:
        candidates = cells_at_manhattan_distance(x, y, manhattan_dist)
        candidates = [c for c in candidates if c in rectangle and c not in forbidden_cells]

        if not candidates:
            # Fallback: if current pos is outside rectangle, try to snap inside rectangle
            if (x, y) in rectangle:
                new_positions.append((x, y))
                forbidden_cells.add((x, y))
            else:
                # Find closest cell inside rectangle to (x, y)
                closest_cell = min(rectangle, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
                new_positions.append(closest_cell)
                forbidden_cells.add(closest_cell)
            continue

        new_pos = random.choice(candidates)
        new_positions.append(new_pos)
        forbidden_cells.add(new_pos)
    return new_positions

def generate_obstacle(i, pos):
    return f'''
        <body name="obstacle{i}" pos="{pos[0]} {pos[1]} 0.5">
            <geom name="obstacle{i}_geom" type="box" size="{OBSTACLE_SIZE[0]} {OBSTACLE_SIZE[1]} {OBSTACLE_SIZE[2]}" rgba="0 1 0 1"/>
        </body>
    '''

def generate_sensor(i):
    return f'        <distance name="dist{i}" geom1="agent_geom" geom2="obstacle{i}_geom" cutoff="{SENSOR_CUTOFF}"/>'


def insert_obstacles_and_sensors(xml_text, obstacle_positions):
    obstacles_xml = "".join(generate_obstacle(i + 1, pos) for i, pos in enumerate(obstacle_positions))
    sensors_xml = "\n".join(generate_sensor(i + 1) for i in range(len(obstacle_positions)))
    return xml_text.replace("<!-- OBSTACLES -->", obstacles_xml).replace("<!-- SENSORS -->", sensors_xml)

# =========================
# Main variant generator
# =========================
def generate_variants(seed=None):
    random.seed(seed)
    with open(BASE_TEMPLATE_PATH) as f:
        base_xml = f.read()

    agent_cell, goal_cell = get_agent_goal_grid_positions(base_xml)
    rectangle = set(rectangle_cells(agent_cell, goal_cell))
    print(f"Agent cell: {agent_cell}, Goal cell: {goal_cell}")
    print(f"rectangle: {rectangle}")
    agent_pos, agent_size, goal_pos, goal_size = get_agent_goal_info(base_xml)
    forbidden = get_occupied_cells(agent_pos, agent_size) | get_occupied_cells(goal_pos, goal_size)

    # obstacle_setups: n_obstacles: (list of manhattan distances, corridor_width)
    # This is for training setups
    obstacle_setups = {
        0: ([0], 0),
        1: ([1, 2, 3], 0),  # Only one obstacle in bresenham line
        3: ([3, 6, 9], 1),
        5: ([3, 6, 9], 1),
        7: ([3, 6, 9], 0),
        9: ([3, 6, 9], 0),
    }
    
    # This is for evaluation setup
    obstacle_setups = {
        0: ([0], 0),
        2: ([1], 1),
        4: ([3], 1),
        6: ([3], 0),
        8: ([3], 0),
        10: ([3], 0),
    }
    
    # obstacle_setups = {
    #     7: ([4, 5, 6], 0),
    # }
    
    for n_obstacles, (displacements, corridor_width) in obstacle_setups.items():
        for dist in displacements:
            # variant_name = f"n{n_obstacles}_dist{dist}_w{corridor_width}"
            
            # For evaluation layouts
            variant_name = f"eval_n{n_obstacles}_dist{dist}_w{corridor_width}"
            print(f"Generating variant {variant_name}")

            forbidden_cells = set(forbidden)
            allowed_cells = rectangle
            
            if corridor_width > 0:
                corridor_cells = corridor_around_bresenham_line(agent_cell[0], agent_cell[1], goal_cell[0], goal_cell[1], corridor_width)
                allowed_cells = corridor_cells.intersection(rectangle)
                print(" ")
                print(f"Allowed cells after corridor {corridor_width}:  {allowed_cells}")
                print(f"For {n_obstacles} dist {dist} corridor width {corridor_width}, allowed cells: {len(allowed_cells)}")
                print("    ")
            else:
                allowed_cells = rectangle
                # print(f"For {n_obstacles} dist {dist} corridor width {corridor_width}, allowed cells: {len(allowed_cells)}")

            if n_obstacles == 0:
                obstacle_positions = []
            elif n_obstacles == 1:
                line_cells = bresenham_line(agent_cell[0], agent_cell[1], goal_cell[0], goal_cell[1])
                # print (f"Bresenham line cells: {line_cells}")
                # Filter line cells to allowed_cells and not forbidden
                line_cells = [c for c in line_cells if c in allowed_cells and c not in forbidden_cells]
                if not line_cells:
                    raise RuntimeError("No allowed cells on Bresenham line free for obstacle")
                chosen_cell = random.choice(line_cells)
                if dist > 0:
                    moved_cells = move_obstacles_with_manhattan_distance([chosen_cell], dist, forbidden_cells, line_cells)
                    chosen_cell = moved_cells[0]
                obstacle_positions = [chosen_cell]
                forbidden_cells.add(chosen_cell)
                # print (f"Chosen obstacle position: {chosen_cell}")
            elif n_obstacles == 2:
                line_cells = bresenham_line(agent_cell[0], agent_cell[1], goal_cell[0], goal_cell[1])
                # print (f"Bresenham line cells: {line_cells}")
                # Filter line cells to allowed_cells and not forbidden
                line_cells = [c for c in line_cells if c in allowed_cells and c not in forbidden_cells]
                
                if len(line_cells) < 2:
                    raise RuntimeError("Not enough allowed cells on Bresenham line for 2 obstacles")
                chosen_cells = random.sample(line_cells, 2)
                if not line_cells:
                    raise RuntimeError("No allowed cells on Bresenham line free for obstacle")
                chosen_cell = random.choice(line_cells)
                if dist > 0:
                    moved_cells = move_obstacles_with_manhattan_distance([chosen_cell], dist, forbidden_cells, line_cells)
                    chosen_cell = moved_cells[0]
                obstacle_positions = chosen_cells
                forbidden_cells.update(chosen_cells)
                # print (f"Chosen obstacle position: {chosen_cell}")
                
            elif n_obstacles == 3:
                line_cells = bresenham_line(agent_cell[0], agent_cell[1], goal_cell[0], goal_cell[1])
                # print(f"Bresenham line cells: {line_cells}")
                
                allowed_line_cells = [c for c in line_cells if c in allowed_cells and c not in forbidden_cells]
                if not allowed_line_cells:
                    raise RuntimeError("No allowed cells on Bresenham line free for obstacle")

                path_obstacle = random.choice(allowed_line_cells)
                forbidden_cells.add(path_obstacle)
                
                remaining_allowed = list(allowed_cells - forbidden_cells)
                if len(remaining_allowed) < 2:
                    raise RuntimeError("Not enough allowed cells to place remaining obstacles")
                
                remaining_obstacles = random.sample(remaining_allowed, 2)
                forbidden_cells.update(remaining_obstacles)
                
                all_obstacles = [path_obstacle] + remaining_obstacles
                
                if dist > 0:
                    moved_obstacles = move_obstacles_with_manhattan_distance(all_obstacles, dist, forbidden_cells, allowed_cells)
                    obstacle_positions = moved_obstacles
                else:
                    obstacle_positions = all_obstacles
                
                print(f"Chosen obstacle positions: {obstacle_positions}")

            else:
                min_in_allowed = n_obstacles
                allowed_available = list(allowed_cells - forbidden_cells)
                if len(allowed_available) < min_in_allowed:
                    raise RuntimeError("Not enough allowed cells to place obstacles")
                allowed_obstacles = random.sample(allowed_available, min_in_allowed)
                forbidden_cells.update(allowed_obstacles)

                remaining = n_obstacles - min_in_allowed
                rectangle_available = list(rectangle - forbidden_cells)   # If don't have enough allowed cells, use rectangle
                if len(rectangle_available) < remaining:
                    raise RuntimeError("Not enough rectangle cells to place remaining obstacles")
                remaining_obstacles = random.sample(rectangle_available, remaining)
                forbidden_cells.update(remaining_obstacles)

                all_obstacles = allowed_obstacles + remaining_obstacles
                
                total_dist = dist  # the total sum of distances to move all obstacles
                num_obs = len(all_obstacles)

                # Randomly split total_dist into num_obs parts (simple method: random proportions)
                proportions = np.random.dirichlet(np.ones(num_obs), size=1)[0]
                distances = [int(round(total_dist * p)) for p in proportions]

                moved_obstacles = []
                leftover = 0  # leftover distance to redistribute

                distances_copy = distances.copy()

                for idx, (x, y) in enumerate(all_obstacles):
                    move_dist = distances_copy[idx] + leftover
                    leftover = 0  # reset leftover before this obstacle's move

                    if move_dist == 0:
                        chosen = (x, y)
                    else:
                        candidates = cells_at_manhattan_distance(x, y, move_dist)
                        candidates = [c for c in candidates if c in allowed_available and c not in forbidden_cells]

                        if not candidates:
                            # Can't move full move_dist, try smaller distances down to 0
                            for d in range(move_dist - 1, -1, -1):
                                candidates = cells_at_manhattan_distance(x, y, d)
                                candidates = [c for c in candidates if c in allowed_available and c not in forbidden_cells]
                                if candidates:
                                    chosen = random.choice(candidates)
                                    leftover = move_dist - d  # leftover distance to redistribute
                                    break
                            else:
                                # No candidates even at distance 0 (stay put)
                                chosen = (x, y)
                                leftover = move_dist  # all distance leftover
                        else:
                            chosen = random.choice(candidates)
                            leftover = 0

                    moved_obstacles.append(chosen)
                    forbidden_cells.add(chosen)
                
                obstacle_positions = moved_obstacles

            obstacle_positions_world = [grid_index_to_position(x, y) for x, y in obstacle_positions]

            xml_variant = insert_obstacles_and_sensors(base_xml, obstacle_positions_world)

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            variant_path = os.path.join(OUTPUT_DIR, f"{variant_name}.xml")
            with open(variant_path, "w") as f:
                f.write(xml_variant)

            print(f"Saved variant {variant_path}")


if __name__ == "__main__":
    clear_output_dir(OUTPUT_DIR)
    generate_variants(seed=SEED)
