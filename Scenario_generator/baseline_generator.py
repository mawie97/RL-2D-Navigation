import os
import math
import random
from typing import List, Tuple

X_MIN, X_MAX = -19.2, 19.2
Y_MIN, Y_MAX = -19.2, 19.2

AGENT_GOAL_MIN_DIST = 2.0        # avoid trivial start on top of goal
OBST_MIN_DIST_AGENT_GOAL = 2   # avoid put obstacle on the agent and the goal
OBST_MIN_DIST_BETWEEN = 2      # avoid obstacles overlap

OBST_SIZE_X = 0.8
OBST_SIZE_Y = 0.8
OBST_SIZE_Z = 0.25

GOAL_POS: Tuple[float, float] = (12.0, -12.0)

OUTPUT_ROOT = "layouts_baseline"


def dist2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def sample_point() -> Tuple[float, float]:
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    return x, y


def sample_agent_and_goal() -> Tuple[Tuple[float, float], Tuple[float, float]]:
    while True:
        agent = sample_point()
        goal = GOAL_POS
        if dist2d(agent, goal) >= AGENT_GOAL_MIN_DIST:
            return agent, goal


# Sample obstacle centers randomly, avoiding overlaps with agent, goal and each other.
def sample_obstacles( 
    n_obstacles: int,  
    agent: Tuple[float, float],
    goal: Tuple[float, float],  
) -> List[Tuple[float, float]]:

    obstacles: List[Tuple[float, float]] = []
    max_tries = 2000 

    tries = 0
    while len(obstacles) < n_obstacles and tries < max_tries:
        tries += 1
        candidate = sample_point()

        if dist2d(candidate, agent) < OBST_MIN_DIST_AGENT_GOAL:
            continue
        if dist2d(candidate, goal) < OBST_MIN_DIST_AGENT_GOAL:
            continue
 
        if any(dist2d(candidate, o) < OBST_MIN_DIST_BETWEEN for o in obstacles):
            continue

        obstacles.append(candidate)

    if len(obstacles) < n_obstacles:
        print(f" Could only place {len(obstacles)}/{n_obstacles} obstacles")
    return obstacles


def build_xml(
    agent: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Tuple[float, float]],
    model_name: str,
) -> str:
    agent_x, agent_y = agent
    goal_x, goal_y = goal

    header = f"""<mujoco model="{model_name}">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option integrator="RK4" timestep="0.01"/>

    <asset>
        <texture builtin="gradient" height="100" rgb1="0.6 0.8 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture name="floor_texture" type="2d" builtin="checker" width="100" height="100" rgb1="0 0 0" rgb2="0.8 0.8 0.8"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="10 10" texture="floor_texture"/>
    </asset>
    
    <!-- Excluding the contact between the agent and the goal, and the agent and the floor -->
    <contact>
        <exclude body1="agent" body2="goal"/>
        <exclude body1 = "agent" body2 = "floor"/>
    </contact>

    <extension>
        <plugin plugin = "mujoco.pid">
            <instance name = "pid1">
                <config key="kp" value="2"/>
                <config key="ki" value="0"/>
                <config key="kd" value="55"/>
            </instance>
        </plugin>
    </extension>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <body name="floor" pos="0 0 0">
            <geom name="floor_geom" type="plane" size="20 20 0.1" material="MatPlane"/>
        </body>
        
        <!-- Goal -->
        <body name="goal" pos="{goal_x:.3f} {goal_y:.3f} 0">
            <geom name="goal_geom" type="plane" size="0.5 0.5 1" pos="0 0 0" rgba="1 0 0 1"/>
        </body>

        <!-- Agent -->
        <body name="agent" pos="{agent_x:.3f} {agent_y:.3f} 0.25">
            <joint name="j1" type="slide" axis ="1 0 0 "/>
            <joint name="j2" type="slide" axis ="0 1 0 "/>
            <geom name="agent_geom" type="box" size="0.25 0.25 0.250" rgba="1 1 0 1"/>
        </body>

        
"""

    # Obstacles
    obstacle_xml_lines = []
    sensor_xml_lines = []
    for i, (ox, oy) in enumerate(obstacles, start=1):
        obstacle_xml_lines.append(
            f"""    <body name="obstacle{i}" pos="{ox:.3f} {oy:.3f} {OBST_SIZE_Z:.3f}">
      <geom name="obstacle{i}_geom" type="box"
            size="{OBST_SIZE_X:.3f} {OBST_SIZE_Y:.3f} {OBST_SIZE_Z:.3f}"
            rgba="0 1 0 1"/>
    </body>"""
        )
    obstacles_block = "\n".join(obstacle_xml_lines)

    worldbody_close = """
    </worldbody>
    """

    actuator_block = """
    <actuator>
        <plugin joint="j1" plugin="mujoco.pid" instance="pid1" />
        <plugin joint="j2" plugin="mujoco.pid" instance="pid1"/>
    </actuator>
"""

    sensor_lines = []
    for i in range(1, len(obstacles) + 1):
        sensor_lines.append(
            f'    <distance name="dist{i}" geom1="agent_geom" geom2="obstacle{i}_geom" cutoff="1.5"/>'
        )
    if sensor_lines:
        sensor_block = "\n  <sensor>\n" + "\n".join(sensor_lines) + "\n  </sensor>\n"
    else:
        sensor_block = ""  # Level 1: no obstacles, no sensors

    footer = "</mujoco>\n"

    return header + obstacles_block + worldbody_close + actuator_block + sensor_block + footer

def generate_level(
    level_id: int,
    out_dir: str,
    obstacle_pattern: List[Tuple[int, int]],
    start_seed: int = 0,
):

    os.makedirs(out_dir, exist_ok=True)

    file_counter = 0
    for n_obs, count in obstacle_pattern:
        for i in range(count):
            seed = start_seed + file_counter
            random.seed(seed)

            agent, goal = sample_agent_and_goal()
            obstacles = sample_obstacles(n_obs, agent, goal)

            model_name = f"baseline_lvl{level_id}_obs{n_obs}_seed{seed}"
            xml_str = build_xml(agent, goal, obstacles, model_name=model_name)

            filename = f"baseline_lvl{level_id}_obs{n_obs}_seed{seed}.xml"
            path = os.path.join(out_dir, filename)
            with open(path, "w") as f:
                f.write(xml_str)

            print(f"[INFO] Wrote {path}")
            file_counter += 1


def generate_all_levels(output_root: str = OUTPUT_ROOT):

    # Level 1: 9 XMLs, 0 obstacles
    generate_level(
        level_id=1,
        out_dir=os.path.join(output_root, "L1"),
        obstacle_pattern=[(0, 9)],
        start_seed=0,
    )

    # Level 2: 9 XMLs, 1 obstacle
    generate_level(
        level_id=2,
        out_dir=os.path.join(output_root, "L2"),
        obstacle_pattern=[(1, 9)],
        start_seed=1000,
    )

    # Level 3: 9 XMLs total: 3 with 2 obs, 3 with 3 obs, 3 with 5 obs
    generate_level(
        level_id=3,
        out_dir=os.path.join(output_root, "L3"),
        obstacle_pattern=[(2, 3), (3, 3), (5, 3)],
        start_seed=2000,
    )

    # Level 4: 12 XMLs total: 4 with 6 obs, 4 with 8 obs, 4 with 10 obs
    generate_level(
        level_id=4,
        out_dir=os.path.join(output_root, "L4"),
        obstacle_pattern=[(6, 4), (8, 4), (10, 4)],
        start_seed=3000,
    )

    # Level 5 baseline: high-density random clutter (no structured deadends/corridors)
    generate_level(
        level_id=5,
        out_dir=os.path.join(output_root, "L5"),
        obstacle_pattern=[(10, 4), (10, 4), (10, 4)],
        start_seed=4000,
    )


if __name__ == "__main__":
    generate_all_levels()
