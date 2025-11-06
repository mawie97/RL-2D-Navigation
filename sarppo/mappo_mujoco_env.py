
# This is a MuJoCo‑based multi‑agent environment. It returns per‑agent 
# observations and actions and also provides a “global state” 
# in the info dictionary that concatenates all agents’ observations.

import csv
import random
from collections import deque
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
import mujoco
import mujoco_viewer
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# PYTHONWARNINGS="ignore::DeprecationWarning" python sarppo/train_rllib_mappo.py  

SWITCH_EVERY = 10
POSITION_HISTORY_LEN = 20
CUTOFF_VALUE = 1 #sensor cutoff value - sensor range
MAX_STEPS = 500 #terminate episode if max_steps is reached
MAX_ACTION = 0.1 #action space
AGENT_PREFIX = "agent_"
OBS_DIM = 16
ACT_DIM = 2
N_RAYS = 12
RAY_LENGTH = 1.5
NOISE_STD = 0.00
TARGET_NR = 3
GRID_BOUNDS = (-2.5, 2.5, -2.5, 2.5) # grid representation bounds

class MultiAgentSAR(MultiAgentEnv):

    def __init__(self, csv_log_path, xml_paths, render_enabled = True, seed = None):
        super().__init__()
        
        self.xml_paths = xml_paths
        self.num_xmls = len(self.xml_paths)
        self.switch_every = SWITCH_EVERY
        self.episode_count = 0     
        self.current_xml_index = 0

        self.viewer = None
        self._seed = None
        self.render_enabled = bool(render_enabled)

        self.position_history = deque(maxlen=POSITION_HISTORY_LEN)
        
        self.cutoff_value = CUTOFF_VALUE
        self.max_steps = MAX_STEPS
        self.csv_log_path = csv_log_path
        # print(f"Loaded XML paths: {self.xml_paths}")
        # print(" ")

        # coverage grid
        self.grid_bounds = GRID_BOUNDS
        self.cell_size = 1.0
        xmin, xmax, ymin, ymax = GRID_BOUNDS
        self.grid_W = int(np.ceil((xmax - xmin) / self.cell_size))
        self.grid_H = int(np.ceil((ymax - ymin) / self.cell_size))
        self.visited = np.zeros((self.grid_H, self.grid_W), dtype=np.uint8)
        
        # Add grid coverage to the critic
        self.cov_dim = self.grid_H * self.grid_W
        self.extra_dim = 3
         
        self.reach_tol = 0.001 #reach tolerance if the agent reah the world target
        self.max_substeps_per_action = 50000
        self.steps_since_discovery = 0 

        #This is used to save the world target for each agent at each timestep
        self._step_targets = {}
        
        self._load_model_and_setup(self.current_xml_index)
        self._reset_episode_state()   

        self.found_targets:set[int] = set()
        self.per_agent_new_cells = {name: 0 for name in self.agent_names}
        self.last_ray_hits: Dict[str, set[int]] = {name : set() for name in self.agent_names}
        self.target_nr = TARGET_NR

        self.debug_dump = bool(render_enabled)  # or from env_config["debug_dump"]
        self._printed_reset = False
        self._printed_step  = False
        self._recent_pos = {name: deque(maxlen=10) for name in self.agent_names}

        
        if self.csv_log_path is not None:
            self.log_file_handle = open(self.csv_log_path, mode='w', newline='')
            self.log_writer = csv.writer(self.log_file_handle)
            # self.log_writer.writerow(['Step', 'pre_x', 'pre_y', 'current_x', 'current_y', 'target_x','target_y', 'delta_x', 'delta_y', 'pre_distance','current_distance','dist_value','RE: dis_change', 'RE: distance', 'RE: obstacle_avoidance', 'Total: reward' , 'Status'])
            self.log_writer.writerow(['Episode', 'Status'])
            
            # step-level log for debug, with ".steps.csv"
            steps_path = (self.csv_log_path[:-4] + ".steps.csv") if self.csv_log_path.endswith(".csv") \
                         else (self.csv_log_path + ".steps.csv")
            self.step_log_path = steps_path
            self.step_log_handle = open(self.step_log_path, mode='w', newline='')
            self.step_log_writer = csv.writer(self.step_log_handle)
            self.step_log_writer.writerow([
                'episode','step','agent',
                # 'prev_x','prev_y','cur_x','cur_y','tgt_x','tgt_y',
                # 'delta_x','delta_y',
                'newly_marked','coverage_total',
                'team_reward_share', 'target_total','coverage_bonus','time_penalty',
                'proximity_penalty','collision_penalty', 'obsDistance_penalty', 'idle_penalty','reward_total',
                'done_reason'
            ])
        else:
            self.step_log_path = None
            self.step_log_handle = None
            self.step_log_writer = None


#  ================= Helper function used in the _init_  =================

    def _load_model_and_setup(self, index):
        self._load_model(index)
        self._get_agents_from_model()
        self.agent_map = self._build_agent_map()
        self._setup_agents_ray_casting()
        self._setup_action_observation_spaces()
        self._last_ray_surface = {name: np.zeros(self.n_rays, dtype=np.float32)
                              for name in self.agent_names}

    def _load_model(self, index):
        self.model = mujoco.MjModel.from_xml_path(self.xml_paths[index])
        self.data = mujoco.MjData(self.model)

    def _get_agents_from_model(self):
        self.agent_names = []
        self.agent_prefix = AGENT_PREFIX
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith(self.agent_prefix): 
                self.agent_names.append(name)

        self.agent_names = sorted(self.agent_names)
        self.n_agents = len(self.agent_names)
    
    def _build_agent_map(self):
        agent_map = {}

        for agent in self.agent_names:
            # Actuator indices
            ids = []

            # Agent's joints
            for j in [1, 2]:
                joint_name = f"{agent}_j{j}"
                # print(joint_name)
                j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                # print("AGENT MAP "+  act_name + str(a_id))
                ids.append(j_id)
            ids = np.array(ids, dtype=np.int32)

            # Original body position (xy)
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, agent)
            pos = self.model.body_pos[body_id][:2].copy()  # only x, y

            agent_map[agent] = {"ctrl_ids": ids, "init_pos": pos, "body_id": body_id}
        
        # print("=== Agent map ===")
        
        for name, m in agent_map.items():
            ctrl_ids = m["ctrl_ids"].tolist()
            init_pos = m["init_pos"].tolist()
            body_id  = int(m["body_id"])
            # print(f"{name}: body_id={body_id}, ctrl_ids={ctrl_ids}, init_pos={init_pos}")
        
        return agent_map
        
    def _setup_agents_ray_casting(self):
        self.n_rays = N_RAYS
        self.ray_length = RAY_LENGTH
        self.ray_directions = self._compute_ray_directions(self.n_rays)
        self.ray_geomid_out = np.zeros(self.n_rays, dtype=np.int32)
        self.ray_dist_out = np.zeros(self.n_rays, dtype=np.float64)

    def _compute_ray_directions(self, n_rays):
        ray_directions = []
        for i in range(n_rays):
            angle_rad = np.deg2rad(i * 360 / n_rays)
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
            dz = 0
            
            # dx *= self.ray_length
            # dy *= self.ray_length

            ray_directions.append([dx, dy, dz])

        ray_directions = np.array(ray_directions, dtype=np.float64).reshape(-1)
        return ray_directions
    
    def _setup_action_observation_spaces(self):
        f32 = np.float32
        cutoff = f32(self.cutoff_value)

        self.max = MAX_ACTION        
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM
        
        self.global_dim = self.obs_dim * self.n_agents
        # ADD: coverage embedding size
        self.state_dim = self.global_dim + self.cov_dim + self.extra_dim
        
        # ---- Build bounds for ACTOR obs (length 16 = 12 rays + dx + dy + 2 indicators) ----
        # rays in [0, cutoff] (12)
        obs_low  = np.concatenate([np.zeros(12, dtype=f32),
                                np.full(2, -5.0, dtype=f32),   # (dx,dy) ~ arena-sized
                                np.zeros(2, dtype=f32)])       # indicators in {0,1}
        obs_high = np.concatenate([np.full(12, cutoff, dtype=f32),
                                np.full(2,  5.0, dtype=f32),   # (dx,dy)
                                np.ones(2, dtype=f32)])
        
        # ---- Build bounds for CRITIC state: [global (2*obs_dim) | coverage(=0/1) | extras(0..1)] ----
        # global = concat of locals for both agents; reuse same per-feature bounds twice
        global_low  = np.tile(obs_low,  self.n_agents)
        global_high = np.tile(obs_high, self.n_agents)

        cov_low   = np.zeros(self.cov_dim, dtype=f32) # 25
        cov_high  = np.ones(self.cov_dim,  dtype=f32)
        extra_low = np.zeros(self.extra_dim, dtype=f32)   # for extra 3 t_norm, found_norm, total_norm ∈ [0,1]
        extra_high= np.ones(self.extra_dim,  dtype=f32)

        gs_low  = np.concatenate([global_low, cov_low,  extra_low]).astype(f32)
        gs_high = np.concatenate([global_high,cov_high, extra_high]).astype(f32)

        self.observation_space = gym.spaces.Dict({
            "obs":   gym.spaces.Box(low=obs_low, high=obs_high, dtype=f32),
            "state": gym.spaces.Box(low=gs_low,  high=gs_high,  dtype=f32),
        })
      
        # One agent's action space
        a_low  = f32(-self.max)
        a_high = f32(self.max)
        self.action_space = gym.spaces.Box(low=a_low, high=a_high, shape=(self.act_dim,), dtype=f32)
        
        # --- sanity checks ---
        decl_local = self.observation_space["obs"].shape[0]
        decl_state = self.observation_space["state"].shape[0]
        assert decl_local == self.obs_dim, f"obs_dim mismatch: {decl_local} vs {self.obs_dim}"
        assert decl_state == self.state_dim, f"state_dim mismatch: {decl_state} vs {self.state_dim}"

        # Helpful prints (once)
        print(" ")
        print(f"[ENV/SPACES] n_agents={self.n_agents} obs_dim={self.obs_dim} "
            f"global_dim={self.global_dim} cov_dim={self.cov_dim} extra_dim={self.extra_dim} "
            f"state_dim={self.state_dim}")


    #  ================= Helper function used in the reset()  =================
    def _switch_model(self):
        self.current_xml_index = (self.current_xml_index + 1) % self.num_xmls
        self._load_model_and_setup(self.current_xml_index)

    def _reset_episode_state(self):
        self._step_count = 0
        self.position_history.clear()
        self.found_targets = set()
        self.steps_since_discovery = 0
        self.visited.fill(0)
        self.last_ray_hits = {name: set() for name in self.agent_names}
        for name in self.agent_names:
           self._last_ray_surface[name].fill(0.0)
        self.per_agent_new_cells = {name: 0 for name in self.agent_names}

    def _get_local_obs_dict(self) -> dict[str, np.ndarray]:
        """
        Return local observations for all agents.
        Used by actors on per agent
        Format: { 'agent_1': obs_vector, 'agent_2': obs_vector, ... }
        """
        obs = {name: self._get_agent_obs(name) 
            for name in self.agent_names}
        return obs
    
    def _get_agent_obs(self, agent_name):
        raw_readings, hit_target, hit_obstacle = self._update_rays(agent_name)
        ray_surface_distances = self._adjust_raw_rays(raw_readings, NOISE_STD)

        self._last_ray_surface[agent_name] = ray_surface_distances.astype(np.float32, copy=False)
        
        # Relative position to other agent
        my_pos = self.data.xpos[self.agent_map[agent_name]["body_id"]][:2]
        others = [n for n in self.agent_names if n != agent_name]
        other_pos = self.data.xpos[self.agent_map[others[0]]["body_id"]][:2]
        rel_pos = other_pos - my_pos # 2D vector
        
        # Hit indicators: 1.0 if hit, 0.0 if not
        target_indicator = 1.0 if hit_target else 0.0
        obstacle_indicator = 1.0 if hit_obstacle else 0.0

        obs = np.asarray([*ray_surface_distances, *rel_pos,target_indicator, obstacle_indicator], dtype=np.float32)
        return obs

    def _get_global_state(self, obs_dict: Dict[str, np.ndarray] | None = None) -> np.ndarray:
       """
       Return a single flat array, the critic's input in MAPPO
       """
       if obs_dict is None:
            obs_dict = self._get_local_obs_dict()

 
       global_state = np.concatenate([obs_dict[name] for name in self.agent_names], axis=0).astype(np.float32)
       return global_state
    
    # ---- max-pool and flatten the visited grid ----
    def _coverage_embedding(self):
        H, W = self.visited.shape
        k = 1
        H2, W2 = H // k, W // k
        trimmed = self.visited[:H2*k, :W2*k]
        # max-pool over k×k blocks (any cell visited in the block → 1)
        pooled = trimmed.reshape(H2, k, W2, k).max(axis=(1, 3))
        emb = pooled.astype(np.float32).reshape(-1)   # shape = (H2*W2,)
        return emb  # values in {0.0, 1.0}

    def _critic_extras(self):
        # Normalized time in [0,1]
        t_norm = float(self._step_count) / float(self.max_steps)
        found = float(len(self.found_targets))
        total = float(self.target_nr)
        
        # Normalize counts to [0,1] assuming target_nr > 0
        found_norm = found / max(total, 1.0)
        total_norm = min(total / 10.0, 1.0)
        
        return np.array([t_norm, found_norm, total_norm], dtype=np.float32)

    def _pack_obs(self, local_obs_dict: dict[str, np.ndarray], gs: np.ndarray):
        # Ensure f32 and correct shapes
        obs_out = {}
        for name in self.agent_names:
            lo = np.asarray(local_obs_dict[name], dtype=np.float32)
            st = np.asarray(gs, dtype=np.float32)
            # Optional guard rails (catch bugs early)
            if lo.shape != (self.obs_dim,):
                raise ValueError(f"{name} local obs has wrong shape {lo.shape}")
            if st.shape != (self.state_dim,):
                raise ValueError(f"global state wrong shape {st.shape}")
            obs_out[name] = {"obs": lo, "state": st}
        return obs_out
        

    #  ================= Helper function used in the step()  =================

    def _apply_actions_for_agents(self, action_dict: Dict[str, np.ndarray], frame_skip: int = 1):
        """
        action_dict: { 'agent_1': np.array([x_setpoint, y_setpoint], dtype=float32), ... }
        self.joint_ctrl_map: { 'agent_1': np.array([id_j1, id_j2], dtype=int32), ... }
        """
        # print(" ")
        # print(f"Predict Action: {action_dict}")
        for name in self.agent_names:
            if name not in action_dict:
                raise KeyError(f"Missing action for agent '{name}'")
            delta = np.asarray(action_dict[name], dtype=np.float32)
            if delta.shape != (2,):
                raise ValueError(f"Action for '{name}' must be shape (2,), got {delta.shape}")
            if name not in self.agent_map:
                raise KeyError(f"No ctrl map entry for agent '{name}'")     
            
            ids = self.agent_map[name]["ctrl_ids"]

            if len(ids) != 2:
                raise ValueError(f"Expected 2 actuator ids for '{name}', got {len(ids)}")
            
            joint_target, target_world = self._calculate_joint_and_world_target(name, delta)
            
            self._step_targets[name] = target_world
            self.data.ctrl[ids] = joint_target
            # print("Agent: " + name + " Apply ctrl id: " + str(ids))
            # print(" ")
    
    def _calculate_joint_and_world_target(self, name, delta):
        """
        Compute new joint targets for an agent based on its current (x, y) world position,
        a delta movement, and the initial position.
        """
        body_id = self.agent_map[name]["body_id"]
        init_pos = self.agent_map[name]["init_pos"]

        # current world position (x, y)
        cur_pos = self.data.xpos[body_id][:2]

        # target world position
        target_world = cur_pos + delta

        # convert to joint targets (relative to initial pos)
        joint_targets = target_world - init_pos
        # print(f"Step: {self._step_count} Namd:  {name}, world target: {target_world}")
        # print(f".{self._step_count} Delta :{delta} ")
        return  joint_targets.astype(np.float32), target_world.astype(np.float32)

    def _check_collision(self):
        ncon = self.data.ncon
        if ncon is not None:
            for i in range(ncon):
                contact = self.data.contact[i]
                g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                if ("agent_" in g1) or ("agent_" in g2):
                    # print(f"{g1} collide with {g2}")
                    return True

        return False   
    
    def _reward_find_target_reward_done(self) -> Tuple[float, bool]:
        """
        Team reward: +1 the first time any agent's rays hit a 'target' geom this episode.
        """
        team_rew = 0.0
        # Aggregate newly discovered targets this step
        newly_found = set()

        # print("")
        for name in self.agent_names:
            for gid in self.last_ray_hits.get(name, ()):
                if gid not in self.found_targets:
                    newly_found.add(gid)
                    print(f" Newly found add gid {gid}")

        if newly_found:
            self.found_targets.update(newly_found)
            team_rew += 1.0 * len(newly_found)
            self.steps_since_discovery = 0
        else:
            self.steps_since_discovery += 1
        
        if len(self.found_targets) == self.target_nr:
            found_all_target = True
            team_rew = 100
            # print(" ")
            # print("All targets has been found, searching is done!")
        else:
            found_all_target = False
        return float(team_rew), bool(found_all_target)
    
    def _reward_distance_between_agents(self) -> float:
        pos = [self.data.xpos[self.agent_map[n]["body_id"]][:2] for n in self.agent_names]
        dist = np.linalg.norm(pos[0] - pos[1])
        d_safe = 1.0
        k = 0.3
        if dist < d_safe:
            return - k * (1 - dist) ** 2
        return 0.0
    
    def _reward_milestone(self) -> float:
        lam = 1.0 / self.max_steps
        return - lam * min(self.steps_since_discovery, self.max_steps)
    
    def _compute_individual_reward(self, name) -> Tuple[float, float, float, float]:
        # Get agent information
        bid = self.agent_map[name]["body_id"]
        cur_xy = self.data.xpos[bid][:2]
        self._recent_pos[name].append(cur_xy)
        newly = self._mark_cells_within_range(cur_xy)
        self.per_agent_new_cells[name] = newly

        visit_new_cells_reward = newly * 1
        obstacle_distance_reward = self._reward_obstacle_distance(name)
        idle_reward = self._reward_idle(name)
        
        # Emphasize the reward for avoiding obstacles
        sum_reward = visit_new_cells_reward + obstacle_distance_reward + idle_reward
        
        return sum_reward, visit_new_cells_reward, obstacle_distance_reward, idle_reward
   
    def _mark_cells_within_range(self, agent_xy):
        newly_marked = 0
        xmin, xmax, ymin, ymax = self.grid_bounds
        for i in range(self.grid_H):
            for j in range(self.grid_W):
                cx = xmin + (j + 0.5) * self.cell_size
                cy = ymin + (i + 0.5) * self.cell_size
                dist = np.linalg.norm(np.array([cx, cy]) - agent_xy)
                if dist <= 0.5 and self.visited[i, j] == 0: # 0.5 is the half of cutoff
                    self.visited[i, j] = 1
                    newly_marked += 1

        return newly_marked
    
    def _reward_obstacle_distance(self, name:str) -> float:
        
        obstacle_distance_reward = 0.0
        rays = self._last_ray_surface.get(name)
        
        if rays is None or len(rays) == 0:
            return 0.0

        min_dist = float(np.min(rays))
        cutoff_value = float(0.8)

        if min_dist >= cutoff_value:
            return 0.0
        
        obstacle_distance_reward = (min_dist -  cutoff_value) /  cutoff_value
        # print(f"   Rays: {rays}, penalty: {obstacle_distance_penalty}")
        # print(" ")
        return obstacle_distance_reward

    def _reward_idle(self, name: str, eps=1e-6):
        q = self._recent_pos[name]
        if len(q) < 10:
            return 0.0
        # mean step-to-step displacement
        disp = [np.linalg.norm(np.array(q[i]) - np.array(q[i-1])) for i in range(1, len(q))]
        mean_move = float(np.mean(disp))
        if mean_move < 0.01:
            return -0.3         
        return 0.0      


#  ================= Helper function related to sensor rays  =================
    
    def _update_rays(self, agent_name: str):
        agent_body_id = self.agent_map[agent_name]["body_id"]
        origin = self.data.xpos[agent_body_id][:3]  # Use current agent position

        # Clear previous outputs
        self.ray_geomid_out.fill(-1)
        self.ray_dist_out.fill(-1.0)

        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin,
            self.ray_directions,
            None,  # No geomgroup filtering
            1,     # Include static geoms
            agent_body_id,  # Exclude agent’s own body
            self.ray_geomid_out,
            self.ray_dist_out,
            self.n_rays,
            float(self.ray_length)
            # 0.0
        )

        hits = set()
        
        hit_target = False
        hit_obstacle = False

        for i in range(self.n_rays):
            gid = int(self.ray_geomid_out[i])
            d   = float(self.ray_dist_out[i])

            # ignore invalid/no-hit/out-of-range rays
            if gid == -1 or not np.isfinite(d) or d < 0.0 or d > self.cutoff_value:
                continue

            gname = self._geom_name(gid)
            # count only targets within range
            if "target" in gname:
                hits.add(gid)
                hit_target = True
                # print(".   ")
                # print(f"Target {gname} gid {gid} found at step {self._step_count}")
                # print(f"Ray distance out put: {self.ray_dist_out}")
                # print(f"Geom out {self.ray_geomid_out}")
                # print(".   ")
            else:
                hit_obstacle = True

        self.last_ray_hits[agent_name] = hits

        return self.ray_dist_out, hit_target, hit_obstacle
    
    def _geom_name(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return name or ""
    
    def _adjust_raw_rays(self, raw_readings, noise_std):
        # raw_readings: np.array of length 12
        offsets = np.array([
            0.5,   # 0°
            0.577, # 30°
            0.577, # 60°
            0.5,   # 90°
            0.577, # 120°
            0.577, # 150°
            0.5,   # 180°
            0.577, # 210°
            0.577, # 240°
            0.5,   # 270°
            0.577, # 300°
            0.577  # 330°
        ])
        
        # print(" ")
        # print(f"Raw_reading with -1 {raw_readings}")
        # Replace -1 (no hit) with 1.6 (max range + margin)
        raw_readings = np.where(raw_readings == -1, 1.6, raw_readings)

        # print(f"Raw_reading without -1  {raw_readings}")

        # Subtract offsets
        surface_distances = raw_readings - offsets
        
        # print(f"Surface distance before clip {surface_distances}")
        
        surface_distances = np.clip(surface_distances, 0, None)
        
        # print(f". Surface distance after clip {surface_distances}")
        # Clip max distances to 1.0
        surface_distances = np.clip(surface_distances, None, self.cutoff_value)
        # print(f". .   Surface distance to cutoff_value {surface_distances}")
        # Add Gaussian noise if noise_std > 0
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=surface_distances.shape)
            surface_distances = surface_distances + noise
            
            # Clip again to [0,1]
            surface_distances = np.clip(surface_distances, 0, 1.0)
        # print(f"Surface distance all {surface_distances}")
        # print(" ")
        return surface_distances
    

    # ================= Gym API =================
    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.episode_count += 1
        self._step_count = 0
        # rotate XML after N episodes
        if self.num_xmls > 1 and self.episode_count > 1 and \
           self.episode_count % self.switch_every == 0:
            self._switch_model()
        print(" ")
        print(f"Env: current episode: {self.episode_count}, current_xml file: {self.xml_paths[self.current_xml_index]}")
        print(" ")
        
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._reset_episode_state()
        
        local_obs = self._get_local_obs_dict() # actor input
        local_obs = {k: np.asarray(v, dtype=np.float32) for k, v in local_obs.items()}
        gs_base = self._get_global_state(local_obs).astype(np.float32,copy=False)
        cov = self._coverage_embedding().astype(np.float32, copy=False)
        extras  = self._critic_extras().astype(np.float32, copy=False)
        gs = np.concatenate([gs_base, cov, extras], axis=0).astype(np.float32, copy=False)

        obs = self._pack_obs(local_obs, gs)
        info = {name: {} for name in self.agent_names}
        # print(f"Observation: {obs}" )
        # print(f"Global: {gs}")
        return obs, info
    
    def step(self, action: Dict[str, np.ndarray]):
        
        #For log: previous positions BEFORE applying action
        prev_pos = {
            name: self.data.xpos[self.agent_map[name]["body_id"]][:2].copy()
            for name in self.agent_names   
        }

        # print(f"Step: {self._step_count} Check prev_pos:  {prev_pos}")

        self._apply_actions_for_agents(action)
        
        reached = {name: False for name in self.agent_names}
        collided = False
        substep = 0
        reason = None


        while True:
            mujoco.mj_step(self.model, self.data)
            substep += 1

            collided = self._check_collision()
            # print(" ")
            # print(f"Substeps {sub}")
            if collided:
                break
            
            for name in self.agent_names:
                if not reached[name]:
                    bid = self.agent_map[name]["body_id"]
                    cur = self.data.xpos[bid][:2]
                    tgt = self._step_targets[name]
    
                    if np.linalg.norm(cur - tgt) <= self.reach_tol:
                        reached[name] = True
                        # print(f"Agent {name} tgt is {tgt}, cur is {cur}")
                        # print(f"Agent {name} reach the target position at substeps {substep}! World step: {self._step_count}")
            
            if all(reached.values()):
                # print(f"All agent arrived to the target position! at substeps {substep}")
                # print(" ")
                break

            if substep >= self.max_substeps_per_action:
                break
        
        self.render()

        # Get observation after move
        local_obs = self._get_local_obs_dict()
        local_obs = {k: np.asarray(v, dtype=np.float32) for k, v in local_obs.items()}
        gs_base = self._get_global_state(local_obs).astype(np.float32, copy=False)
        cov = self._coverage_embedding().astype(np.float32, copy=False)
        extras  = self._critic_extras().astype(np.float32, copy=False)
        gs = np.concatenate([gs_base, cov, extras], axis=0).astype(np.float32, copy=False)        
        obs = self._pack_obs(local_obs, gs)
        
        # Calculating common reward: inclduing time step reward: milestone_pen and time_pen. General group reward: agent_distance_pelnaty and find targer reward(as group reward)
        reward = {}
        team_rew, found_all_target = self._reward_find_target_reward_done()
        agent_distance_penalty = self._reward_distance_between_agents()
        milestone_pen = self._reward_milestone()
        # time_pen = -0.01
        
        per_agent_logs = {}
        # Calculating individual reward ( per agent)
        for name in self.agent_names:
         
            r = 0.0
            r += team_rew / self.n_agents
            r += agent_distance_penalty
            r += (0.0 if found_all_target else milestone_pen)
            # r += time_pen
            
            sum_reward, visit_new_cells_reward, obstacle_distance_reward, idle_reward = self._compute_individual_reward(name)
            r = r + sum_reward
            coll_pen = 0
            if collided:
                coll_pen += -50.0
            r += coll_pen                         # collision penalty
            reward[name] = float(r)
            
            # cache PER-AGENT values for logging
            per_agent_logs[name] = {
                "coverage_bonus": float(visit_new_cells_reward),
                "obstacle_distance_reward": float(obstacle_distance_reward),
                "idle_reward": float(idle_reward),
                "team_reward_share": float(team_rew / self.n_agents),
                "prox_pen": float(agent_distance_penalty),
                "coll_pen": float(coll_pen),
                "time_pen": float(0.0 if found_all_target else milestone_pen),
                "newly_marked": int(self.per_agent_new_cells[name]),
                "reward_total": float(r),
            }
        
        self._step_count += 1

        # Episode-over flags
        time_up = (self._step_count >= self.max_steps)
        terminated = {name: False for name in self.agent_names}
        truncated  = {name: False for name in self.agent_names}
        episode_done = bool(found_all_target or collided)
        
        # Task termination (success or collision)
        if found_all_target:
            reason = "success"
            terminated = {name: True for name in self.agent_names}
        elif collided:
            reason = "collision"
            terminated = {name: True for name in self.agent_names}

        # Time-limit truncation (ends episode even if not success/collision)
        if time_up:
            reason = "timeout"
            truncated = {name: True for name in self.agent_names}
        if episode_done or time_up:
            print(f"[EP END] steps={self._step_count} done={episode_done} time_up={time_up}")
        
        terminated["__all__"] = any(terminated.values())
        truncated["__all__"]  = time_up

         # For log sum of visited cells
        coverage_total = int(self.visited.sum())

        # Log per-agent per step
        if self.step_log_writer is not None:
            for name in self.agent_names:
                pl = per_agent_logs[name]

                # Positions
                bid = self.agent_map[name]["body_id"]
                cur_xy = self.data.xpos[bid][:2].copy()
                tgt_xy = self._step_targets.get(name, cur_xy).copy()
                prev_xy = prev_pos[name]
                delta_xy = tgt_xy - prev_xy


                self.step_log_writer.writerow([
                    self.episode_count, self._step_count - 1, name,
                    # float(prev_xy[0]), float(prev_xy[1]),
                    # float(cur_xy[0]),  float(cur_xy[1]),
                    # float(tgt_xy[0]),  float(tgt_xy[1]),
                    # float(delta_xy[0]), float(delta_xy[1]),
                    pl["newly_marked"],
                    coverage_total,
                    pl["team_reward_share"], len(self.found_targets), pl["coverage_bonus"], pl["time_pen"],
                    pl["prox_pen"], pl["coll_pen"], pl["obstacle_distance_reward"], pl["idle_reward"],
                    pl["reward_total"], reason,
                ])

        info = {name: {} for name in self.agent_names}
        if reason is not None:
            for name in self.agent_names:
                info[name]["done_reason"] = reason
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if not self.render_enabled:
            return
        if self.viewer is None:
            try:
                self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
            except Exception:
                return
        try:
            self.viewer.render()
        except Exception:
            self.render_enabled = False
            self.viewer = None

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        if hasattr(self, "log_file_handle") and self.log_file_handle is not None:
            try: self.log_file_handle.close()
            except Exception: pass
            self.log_file_handle = None

        if hasattr(self, "step_log_handle") and self.step_log_handle is not None:
            try: self.step_log_handle.close()
            except Exception: pass
            self.step_log_handle = None

