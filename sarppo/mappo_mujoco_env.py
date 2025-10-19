
# This is a MuJoCo‑based multi‑agent environment. It returns per‑agent 
# observations and actions and also provides a “global state” 
# in the info dictionary that concatenates all agents’ observations.

import csv
import random
from collections import deque
from typing import Dict, Tuple, Any
import numpy as np
import gymnasium as gym
import mujoco
import mujoco_viewer
from ray.rllib.env.multi_agent_env import MultiAgentEnv


SWITCH_EVERY = 10
POSITION_HISTORY_LEN = 20
CUTOFF_VALUE = 1
MAX_DISTANCE = 50.0
MAX_STEPS = 1000
MAX_ACTION = 0.1
AGENT_PREFIX = "agent_"
OBS_DIM = 12
ACT_DIM = 2
N_RAYS = 12
RAY_LENGTH = 1.5
NOISE_STD = 0.00
TARGET_NR = 3

class MultiAgentSAR(MultiAgentEnv):
    """
    MuJoCo + multi-agent wrapper for CTDE (MAPPO).
    - Per-agent obs/action via Dict spaces
    - info['global_state'] returned each step for centralized critic
    - Can rotate between multiple XML models
    """
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
        self.max_distance = MAX_DISTANCE
        self.max_steps = MAX_STEPS
        self.csv_log_path = csv_log_path
        # print(f"Loaded XML paths: {self.xml_paths}")
        # print(" ")

        # coverage grid
        self.grid_bounds = (-10.0, 10.0, -10.0, 10.0)  # xmin, xmax, ymin, ymax
        self.cell_size = 1.0
        xmin, xmax, ymin, ymax = self.grid_bounds
        self.grid_W = int(np.ceil((xmax - xmin) / self.cell_size))
        self.grid_H = int(np.ceil((ymax - ymin) / self.cell_size))
        self.visited = np.zeros((self.grid_H, self.grid_W), dtype=np.uint8)
        
        # Add grid coverage to the critic
        self.cover_k = 4  # try 2 or 4 then see difference; 4 → 5x5=25 dims
        self.cov_dim = (self.grid_H // self.cover_k) * (self.grid_W // self.cover_k)
         

        self.reach_tol = 0.01
        self.max_substeps_per_action = 50000 

        #This is used to save the world target for each agent
        self._step_targets = {}
        
        self._load_model_and_setup(self.current_xml_index)
        self._reset_episode_state()   

        self.found_targets:set[int] = set()
        self.last_ray_hits: Dict[str, set[int]] = {name : set() for name in self.agent_names}
        self.target_nr = TARGET_NR

        self.debug_dump = bool(render_enabled)  # or from env_config["debug_dump"]
        self._printed_reset = False
        self._printed_step  = False

        
        if self.csv_log_path is not None:
            self.log_file_handle = open(self.csv_log_path, mode='w', newline='')
            self.log_writer = csv.writer(self.log_file_handle)
            # self.log_writer.writerow(['Step', 'pre_x', 'pre_y', 'current_x', 'current_y', 'target_x','target_y', 'delta_x', 'delta_y', 'pre_distance','current_distance','dist_value','RE: dis_change', 'RE: distance', 'RE: obstacle_avoidance', 'Total: reward' , 'Status'])
            self.log_writer.writerow(['Episode', 'Status'])

    def _load_model_and_setup(self, index):
        self._load_model(index)
        self._get_agents_from_model()
        self.agent_map = self._build_agent_map()
        self._setup_agents_ray_casting()
        self._setup_action_observation_spaces()

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
            ray_directions.append([dx, dy, dz])

        ray_directions = np.array(ray_directions, dtype=np.float64).reshape(-1)
        return ray_directions
    
    # x,y to grid index ij
    def _xy_to_ij(self, xy):
        xmin, xmax, ymin, ymax = self.grid_bounds
        x, y = float(xy[0]), float(xy[1])
        j = int(np.floor((x - xmin) / self.cell_size))
        i = int(np.floor((y - ymin) / self.cell_size))
        if i < 0 or i >= self.grid_H or j < 0 or j >= self.grid_W:
            return None
        return i, j

    def _mark_cell_of_point(self, xy):
        ij = self._xy_to_ij(xy)
        if ij is None:
            return 0
        i, j = ij
        if self.visited[i, j] == 0:
            self.visited[i, j] = 1
            return 1
        return 0


    def _setup_action_observation_spaces(self):
        f32 = np.float32
        cutoff = f32(self.cutoff_value)

        self.max = MAX_ACTION        
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM
        self.global_dim = self.obs_dim * self.n_agents
        # ADD: coverage embedding size
        self.state_dim = self.global_dim + self.cov_dim
        
        gs_low  = np.concatenate([
            np.zeros((self.global_dim,), dtype=f32),         # all agents' rays >= 0
            np.zeros((self.cov_dim,),    dtype=f32),         # coverage >= 0
        ]).astype(f32)
        gs_high = np.concatenate([
            np.full((self.global_dim,), cutoff, dtype=f32),  # rays <= cutoff
            np.ones((self.cov_dim,),      dtype=f32),        # coverage in {0,1}
        ]).astype(f32)

        obs_low  = np.zeros((self.obs_dim,), dtype=f32)
        obs_high = np.full((self.obs_dim,), cutoff, dtype=f32)
        # self.observation_space = gym.spaces.Dict({
        #     "obs":   gym.spaces.Box(low=f32(-np.inf), high=f32(np.inf), shape=(self.obs_dim,),   dtype=f32),
        #     "state": gym.spaces.Box(low=f32(-np.inf), high=f32(np.inf), shape=(self.state_dim,), dtype=f32),
        # })

        self.observation_space = gym.spaces.Dict({
            "obs":   gym.spaces.Box(low=obs_low, high=obs_high, dtype=f32),
            "state": gym.spaces.Box(low=gs_low,  high=gs_high,  dtype=f32),
        })
      
        # One agent's action space: (2,)
        a_low  = f32(-self.max)
        a_high = f32(self.max)
        self.action_space = gym.spaces.Box(low=a_low, high=a_high, shape=(self.act_dim,), dtype=f32)

    def _switch_model(self):
        self.current_xml_index = (self.current_xml_index + 1) % self.num_xmls
        self._load_model_and_setup(self.current_xml_index)

    def _reset_episode_state(self):
        self._step_count = 0
        self.position_history.clear()
        self.found_targets = set()
        self.visited.fill(0)
        # Check if it need to be reset every step
        self.last_ray_hits = {name: set() for name in self.agent_names}

    def _geom_name(self, geom_id: int) -> str:
        if geom_id < 0:
            return ""
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        return name or ""

    def _get_agent_obs(self, agent_name):
        raw_readings = self._update_rays(agent_name)
        ray_surface_distances = self._adjust_raw_rays(raw_readings, NOISE_STD)
        obs = np.asarray([*ray_surface_distances ], dtype=np.float32)
        return obs
    
    def _get_local_obs_dict(self) -> dict[str, np.ndarray]:
        """
        Return local observations for all agents.
        Used by actors on per agent
        Format: { 'agent_1': obs_vector, 'agent_2': obs_vector, ... }
        """
        obs = {name: self._get_agent_obs(name) 
            for name in self.agent_names}
        return obs

    def _get_global_state(self, obs_dict: Dict[str, np.ndarray] | None = None) -> np.ndarray:
       """
       Return a single flat array, the critic's input in MAPPO
       """
       if obs_dict is None:
            obs_dict = self._get_local_obs_dict()

 
       global_state = np.concatenate([obs_dict[name] for name in self.agent_names], axis=0).astype(np.float32)
       return global_state

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

        return  joint_targets.astype(np.float32), target_world.astype(np.float32)


    def _compute_reward_done(self) -> Tuple[float, bool]:
        """
        Team reward: +5 the first time any agent's rays hit a 'target' geom this episode.
        """
        team_rew = 0.0
        # Aggregate newly discovered targets this step
        newly_found = set()
        # print("")
        for name in self.agent_names:
            for gid in self.last_ray_hits.get(name, ()):
                if gid not in self.found_targets:
                    newly_found.add(gid)

        if newly_found:
            self.found_targets.update(newly_found)
            team_rew += 5.0 * len(newly_found)
        
        if len(self.found_targets) == self.target_nr:
            done = True
            print(" ")
            print("All targets has been found, searching is done!")
        else:
            done = False
        return float(team_rew), bool(done)


    def _update_rays(self, agent_name: str):
        agent_body_id = self.agent_map[agent_name]["body_id"]
        origin = self.data.xpos[agent_body_id][:3]  # Use current agent position
         
        # Clear previous outputs (robustness)
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
            self.ray_length
        )

        hits = set()

        for gid in self.ray_geomid_out:
            # Check if the gid is actually return -1 if no hit at all
            if gid != -1:
                gname = self._geom_name(gid)
                if "target" in gname:
                    hits.add(int(gid))
        self.last_ray_hits[agent_name] = hits

        return self.ray_dist_out
    
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
        
        # print(f"Surface distance after clip {surface_distances}")
        # Clip max distances to 1.0
        surface_distances = np.clip(surface_distances, None, self.cutoff_value)
        # print(f"Surface distance to cutoff_value {surface_distances}")
        # Add Gaussian noise if noise_std > 0
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=surface_distances.shape)
            surface_distances = surface_distances + noise
            
            # Clip again to [0,1]
            surface_distances = np.clip(surface_distances, 0, 1.0)
        # print(f"Surface distance all {surface_distances}")
        # print(" ")
        return surface_distances
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
        gs = np.concatenate([gs_base, cov], axis=0).astype(np.float32, copy=False)

        # obs = {name: {"obs": local_obs[name], "state": gs} for name in self.agent_names}
        obs = self._pack_obs(local_obs, gs)
        info = {name: {} for name in self.agent_names}
        print("Spaces dtypes:",
        self.observation_space["obs"].dtype,
        self.observation_space["state"].dtype,
        " | action dtype:", self.action_space.dtype)

        if self.episode_count == 1:
            one = next(iter(obs.values()))
            print("First reset dtypes:",
                one["obs"].dtype, one["state"].dtype,
                "shapes:", one["obs"].shape, one["state"].shape)

        # Debug
        # if self.debug_dump and not self._printed_reset:
        #     np.set_printoptions(precision=4, suppress=True)
        #     for name in self.agent_names:
        #         print(f" [ENV_RESET_Agent] {name} local={np.array2string(local_obs[name], separator=', ')}  sum={local_obs[name].sum():.6f}")
        #     print(f" [ENV_RESET_Global] global={np.array2string(gs, separator=', ')}  sum={gs.sum():.6f}")
        #     print(" ")
        #     self._printed_reset = True

        return obs, info
    
    def step(self, action: Dict[str, np.ndarray]):
        
        self._apply_actions_for_agents(action)
        
        reached = {name: False for name in self.agent_names}
        collided = False
        substep = 0
        reason = None
        
        per_agent_new_cells = {name: 0 for name in self.agent_names}
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
                    # NEW: mark grid cell (counts only first time a cell is visited)
                    per_agent_new_cells[name] += self._mark_cell_of_point(cur)
                    
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
        
        local_obs = self._get_local_obs_dict()
        local_obs = {k: np.asarray(v, dtype=np.float32) for k, v in local_obs.items()}
        gs_base = self._get_global_state(local_obs).astype(np.float32, copy=False)
        cov = self._coverage_embedding().astype(np.float32, copy=False)                         # (cov_dim,)
        gs = np.concatenate([gs_base, cov], axis=0).astype(np.float32, copy=False)  
        # obs = {name: {"obs": local_obs[name], "state": gs} for name in self.agent_names}
        obs     = self._pack_obs(local_obs, gs)
        #To be implement
        team_rew, found_all_target = self._compute_reward_done()

        reward = {}
        for name in self.agent_names:
            r = 0.0
            r += per_agent_new_cells[name] * 0.5     # per-agent coverage bonus
            r += team_rew / self.n_agents            # share team reward evenly
            if collided:
                r += -2.0                            # optional collision penalty
            reward[name] = float(r)
        
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
        # __all__ keys (required!)
        terminated["__all__"] = any(terminated.values())
        truncated["__all__"]  = time_up

        info = {name: {} for name in self.agent_names}
        if reason is not None:
            for name in self.agent_names:
                info[name]["done_reason"] = reason
        return obs, reward, terminated, truncated, info
    
    def _check_collision(self):
        ncon = self.data.ncon
        if ncon is not None:
            for i in range(ncon):
                contact = self.data.contact[i]
                g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                if ("agent_" in g1) or ("agent_" in g2):
                    print(f"{g1} collide with {g2}")
                    return True

                # if ("agent_" in g1 and "obstacle_" in g2) or ("agent_" in g2 and "obstacle_" in g1):
                #     return True
        return False
    
    # ---- helper: max-pool and flatten the visited grid ----
    def _coverage_embedding(self):
        H, W = self.visited.shape
        k = self.cover_k
        H2, W2 = H // k, W // k
        trimmed = self.visited[:H2*k, :W2*k]
        # max-pool over k×k blocks (any cell visited in the block → 1)
        pooled = trimmed.reshape(H2, k, W2, k).max(axis=(1, 3))
        emb = pooled.astype(np.float32).reshape(-1)   # shape = (H2*W2,)
        return emb  # values in {0.0, 1.0}

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
