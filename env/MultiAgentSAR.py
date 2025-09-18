# mappo_mujoco_env.py
import csv
import random
from collections import deque
from typing import Dict, Tuple, Any
import numpy as np
import gymnasium as gym
import mujoco

SWITCH_EVERY = 10
POSITION_HISTORY_LEN = 20
CUTOFF_VALUE = 0.2
MAX_DISTANCE = 50.0
MAX_STEPS = 1000
MAX_ACTION = 0.1
AGENT_PREFIX = "agent_"
OBS_DIM = 17
ACT_DIM = 2
N_RAYS = 12
RAY_LENGTH = 1.5
NOISE_STD = 0.01

class MultiAgentSAR(gym.Env):
    """
    MuJoCo + multi-agent wrapper for CTDE (MAPPO).
    - Per-agent obs/action via Dict spaces
    - info['global_state'] returned each step for centralized critic
    - Can rotate between multiple XML models
    """
    def __init__(self, csv_log_path, xml_paths):
        super().__init__()
        
        self.xml_paths = xml_paths
        self.num_xmls = len(self.xml_paths)
        self.switch_every = SWITCH_EVERY
        self.episode_count = 0     
        self.current_xml_index = 0

        self.viewer = None
        self._seed = None

        self.position_history = deque(maxlen=POSITION_HISTORY_LEN)
        
        self.cutoff_value = CUTOFF_VALUE
        self.max_distance = MAX_DISTANCE
        self.max_steps = MAX_STEPS
        self.csv_log_path = csv_log_path
        print(f"Loaded XML paths: {self.xml_paths}")
        
        self._load_model_and_setup(self.current_xml_index)
        self._reset_episode_state()    
       
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

        self.agent_names = list(self.agent_names)
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
    
    def _setup_action_observation_spaces(self):
        self.max = MAX_ACTION        
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM
        
        self.observation_space = gym.spaces.Dict({
        name: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        for name in self.agent_names
        })
      
        self.action_space = gym.spaces.Dict({
            name: gym.spaces.Box(low=-self.max, high=self.max, shape=(self.act_dim,), dtype=np.float32)
            for name in self.agent_names
        })

    def _switch_model(self):
        self.current_xml_index = (self.current_xml_index + 1) % self.num_xmls
        self._load_model_and_setup(self.current_xml_index)

    def _reset_episode_state(self):
        self._step_count = 0
        self.position_history.clear()

    def _get_agent_obs(self, agent_id):
        raw_readings = self._update_rays(agent_id)
        ray_surface_distances = self._adjust_raw_rays(raw_readings, NOISE_STD)
        obs = np.array([*ray_surface_distances ], dtype=np.float32)
        return obs

    def _get_global_state(self):
       obs_all = [self._get_agent_obs(i) for i, _ in enumerate(self.agent_names)]
       global_state = np.concatenate(obs_all, axis=0).astype(np.float32)
       return global_state

    def _apply_actions_for_agents(self, action_dict: Dict[str, np.ndarray], frame_skip: int = 1):
        """
        action_dict: { 'agent_1': np.array([x_setpoint, y_setpoint], dtype=float32), ... }
        self.joint_ctrl_map: { 'agent_1': np.array([id_j1, id_j2], dtype=int32), ... }
        """
        
        for name in self.agent_names:
            if name not in action_dict:
                raise KeyError(f"Missing action for agent '{name}'")
            delta = np.asarray(action_dict[name], dtype=np.float32)
            if delta.shape != (2,):
                raise ValueError(f"Action for '{name}' must be shape (2,), got {a.shape}")
            if name not in self.joint_ctrl_map:
                raise KeyError(f"No ctrl map entry for agent '{name}'")
            ids = self.agent_map[name]["ctrl_ids"]
            if len(ids) != 2:
                raise ValueError(f"Expected 2 actuator ids for '{name}', got {len(ids)}")
            joint_target = self._calculate_joint_target(name, delta)
            
            self.data.ctrl[ids] = joint_target
 
    def _calculate_joint_target(self, name, delta):
        """
        Compute new joint targets for an agent based on its current (x, y) world position,
        a delta movement, and the initial position.
        """
        body_id = self.agent_map[name]["body_id"]
        init_pos = self.agent_map[name]["init_pos"]

        # current world position (x, y)
        cur_pos = self.data.xpos[body_id][:2]

        # target world position
        target_pos = cur_pos + delta

        # convert to joint targets (relative to initial pos)
        joint_targets = target_pos - init_pos

        return joint_targets.astype(np.float32)


    def _compute_reward_done(self) -> Tuple[float, bool]:
        # TODO: define team reward and termination
        # e.g., exploration coverage, collision penalty, success on find
        team_rew = 0.0
        done = False
        return float(team_rew), bool(done)

    def _update_rays(self, agent_body_id):
        origin = self.data.xpos[agent_body_id][:3]  # Use current agent position
        
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
        
        # Replace -1 (no hit) with 1.6 (max range + margin)
        raw_readings = np.where(raw_readings == -1, 1.6, raw_readings)
        
        # Subtract offsets
        surface_distances = raw_readings - offsets
        
        # Clip negative distances to zero
        surface_distances = np.clip(surface_distances, 0, None)
        
        # Clip max distances to 1.0
        surface_distances = np.clip(surface_distances, None, self.cutoff_value)
        
        # Add Gaussian noise if noise_std > 0
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=surface_distances.shape)
            surface_distances = surface_distances + noise
            
            # Clip again to [0,1]
            surface_distances = np.clip(surface_distances, 0, 1.0)
        
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

        # rotate XML after N episodes
        if self.num_xmls > 1 and self.episode_count > 1 and \
           self.episode_count % self.switch_every == 0:
            self._switch_model()
        print(f"current episode: {self.episode_count}, current_xml file: {self.xml_paths[self.current_xml_index]}")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._reset_episode_state()

        obs = {name: self._get_agent_obs(i) for i, name in enumerate(self.agent_names)}
        info = {"global_state": self._get_global_state()}
        return obs, info
    
    # {
    # "agent_1": np.array([0.2, -0.1]),
    # "agent_2": np.array([1.0, 0.0]),
    # }
    def step(self, action: Dict[str, np.ndarray]):
        
        mujoco.mj_step(self.model, self.data)

        self._apply_actions_for_agents(action)


        # Check if it is necessary!
        # if self.steps == 0:
        #     self._build_agent_map()

        obs = {name: self._get_agent_obs(i) for i, name in enumerate(self.agent_names)}
        team_rew, done = self._compute_reward_done()

        reward = {name: float(team_rew) for name in self.agent_names}
        terminated = {name: bool(done) for name in self.agent_names}
        truncated = {name: (self._step_count >= self.max_steps) for name in self.agent_names}
        info = {"global_state": self._get_global_state()}
        self._step_count += 1

        return obs, reward, terminated, truncated, info
    
    def _build_agent_map(self):
        agent_map = {}

        for agent in self.agent_names:
            # Actuator indices
            ids = []
            for j in [1, 2]:
                act_name = f"{agent}_j{j}"
                a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                ids.append(a_id)
            ids = np.array(ids, dtype=np.int32)

            # Original body position (xy)
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, agent)
            pos = self.model.body_pos[body_id][:2].copy()  # only x, y

            agent_map[agent] = {"ctrl_ids": ids, "init_pos": pos, "body_id": body_id}

        return agent_map

    # ================= Render / Close =================
    def render(self):
        if not hasattr(self, "viewer") or self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except RuntimeError:
                        pass
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None