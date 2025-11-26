import os
import sys
import random
import gymnasium as gym
import mujoco
import mujoco_viewer
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
import csv
from datetime import datetime
from collections import deque


from multi_ray_goal_config import (
    N_RAYS,
    RAY_LENGTH,
    MAX_DISTANCE,
    MAX_STEPS,
    CUTOFF_VALUE,
    SWITCH_EVERY,
    MAX_X,
    MAX_Y,
    NOISE_STD,
    TIME_PENALTY,
    STUCK_PENALTY,
    GOAL_THRESHOLD,
    POSITION_HISTORY_LEN,
    X_MIN,X_MAX,Y_MIN,Y_MAX,
)

class EpisodeCounterCallback(BaseCallback):
    def __init__(self, total_episodes: int):
        super(EpisodeCounterCallback, self).__init__()
        self.total_episodes = total_episodes
        self.episode_count = 0

    def _on_step(self) -> bool:
        done = self.locals.get("dones", [False])[0]  # Access the first element of 'dones'
        if done:
            self.episode_count += 1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # print(f"Episode {self.episode_count} finished at {current_time}")
            # print(" ")
            
            if self.episode_count >= self.total_episodes:
                    # print(f"Training stopped after {self.total_episodes} episodes.")
                    return False
        return True

class MujocoGoalEnv(gym.Env):
    def __init__(self, csv_log_path, xml_paths):
        super(MujocoGoalEnv, self).__init__()
        
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
        self.x_min, self.x_max, self.y_min, self.y_max = X_MIN, X_MAX, Y_MIN, Y_MAX

        self.deadend_centers = []   # list of np.array([x, y])
        self.DEADEND_RADIUS = 1.0
        self.DEADEND_PENALTY = -1.0
        self.best_distance = np.inf
        self.steps_since_improvement = 0
        self.prev_dist_to_deadend = None
        self.stuck = False

        print(" ")
        print(f" path : {self.xml_paths}")
        print(" ")
        self._load_model_and_setup(self.current_xml_index)
        self._reset_episode_state()
        
        if self.csv_log_path is not None:
            # 1. Step-level log
            self.step_log_file = open(self.csv_log_path, mode='w', newline='')
            self.step_log_writer = csv.writer(self.step_log_file)
            self.step_log_writer.writerow([
                'Episode', 'Step',
                'pre_x', 'pre_y',
                'current_x', 'current_y',
                'target_x', 'target_y',
                'delta_x', 'delta_y',
                'pre_distance', 'current_distance',
                'dist_value',
                'RE: dis_change',
                'RE: distance',
                'RE: obstacle_avoidance',
                'RE: deadend_penalty',
                'RE: escape_reward',
                'Total: reward',
                'Status'
            ])

            # 2. Episode-level log
            episodes_log_path = self.csv_log_path.replace('.csv', '_episodes.csv')
            self.episode_log_file = open(episodes_log_path, mode='w', newline='')
            self.episode_log_writer = csv.writer(self.episode_log_file)
            self.episode_log_writer.writerow(['Episode', 'Status'])

    def _load_model_and_setup(self, index):
        self._load_model(index)
        self._setup_ray_casting()
        self._setup_action_observation_spaces()

    def _load_model(self, index):
        self.model = mujoco.MjModel.from_xml_path(self.xml_paths[index])
        self.data = mujoco.MjData(self.model)

        self.goal_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.agent_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        
        self.goal_pos = self.data.xpos[self.goal_id][:2]
        self.agent_pos = self.data.xpos[self.agent_id][:2]
        
        self.dist_sensor_ids = self._find_dist_sensor_ids()
        self.num_obstacles = len(self.dist_sensor_ids)
    
    def _setup_ray_casting(self):
        self.n_rays = N_RAYS
        self.ray_length = RAY_LENGTH
        self.ray_directions = self._compute_ray_directions(self.n_rays)
        self.ray_geomid_out = np.zeros(self.n_rays, dtype=np.int32)
        self.ray_dist_out = np.zeros(self.n_rays, dtype=np.float64)

    def _setup_action_observation_spaces(self):
        self.max_x = MAX_X
        self.max_y = MAX_Y
        self.action_space = spaces.Box(low=np.array([-self.max_x, -self.max_y]), high=np.array([self.max_x, self.max_y]), dtype=np.float32)
        
        obs_dim = 2 + 2 + 1 + self.n_rays
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def _reset_episode_state(self):
        self.initial_distance = 0
        self.prev_pos = np.zeros(2)
        self.prev_distance = 0
        self.steps = 0
        self.original_agent_pos = np.zeros(2)
        self.position_history.clear()

        # per-episode stuck tracking
        self.best_distance = np.inf
        self.steps_since_improvement = 0
        self.prev_dist_to_deadend = None
        self.stuck = False
    
    def _find_dist_sensor_ids(self):
        dist_sensor_ids = []
        for sensor_id in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if sensor_name and sensor_name.startswith("dist"):
                dist_sensor_ids.append(sensor_id)
        
        return dist_sensor_ids

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

    def adjust_raw_rays(self, raw_readings, noise_std):
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

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)

        return [seed]
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.episode_count += 1
        print (" ")
        if self.num_xmls > 1 and self.episode_count > 1 and self.episode_count % self.switch_every == 0:
            self._switch_model()
            # print(f"Switch model")
        print(f"current episode: {self.episode_count}, current_xml file: {self.xml_paths[self.current_xml_index]}")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        
        self._reset_episode_state()
        obs = self._get_obs()
        return obs, {}
    
    def _switch_model(self):

        self.close()  # Close the current model and viewer
        self.current_xml_index = (self.current_xml_index + 1) % self.num_xmls
        if self.current_xml_index >= self.num_xmls:
            self.current_xml_index = 0
        self._load_model(self.current_xml_index)
        self.deadend_centers = []

    def _get_obs(self):
        goal_pos = self.data.xpos[self.goal_id][:2]
        agent_pos = self.data.xpos[self.agent_id][:2]
        
        distance_to_goal = np.linalg.norm(goal_pos - agent_pos)
        raw_readings = self._update_rays()
        surface_distances = self.adjust_raw_rays(raw_readings, NOISE_STD)
        
        obs = np.array([*agent_pos, *goal_pos, distance_to_goal, *surface_distances ], dtype=np.float32)

        return obs
    
    def step(self, action):
        reward, done  = 0, False
        status = ""
        
        delta_x, delta_y = action

        mujoco.mj_step(self.model, self.data)
        
        # Initialize pos/distance if first step
        if self.steps == 0:
            self._initialize_tracking()
            
        target_x, target_y = self._calculate_targets_and_apply_control(delta_x, delta_y)

        # Move the agent to the target position and check for collision
        collided = self._move_agent_to_target(target_x, target_y)

        # End episode if collision detected
        if collided:
            print("Collision detected!")
            reward -= 100
            status = "Collision"
            if hasattr(self, "episode_log_writer"):
                self.episode_log_writer.writerow([self.episode_count, status])
            self.episode_log_file.flush()
            done = True
                    
        current_pos = self.data.xpos[self.agent_id][:2]
        self.position_history.append(current_pos.tolist())
        current_distance_to_goal = np.linalg.norm(current_pos - self.goal_pos)
        # print(f"{current_distance_to_goal}")
        
        # New added Update progress
        self._update_progress(current_distance_to_goal)
        # if stuck, memorize this region as a deadend
        self.stuck = self.is_stuck()
        if self.stuck:
            self._add_deadend_center(current_pos)


        sum_reward, distance_change_reward, distance_reward, dist_obstacle_reward, deadend_penalty, escape_reward = self._calculate_rewards(current_pos, current_distance_to_goal)
        reward += sum_reward
        
        if self._is_out_of_bounds(current_pos):
            print(f"Out of bounds: {current_pos}")
            reward -= 100
            status = "Out_of_bounds"
            if hasattr(self, "episode_log_writer"):
                self.episode_log_writer.writerow([self.episode_count, status])
            self.episode_log_file.flush()
            done = True

        # End episode if agent reaches the goal
        if self._check_goal_reached(current_distance_to_goal):
            reward += 300
            status = "Goal_reached"
            if hasattr(self, "episode_log_writer"):
                self.episode_log_writer.writerow([self.episode_count, status])
            print("Goal reached!")
            self.episode_log_file.flush()
            done = True
        
        # End episode if max steps reached
        if self.steps >= self.max_steps:
            print(f"Exceed max steps!  Max Steps: {self.max_steps}")
            reward -= 100
            status = "Over_max_steps"
            if hasattr(self, "episode_log_writer"):
                self.episode_log_writer.writerow([self.episode_count, status])
            self.episode_log_file.flush()
            done = True
            
        if hasattr(self, "step_log_writer"):
            self.step_log_writer.writerow([
                self.episode_count,
                self.steps,
                self.prev_pos[0], self.prev_pos[1],
                current_pos[0], current_pos[1],
                target_x, target_y,
                delta_x, delta_y,
                self.prev_distance,
                current_distance_to_goal,
                0,
                distance_change_reward,
                distance_reward,
                dist_obstacle_reward,
                deadend_penalty, 
                escape_reward,
                reward,
                status
            ])

        self.prev_pos = np.copy(current_pos)
        self.prev_distance = np.copy(current_distance_to_goal)
        obs = self._get_obs()
        self.steps += 1
        return obs, reward, done, False, {}
    
    def _add_deadend_center(self, current_pos):
        MERGE_RADIUS = 0.5
        current_pos = np.array(current_pos)

        for c in self.deadend_centers:
            if np.linalg.norm(current_pos - c) < MERGE_RADIUS:
                return

        self.deadend_centers.append(current_pos)
        print("New deadend center:", current_pos)

    # If the steps is 0, then initialize the original_agent position, previous position and distance
    def _initialize_tracking(self):
        self.original_agent_pos = np.copy(self.agent_pos)
        self.prev_pos = np.copy(self.agent_pos)
        self.prev_distance = np.linalg.norm(self.prev_pos - self.goal_pos)

    def _calculate_targets_and_apply_control(self, delta_x, delta_y):
        # Calculate the target position in goal / world frame
        target_x =  self.data.xpos[self.agent_id][0] + delta_x
        target_y = self.data.xpos[self.agent_id][1] + delta_y
        
        # Calculate the target joint position on joint frame
        target_x_j = target_x - self.original_agent_pos[0]
        target_y_j = target_y - self.original_agent_pos[1]
        
        # Set target joint position the pid control position for the joint
        self.data.ctrl[0] = target_x_j
        self.data.ctrl[1] = target_y_j
        
        return target_x, target_y
    

    def _move_agent_to_target(self, target_x, target_y):
        collided = False
        steps_taken = 0
        reach = False
        
        while not reach:
            mujoco.mj_step(self.model, self.data)
            if self.check_collision():
                collided = True
                break
            current_pos = self.data.xpos[self.agent_id][:2]
            steps_taken += 1
            
            self.render() # Render during the evaluation will make the simulation smoother
            if np.abs(current_pos[0] - target_x) < 0.01 and np.abs(current_pos[1] - target_y) < 0.01:
                reach = True

        return collided
    
    def _calculate_rewards(self, current_pos, current_distance_to_goal):
        time_penalty = TIME_PENALTY
        distance_reward = 1 - (current_distance_to_goal / self.max_distance) # Reward between 0 and 1, where 1 is the closest to the goal
        distance_change_reward = np.clip((self.prev_distance - current_distance_to_goal) / 0.15, -1, 1) # 0.15 is the maximum distance per steps
        dist_obstacle_reward = self.check_obstacle_distance() # Reward range [-1, 0]
        
        # Special case: dead end
        deadend_penalty = 0.0
        escape_reward = 0.0
        if self.deadend_centers:
            dists = [np.linalg.norm(current_pos - c) for c in self.deadend_centers]
            min_d = min(dists)
            
            if min_d < self.DEADEND_RADIUS:
                # Strong penalty near the center, fades to 0 at the radius edge
                deadend_penalty = self.DEADEND_PENALTY * (self.DEADEND_RADIUS - min_d) / self.DEADEND_RADIUS
                print(f"Deadend penalty is {deadend_penalty}")
            
            if self.prev_dist_to_deadend is not None:
                delta_dead = min_d - self.prev_dist_to_deadend
                ESCAPE_WEIGHT = 0.5
                escape_reward = ESCAPE_WEIGHT * np.clip(delta_dead / 0.15, -1, 1) # if delta_dead > 0 encourage moving away else moving toward dead end

            self.prev_dist_to_deadend = min_d
        else:
            self.prev_dist_to_deadend = None
        

        # if the agent is stuck
        if self.stuck:
            # temporarily ignore the goal-distance attraction to encourage the get away
            distance_reward = 0.0
            distance_change_reward = 0.0
            time_penalty = TIME_PENALTY + STUCK_PENALTY
        
        else:
            pass
        
        # Emphasize the reward for avoiding obstacles
        sum_reward = (0.5 * distance_reward + 1 * distance_change_reward + 2 * dist_obstacle_reward + deadend_penalty + escape_reward + time_penalty)

        return sum_reward, distance_change_reward, distance_reward, dist_obstacle_reward, deadend_penalty, escape_reward
    
    def check_obstacle_distance(self):
        obstacle_distance_penalty = 0.0
        dist_values = [self.data.sensordata[dist_id] for dist_id in self.dist_sensor_ids]
        if len(dist_values) == 0:
            return 0
        min_dist = np.min(dist_values)

        if min_dist >= self.cutoff_value:
            return 0.0
        else:
            obstacle_distance_penalty= (min_dist - self.cutoff_value)/ self.cutoff_value
            return obstacle_distance_penalty
    
    def is_stuck(self):
        if len(self.position_history) < POSITION_HISTORY_LEN:
            return False

        xs = [pos[0] for pos in self.position_history]
        ys = [pos[1] for pos in self.position_history]
        
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        RANGE_THRESH = 0.5          # how small the movement box
        NO_PROGRESS_STEPS = 50      # max steps with no improvement

        small_region = (x_range < RANGE_THRESH) and (y_range < RANGE_THRESH)
        no_progress = (self.steps_since_improvement >= NO_PROGRESS_STEPS)
        if small_region and no_progress:
            print("Agent is stuck")
        return small_region and no_progress
    
    def _is_out_of_bounds(self, pos):
        x, y = pos
        return (x < self.x_min or x > self.x_max or
                y < self.y_min or y > self.y_max)

    # In the xml file, the contact between the agent and the goal, and the agent and the floor, so the only contact is between the agent and the obstacles.
    # The name of the obstacle need to have "obstacle" as prefix
    def check_collision(self):
        if self.data.ncon is not None:
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                
                if ("obstacle" in geom1_name) or ("obstacle" in geom2_name):
                    return True
        return False
    
    def _update_rays(self):
        origin = self.data.xpos[self.agent_id][:3]  # Use current agent position
        
        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin,
            self.ray_directions,
            None,  # No geomgroup filtering
            1,     # Include static geoms
            self.agent_id,  # Exclude agent’s own body
            self.ray_geomid_out,
            self.ray_dist_out,
            self.n_rays,
            self.ray_length
        )
        return self.ray_dist_out
        
    def _check_goal_reached(self, current_distance_to_goal):
        return current_distance_to_goal < GOAL_THRESHOLD
    
    def render(self):
        # Launch viewer once
        if not hasattr(self, "viewer") or self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except RuntimeError:
                pass

        # Always sync viewer with latest sim state
        if self.viewer is not None:
            self.viewer.sync()
    
    def _update_progress(self, current_distance_to_goal):
        """
        Check if it stopped making progress
        """

        progress = 0.05  # minimum improvement progress

        if current_distance_to_goal < self.best_distance - progress:
            self.best_distance = current_distance_to_goal
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

    def close(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def close_file(self):
        if hasattr(self, "step_log_file") and self.step_log_file:
            self.step_log_file.close()
        if hasattr(self, "episode_log_file") and self.episode_log_file:
            self.episode_log_file.close()


