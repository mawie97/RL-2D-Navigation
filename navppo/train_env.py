import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from multi_ray_goal_env import EpisodeCounterCallback

def train_ppo(
    env_class,
    xml_paths,
    base_dir,
    log_dir,
    csv_log_path,
    model_dir,
    env_dir,
    seed_value,
    num_episodes,
    total_timesteps=int(1e6)

):

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(env_dir, exist_ok=True)

    # Step 1: Create and wrap environment
    
    env = DummyVecEnv([lambda: Monitor(env_class(csv_log_path, xml_paths, headless = False), filename=os.path.join(log_dir, "monitor.csv"))])
    
    model_path = os.path.join(model_dir, "model_ppo.zip")
    vecnorm_path = os.path.join(env_dir, "vecnormalize.pkl")

    if os.path.exists(model_path) and os.path.exists(vecnorm_path):
        print("[INFO] Found existing model + VecNormalize. Resuming training.")

        # Load VecNormalize stats and wrap the env
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True

        # Load model and attach env
        model = PPO.load(model_path, env=env)
        model.set_env(env)

        reset_num_timesteps = False
    else:
        print(f"[INFO] Starting training from scratch")
        env.seed(seed_value)
        env.action_space.seed(seed_value)
        env.observation_space.seed(seed_value)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

        # Step 2: Create and train model
        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.01, seed = seed_value, tensorboard_log=log_dir)
        reset_num_timesteps = True
    # Step 3: Log for the tensorboard
    new_logger = configure(log_dir, ["csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Step 4: Setup the callback
    episode_counter_callback = EpisodeCounterCallback(total_episodes = num_episodes)
    model.learn(total_timesteps=total_timesteps, callback=episode_counter_callback, reset_num_timesteps=reset_num_timesteps,)
    
    env.close()
    
    # Step 5: Save the model
    model.save(os.path.join(model_dir, "model_ppo.zip"))
    env.save(os.path.join(env_dir, "vecnormalize.pkl"))
    
    print(f"[INFO] Training completed and model saved.")
