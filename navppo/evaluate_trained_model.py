import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard.writer import SummaryWriter
from multi_ray_goal_env import MujocoGoalEnv

current_dir = os.path.dirname(__file__)
xml_base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'layouts', 'eval'))
xml_paths = sorted([os.path.join(xml_base_dir, f) for f in os.listdir(xml_base_dir) if f.endswith(".xml")])

current_setup = "setup_1_noise_0"
csv_log_path = f"../../runs/{current_setup}/eval/noise_03/log/eval_log.csv"
os.makedirs(os.path.dirname(csv_log_path), exist_ok=True)
tensorboard_eval_log_dir = f"../../runs/{current_setup}/eval/noise_03/tensorboard/"

def make_env():
    return Monitor(MujocoGoalEnv(csv_log_path, xml_paths))

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize.load(f"../../runs/{current_setup}/envs/vecnormalize.pkl", eval_env)

eval_env.training = False
eval_env.norm_reward = False
eval_env.norm_obs = True

model = PPO.load(f"../../runs/{current_setup}/models/model_ppo.zip")

writer = SummaryWriter(log_dir=tensorboard_eval_log_dir)

num_episodes = 0
max_episodes = 60
episode_rewards = []

obs = eval_env.reset()
while num_episodes < max_episodes:
    action, _ = model.predict(obs)  # type: ignore
    obs, reward, done, info = eval_env.step(action)

    eval_env.venv.envs[0].render()  # type: ignore

    if done[0]:
        num_episodes += 1
        ep_info = info[0].get("episode")
        if ep_info is not None:
            episode_rewards.append(ep_info["r"])
            writer.add_scalar("Eval/episode_reward", ep_info["r"], num_episodes)
            print(f"Episode {num_episodes + 1} reward: {ep_info['r']}")
            

avg_reward = sum(episode_rewards) / len(episode_rewards)
print(f"Average reward over {max_episodes} episodes: {avg_reward}")
writer.add_scalar("Eval/average_reward", avg_reward, 0)

writer.close()
eval_env.close()
