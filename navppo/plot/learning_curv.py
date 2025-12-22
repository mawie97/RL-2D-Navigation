from curses import window
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

current_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'runs'))
fixed_path = os.path.join(base_dir, "setup_1_noise_0.03", "combined_training_log.csv")
randomized_path = os.path.join(base_dir, "setup_2_noise_0.03", "combined_training_log.csv")

fixed_path = glob.glob(fixed_path)
rand_path = glob.glob(randomized_path)

plot_dir = os.path.join(current_dir, "..", "..",'plots')
os.makedirs(plot_dir, exist_ok=True)

def load_and_label(paths, label):
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df['layout'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Load all CSVs
fixed_df = load_and_label(fixed_path, 'fixed')
rand_df = load_and_label(rand_path, 'randomized')

# Compute rolling mean and std
window = 50  # you can adjust the smoothing window
fixed_mean = fixed_df['reward'].rolling(window).mean()
fixed_mean = fixed_mean.bfill()

rand_mean = rand_df['reward'].rolling(window).mean()
rand_mean = rand_mean.bfill()

fixed_std = fixed_df['reward'].rolling(window).std().fillna(0)
rand_std = rand_df['reward'].rolling(window).std().fillna(0)

# Plot
plt.figure(figsize=(10,6))
plt.plot(fixed_mean, label='Fixed Layout', color='blue')
plt.fill_between(fixed_mean.index, fixed_mean - fixed_std, fixed_mean + fixed_std, color='blue', alpha=0.2)

plt.plot(rand_mean, label='Randomized Layout', color='orange')
plt.fill_between(rand_mean.index, rand_mean - rand_std, rand_mean + rand_std, color='orange', alpha=0.2)

plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Smoothed Learning Curve: Fixed vs Randomized Layouts (noise 0.03)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(plot_dir, 'learning_curve_noise_0.03.png'))
plt.show()


paths = {
    0.00: os.path.join(base_dir, "setup_1_noise_0", "combined_training_log.csv"),
    0.01: os.path.join(base_dir, "setup_1_noise_0.01", "combined_training_log.csv"),
    0.03: os.path.join(base_dir, "setup_1_noise_0.03", "combined_training_log.csv")
}
# Smoothing window
window = 50

plt.figure(figsize=(10,6))

for noise, path in paths.items():
    df = pd.read_csv(path)
    mean_reward = df['reward'].rolling(window).mean().bfill()
    std_reward = df['reward'].rolling(window).std().fillna(0)
    
    plt.plot(mean_reward, label=f'Noise {noise:.2f}')
    plt.fill_between(mean_reward.index, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)

plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.title('Learning Curves for Setup 1 with Different Sensor Noise Levels')
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs(plot_dir, exist_ok=True)
plt.savefig(os.path.join(plot_dir, 'learning_curve_setup1_all_noises.png'))
plt.show()
