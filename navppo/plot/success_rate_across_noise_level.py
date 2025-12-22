import pandas as pd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, "..", "..",'plots')
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'runs'))

train_noises = [0, 0.01, 0.03]
eval_noises = [0.00, 0.01, 0.03]


def compute_success_rate(df):
    total_episodes = len(df)
    success_episodes = df['EpisodeStatus'].apply(lambda x: x == 'Goal_reached').sum()
    return (success_episodes / total_episodes) * 100  # percentage

results = {}
for train_noise in train_noises:
    results[train_noise] = {}
    for n in eval_noises:

        if n == 0:
            folder_name = "noise_0"
        else:
            folder_name = f"noise_0{int(n*100)}"

        path = os.path.join(base_dir, f'setup_1_noise_{train_noise}', 'eval', folder_name, 'combined.csv')
        df = pd.read_csv(path)
        results[train_noise][n] = compute_success_rate(df)

df_rates = pd.DataFrame(results).T  # train_noise as rows, eval_noise as columns

plt.figure(figsize=(8,6))
for train_noise in df_rates.index:
    plt.plot(df_rates.columns, df_rates.loc[train_noise], marker='o', label=f'Train Noise {train_noise}')

plt.xlabel('Evaluation Noise Level')
plt.ylabel('Success Rate (%)')
plt.title('Setup 1 - Policy Generalization Across Evaluation Noise Levels')
plt.ylim(0, 100)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'setup_2_success_rate_comparison.png'))
plt.show()
# After df_rates = pd.DataFrame(results).T
print("\nSuccess Rates (%)")
print(df_rates.round(2))
