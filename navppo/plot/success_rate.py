import pandas as pd
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, "..", "..",'plots')
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'runs'))
fixed_path = os.path.join(base_dir, "setup_1_noise_0", "eval","noise_0", "combined.csv")
randomized_path = os.path.join(base_dir, "setup_2_noise_0", "eval","noise_0", "combined.csv")

# Load datasets
df_fixed = pd.read_csv(fixed_path)
df_random = pd.read_csv(randomized_path)

# Compute rates function
def compute_rates(df, group_size=10):
    df['Success'] = df['EpisodeStatus'].apply(lambda x: 1 if x == 'Goal_reached' else 0)
    df['Collision'] = df['EpisodeStatus'].apply(lambda x: 1 if x == 'Collision' else 0)  # use your collision/failure condition
    df['Group'] = (df['Step'] - 1) // group_size + 1
    grouped = df.groupby('Group').agg(
        SuccessRate=('Success','mean'),
        CollisionRate=('Collision','mean')
    ) * 100  # percentage
    return grouped

group_size = 10
fixed_rates = compute_rates(df_fixed, group_size)
random_rates = compute_rates(df_random, group_size)

# Plot line chart
plt.figure(figsize=(12,6))

plt.plot(fixed_rates.index, fixed_rates['SuccessRate'], marker='o', linestyle='-', color='green', label='Fixed Success')
plt.plot(fixed_rates.index, fixed_rates['CollisionRate'], marker='x', linestyle='--', color='red', label='Fixed Collision')

plt.plot(random_rates.index, random_rates['SuccessRate'], marker='o', linestyle='-', color='blue', label='Randomized Success')
plt.plot(random_rates.index, random_rates['CollisionRate'], marker='x', linestyle='--', color='orange', label='Randomized Collision')

plt.xticks(fixed_rates.index)
plt.xlabel(f'Group (every {group_size} layout)')
plt.ylabel('Rate (%)')
plt.title('Success and Collision Rate for Fixed vs Randomized Layouts\nTrained Model per Layout (noise = 0)')
plt.ylim(-10, 110)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'success_rate_comparison1.png'))
plt.show()
