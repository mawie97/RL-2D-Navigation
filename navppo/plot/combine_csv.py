import os
import pandas as pd

# current_dir = os.path.dirname(__file__)
# base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'runs', 'setup_1_noise_0.03'))
# self_log_path = os.path.join(base_dir, "self_log.csv")
# monitor_path = os.path.join(base_dir, "logs", "monitor.csv")

# self_log = pd.read_csv(self_log_path, skiprows=1, header=None, names=["episode", "status"])
# monitor = pd.read_csv(monitor_path, skiprows=2, names=["reward", "length", "time"])

# combined = pd.concat([self_log, monitor], axis=1)

# combined_path = os.path.join(base_dir, "combined_training_log.csv")
# combined.to_csv(combined_path, index=False)
# print(f"Combined CSV saved as {combined_path}")


current_dir = os.path.dirname(__file__)
base_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'runs'))
fixed_path = os.path.join(base_dir, "setup_1_noise_0.01", "eval","noise_03","log")
output_path = os.path.join(base_dir, "setup_1_noise_0.01", "eval","noise_03")
value_data_file = os.path.join(fixed_path, "csv.csv")
status_data_file = os.path.join(fixed_path,"eval_log.csv")

df_values = pd.read_csv(value_data_file)       # Wall time, Step, Value
df_status = pd.read_csv(status_data_file)      # EpisodeStatus only


min_len = min(len(df_values), len(df_status))
df_values = df_values.iloc[:min_len]
df_status = df_status.iloc[:min_len]

# Combine by column
df_combined = pd.concat([df_values, df_status.reset_index(drop=True)], axis=1)
df_combined.to_csv(os.path.join(output_path, "combined.csv"), index=False)



