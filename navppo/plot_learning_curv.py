import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

A_DIR = "/Users/susu/Desktop/Thesis_Project/navppo/runs/new_random_bresenham_noise0/logs"
B_DIR = "/Users/susu/Desktop/Thesis_Project/navppo/runs/new_l1_l4_noise0/logs"

LABEL_A = "Non-Curriculum (L1-L4)"
LABEL_B = "Curriculum (L1–L4)"

TAGS_TO_PLOT = [
    ("rollout/ep_len_mean", "Mean episode length", "fig_learning_curve_ep_len_mean.png"),
    ("rollout/ep_rew_mean", "Mean episode reward", "fig_learning_curve_ep_rew_mean.png"),
]

def find_latest_event_file(log_dir: str) -> str:
    files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {log_dir}")
    return max(files, key=os.path.getmtime)

def load_ea(event_file: str):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    return ea, ea.Tags().get("scalars", [])

def extract_series(ea, tag: str):
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    vals  = np.array([e.value for e in events], dtype=float)
    return steps, vals

def plot_two_runs(tag: str, y_label: str, out_png: str):
    a_event = find_latest_event_file(A_DIR)
    b_event = find_latest_event_file(B_DIR)

    ea_a, tags_a = load_ea(a_event)
    ea_b, tags_b = load_ea(b_event)

    if tag not in tags_a:
        raise ValueError(f"Tag '{tag}' not found in A.\nAvailable tags:\n{tags_a}")
    if tag not in tags_b:
        raise ValueError(f"Tag '{tag}' not found in B.\nAvailable tags:\n{tags_b}")

    xa, ya = extract_series(ea_a, tag)
    xb, yb = extract_series(ea_b, tag)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(xa, ya, label=LABEL_A)
    plt.plot(xb, yb, label=LABEL_B)

    plt.xlabel("Training timesteps")
    plt.ylabel(y_label)
    plt.title(y_label + " during training")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()

    print("Saved:", out_png)
    print("A event:", a_event)
    print("B event:", b_event)

def main():
    for tag, y_label, out_png in TAGS_TO_PLOT:
        plot_two_runs(tag, y_label, out_png)

if __name__ == "__main__":
    main()
