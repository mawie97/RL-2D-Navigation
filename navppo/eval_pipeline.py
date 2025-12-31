import os
import re
import glob
from io import StringIO
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STATUS_SUCCESS = {"goal_reached"}  # after normalization

EVAL_NOISE_LEVELS = [0.0, 0.01, 0.03]

TRAIN_NOISE_LEVELS = [0.0, 0.01, 0.03]

# Evaluation sets folder names lvl_1_4, lvl_1_5, lvl_5
EVAL_SET_ORDER = ["lvl_1_5", "lvl_1_4", "lvl_5"]

STRATEGY_ORDER = [
    "ours_l1l5_solverl5",
    "naive_random",
    "random_bresenham",
    "l1l4_baseline",
]

STRATEGY_LABEL = {
    "ours_l1l5_solverl5": "Ours (Hybrid / L1–L5)",
    "naive_random": "Naive random",
    "random_bresenham": "Random (Bresenham)",
    "l1l4_baseline": "L1–L4 baseline",
}

def normalize_status(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w]+$", "", s)
    return s.lower()

NOISE_CODE_MAP = {
    "0": 0.0,
    "1": 0.01,
    "3": 0.03,
    "0.0": 0.0,
    "0.01": 0.01,
    "0.03": 0.03,
    "01": 0.01,
    "03": 0.03,
}

def parse_noise_token(token: str) -> float:
    token = token.lower()
    m = re.match(r"noise[_-]?([0-9]*\.?[0-9]+)$", token)
    if not m:
        raise ValueError(f"Noise token not recognized: {token}")
    raw = m.group(1)
    if raw in NOISE_CODE_MAP:
        return NOISE_CODE_MAP[raw]
    raise ValueError(f"Noise value '{raw}' not supported. Update NOISE_CODE_MAP.")


def parse_episode_status_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "episode" not in df.columns or "status" not in df.columns:
        raise ValueError(f"{fp} must contain columns Episode,Status. Got: {list(df.columns)}")

    df["episode"] = df["episode"].astype(int)
    df["status"] = df["status"].astype(str)
    df["status_norm"] = df["status"].apply(normalize_status)
    df["is_success"] = df["status_norm"].isin(STATUS_SUCCESS)

    return df[["episode", "status", "status_norm", "is_success"]].sort_values("episode").reset_index(drop=True)


def parse_monitor_csv(fp: str) -> pd.DataFrame:
    """
    monitor.csv:
    """
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        for ln in f.readlines():
            if ln.strip() == "":
                continue
            if ln.lstrip().startswith("#"):
                continue
            lines.append(ln)

    if not lines:
        raise ValueError(f"{fp} is empty after removing comment/blank lines")

    df = pd.read_csv(StringIO("".join(lines)))
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    req = {"r", "l", "t"}
    if not req.issubset(df.columns):
        raise ValueError(f"{fp} must contain columns {req}. Got: {list(df.columns)}")

    df = df.rename(columns={"r": "reward", "l": "length", "t": "t"})
    df["episode"] = np.arange(1, len(df) + 1, dtype=int)

    return df[["episode", "reward", "length", "t"]]


def merge_eval_dir(eval_dir: str) -> pd.DataFrame:
    """
    Merge eval_log_episodes.csv + monitor.csv inside one eval directory.
    """
    ep_fp = os.path.join(eval_dir, "eval_log_episodes.csv")
    mon_fp = os.path.join(eval_dir, "monitor.csv")

    if not os.path.exists(ep_fp):
        raise FileNotFoundError(f"Missing {ep_fp}")
    if not os.path.exists(mon_fp):
        raise FileNotFoundError(f"Missing {mon_fp}")

    ep = parse_episode_status_csv(ep_fp)
    mon = parse_monitor_csv(mon_fp)

    merged = ep.merge(mon, on="episode", how="inner")
    if len(merged) == 0:
        raise ValueError(f"0 merged rows in {eval_dir} — check episode numbering")

    if len(ep) != len(mon):
        print(f"[WARN] Episode count mismatch in {eval_dir}: status={len(ep)} monitor={len(mon)} overlap={len(merged)}")

    return merged

def infer_strategy_from_model_id(model_id: str) -> str:
    mid = model_id.lower()

    # 1) 先识别 bresenham
    if "bresenham" in mid:
        return "random_bresenham"

    # 2) 再识别 naive_random（避免误伤 random_bresenham）
    if "naive_random" in mid or ("naive" in mid and "random" in mid):
        return "naive_random"

    # 3) hybrid / hybird
    if "hybird" in mid or "hybrid" in mid:
        return "ours_l1l5_solverl5"

    # 4) l1l4
    if "l1_l4" in mid or "l1l4" in mid:
        return "l1l4_baseline"

    return "unknown"


def parse_meta_from_eval_dir(eval_dir: str) -> Dict:
    """
    eval_dir:
      runs/<model_name>/eval/noise<eval_noise>/<eval_set>/
    """
    p = os.path.normpath(eval_dir)
    parts = p.split(os.sep)

    if "runs" not in parts:
        raise ValueError(f"'runs' not in path: {eval_dir}")
    i_runs = parts.index("runs")
    model_id = parts[i_runs + 1]
    if "eval" not in parts[i_runs + 2:]:
        raise ValueError(f"'eval' not in path after model folder: {eval_dir}")
    i_eval = parts.index("eval", i_runs + 2)

    noise_folder = parts[i_eval + 1]
    eval_set = parts[i_eval + 2]

    eval_noise = parse_noise_token(noise_folder)

    m = re.search(r"noise([0-9]*\.?[0-9]+)$", model_id.lower())
    train_noise = parse_noise_token("noise" + m.group(1)) if m else np.nan


    strategy = infer_strategy_from_model_id(model_id)

    return {
        "model_id": model_id,
        "strategy": strategy,
        "train_noise": train_noise,
        "eval_noise": eval_noise,
        "eval_set": eval_set,
    }

def load_all_evaluations(runs_root: str = "runs") -> pd.DataFrame:
    """
    Scan:
      runs/*/eval/noise*/<eval_set>/
    and merge eval_log_episodes.csv + monitor.csv for each folder.
    """
    pattern = os.path.join(runs_root, "*", "eval", "noise*", "*")
    eval_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    if not eval_dirs:
        raise FileNotFoundError(f"No eval directories found with pattern: {pattern}")

    all_rows = []
    for d in eval_dirs:
        if not os.path.exists(os.path.join(d, "eval_log_episodes.csv")):
            continue
        df = merge_eval_dir(d)
        meta = parse_meta_from_eval_dir(d)
        for k, v in meta.items():
            df[k] = v
        all_rows.append(df)

    if not all_rows:
        raise FileNotFoundError("Found eval dirs, but none contained eval_log_episodes.csv")

    return pd.concat(all_rows, ignore_index=True)

def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 123) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (values[0], values[0])

    rng = np.random.default_rng(seed)
    n = len(values)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = rng.choice(values, size=n, replace=True).mean()
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return lo, hi


def condition_summary_with_ci(df: pd.DataFrame) -> pd.DataFrame:

    keys = ["strategy", "train_noise", "eval_noise", "eval_set"]
    rows = []
    for key, grp in df.groupby(keys):
        succ = grp["is_success"].astype(float).to_numpy()
        mean = float(np.mean(succ))
        lo, hi = bootstrap_ci_mean(succ)
        rows.append(dict(zip(keys, key), mean=mean, ci_lo=lo, ci_hi=hi, n_ep=len(grp)))
    return pd.DataFrame(rows)


# -----------------------------
# 5) Plot 1: 4 curves per train noise
# -----------------------------

def plot_success_vs_eval_noise(df: pd.DataFrame, train_noise: float, eval_set: str, savepath: str = None):
    summ = condition_summary_with_ci(df[(df["train_noise"] == train_noise) & (df["eval_set"] == eval_set)])

    plt.figure(figsize=(7.2, 4.2))
    for strat in STRATEGY_ORDER:
        s = summ[summ["strategy"] == strat].sort_values("eval_noise")
        if s.empty:
            continue
        plt.plot(s["eval_noise"], s["mean"], marker="o", label=STRATEGY_LABEL.get(strat, strat))
        plt.fill_between(s["eval_noise"], s["ci_lo"], s["ci_hi"], alpha=0.15)

    plt.xticks(EVAL_NOISE_LEVELS, [str(x) for x in EVAL_NOISE_LEVELS])
    plt.ylim(0, 1)
    plt.xlabel("Evaluation sensor noise (σ)")
    plt.ylabel("Success rate")
    plt.title(f"Success rate vs eval noise (train σ={train_noise}, set={eval_set})")
    plt.legend(frameon=False)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


# -----------------------------
# 6) Plot 2: grouped bars (3 bars per strategy)
# -----------------------------

def plot_success_bar_by_strategy(df: pd.DataFrame, train_noise: float, eval_set: str, savepath: str = None):
    summ = condition_summary_with_ci(df[(df["train_noise"] == train_noise) & (df["eval_set"] == eval_set)])

    x = np.arange(len(STRATEGY_ORDER))
    width = 0.25

    plt.figure(figsize=(8.0, 4.2))
    for j, enoise in enumerate(EVAL_NOISE_LEVELS):
        means, loerr, hierr = [], [], []
        for strat in STRATEGY_ORDER:
            row = summ[(summ["strategy"] == strat) & (summ["eval_noise"] == enoise)]
            if row.empty:
                means.append(np.nan); loerr.append(np.nan); hierr.append(np.nan)
            else:
                m = float(row["mean"].iloc[0])
                lo = float(row["ci_lo"].iloc[0])
                hi = float(row["ci_hi"].iloc[0])
                means.append(m); loerr.append(m - lo); hierr.append(hi - m)

        yerr = np.vstack([loerr, hierr])
        plt.bar(x + (j - 1) * width, means, width=width, yerr=yerr, capsize=3, label=f"Eval σ={enoise}")

    plt.xticks(x, [STRATEGY_LABEL[s] for s in STRATEGY_ORDER], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Success rate")
    plt.title(f"Success rate by strategy (train σ={train_noise}, set={eval_set})")
    plt.legend(frameon=False)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


# -----------------------------
# 7) Stats: paired permutation test (episode-level)
# -----------------------------

def paired_signflip_test(df: pd.DataFrame, cond: Dict, strategy_a: str, strategy_b: str,
                         n_perm: int = 10000, seed: int = 0) -> Tuple[float, float, int]:
    """
    Paired sign-flip test on episode-level success indicators.
    Pairing assumes both strategies have same number of episodes under the condition.
    (Best is scenario-level pairing, but you don't have xml_id yet.)
    Returns: (delta_mean, p_value, n_paired)
    """
    d = df.copy()
    for k, v in cond.items():
        d = d[d[k] == v]

    a = d[d["strategy"] == strategy_a]["is_success"].astype(float).to_numpy()
    b = d[d["strategy"] == strategy_b]["is_success"].astype(float).to_numpy()

    n = min(len(a), len(b))
    if n < 10:
        raise ValueError(f"Too few paired episodes (n={n}) for test under cond={cond}")

    a = a[:n]; b = b[:n]
    diff = a - b
    obs = float(diff.mean())

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1, 1], size=(n_perm, n), replace=True)
    perm_means = (signs * diff).mean(axis=1)

    p = (np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p, n

# -----------------------------
# 8) Main: run everything
# -----------------------------

def main():
    df = load_all_evaluations("runs")

    # Quick sanity
    print("Loaded rows:", len(df))
    print(df[["model_id", "strategy", "train_noise"]]
          .drop_duplicates()
          .sort_values(["strategy", "train_noise"]))

    # 选你要画图的 eval set
    main_eval_set = "lvl_1_4"  # or lvl_5 / lvl_1_5

    # 画图：每个 train_noise 一张曲线图 + 一张柱状图
    for tn in TRAIN_NOISE_LEVELS:
        plot_success_vs_eval_noise(
            df, train_noise=tn, eval_set=main_eval_set,
            savepath=f"fig_success_line_{main_eval_set}_train{tn}.png"
        )
        plot_success_bar_by_strategy(
            df, train_noise=tn, eval_set=main_eval_set,
            savepath=f"fig_success_bar_{main_eval_set}_train{tn}.png"
        )

    # 单一对照检验：L1–L5 vs L1–L4，在 lvl_5 set，eval_noise=0.01，train_noise=0.01
    cond = {"train_noise": 0.01, "eval_noise": 0.01, "eval_set": "lvl_5"}
    delta, p, n = paired_signflip_test(
        df,
        cond=cond,
        strategy_a="ours_l1l5_solverl5",
        strategy_b="l1l4_baseline"
    )

    print("\nPaired sign-flip test (episode-level)")
    print("Condition:", cond)
    print(f"Δ(success rate) = {delta:+.3f}, p = {p:.4g}, n_paired = {n}")


if __name__ == "__main__":
    main()
