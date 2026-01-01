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
    # "naive_random",
    # "random_bresenham",
    "l1l4_baseline",
]

STRATEGY_LABEL = {
    "ours_l1l5_solverl5": "L1-L5 (Hybrid)",
    # "naive_random": "Naive random",
    # "random_bresenham": "Bresenham random",
    "l1l4_baseline": "L1–L4 (Procedural)",
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

    # detect eval_set from path (folder name right after noise*)
    # runs/<model>/eval/noiseX/<eval_set>/
    eval_set = os.path.normpath(eval_dir).split(os.sep)[-1]

    # default: all episodes in one bucket
    merged["case"] = "all"

    if eval_set == "lvl_5":
        merged["case"] = np.where(merged["episode"] <= 50, "corridor", "deadend")

    if len(ep) != len(mon):
        print(f"[WARN] Episode count mismatch in {eval_dir}: status={len(ep)} monitor={len(mon)} overlap={len(merged)}")

    return merged


def infer_strategy_from_model_id(model_id: str) -> str:
    mid = model_id.lower()
    
    base = re.sub(r"[_-]?noise[0-9]*\.?[0-9]+$", "", mid)
    
    STRATEGY_MAP = {
        "new_hybird": "ours_l1l5_solverl5",
        "new_naive_random": "naive_random",
        "new_random_bresenham": "random_bresenham",
        "new_l1_l4": "l1l4_baseline",
    }

    if base in STRATEGY_MAP:
        return STRATEGY_MAP[base]

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
    keys = ["strategy", "train_noise", "eval_noise", "eval_set", "case"]  # <- add case
    rows = []
    for key, grp in df.groupby(keys):
        succ = grp["is_success"].astype(float).to_numpy()
        mean = float(np.mean(succ))
        lo, hi = bootstrap_ci_mean(succ)
        rows.append(dict(zip(keys, key), mean=mean, ci_lo=lo, ci_hi=hi, n_ep=len(grp)))
    return pd.DataFrame(rows)


def plot_success_vs_eval_noise(df: pd.DataFrame, train_noise: float, eval_set: str, case_name: str = "all", savepath: str = None):
    sub = df[(df["train_noise"] == train_noise) & (df["eval_set"] == eval_set) & (df["case"] == case_name)]
    summ = condition_summary_with_ci(sub)

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
    plt.title(f"Success Rate vs Eval Noise")
    plt.legend(frameon=False)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()

    return summ

def plot_success_bar_by_strategy(
    df: pd.DataFrame,
    train_noise: float,
    eval_set: str,
    case_name: str,          # <- new
    savepath: str = None
):
    sub = df[
        (df["train_noise"] == train_noise) &
        (df["eval_set"] == eval_set) &
        (df["case"] == case_name)        # <- filter case
    ]
    summ = condition_summary_with_ci(sub)

    x = np.arange(len(STRATEGY_ORDER))
    width = 0.25

    plt.figure(figsize=(8.0, 4.2))
    for j, enoise in enumerate(EVAL_NOISE_LEVELS):
        means, loerr, hierr = [], [], []
        for strat in STRATEGY_ORDER:
            row = summ[
                (summ["strategy"] == strat) &
                (summ["eval_noise"] == enoise) &
                (summ["case"] == case_name)   # <- ensure case match
            ]
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
    plt.title(f"Success Rate by Strategy ({case_name})")
    plt.legend(frameon=False)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()

    return summ

def make_table_for_overleaf(
    summ: pd.DataFrame,
    train_noise: float,
    eval_set: str,
    case_name: str,
    strategies: List[str],
    eval_noises: List[float],
    out_prefix: str,
):
    """
    Creates:
      1) a "long" CSV: one row per (strategy, eval_noise)
      2) a "wide" CSV: one row per strategy, columns for each eval_noise
      3) a LaTeX tabular (printed + saved as .tex)
    Values shown as: mean [ci_lo, ci_hi]
    """
    sub = summ[
        (summ["train_noise"] == train_noise) &
        (summ["eval_set"] == eval_set) &
        (summ["case"] == case_name) &
        (summ["strategy"].isin(strategies)) &
        (summ["eval_noise"].isin(eval_noises))
    ].copy()

    # ---- long format CSV (good for debugging) ----
    long_csv = f"{out_prefix}_long.csv"
    sub.sort_values(["strategy", "eval_noise"]).to_csv(long_csv, index=False)

    # ---- make a pretty cell string ----
    def fmt_row(r):
        return f'{r["mean"]:.3f} [{r["ci_lo"]:.3f}, {r["ci_hi"]:.3f}]'

    sub["cell"] = sub.apply(fmt_row, axis=1)

    # ---- wide format ----
    wide = sub.pivot_table(index="strategy", columns="eval_noise", values="cell", aggfunc="first")
    wide = wide.reindex(strategies)
    wide = wide.rename(index=STRATEGY_LABEL)  # use nice names in the table

    # rename columns to "σ=..."
    wide.columns = [f"$\\sigma={c}$" for c in wide.columns]

    wide_csv = f"{out_prefix}_wide.csv"
    wide.to_csv(wide_csv)

    # ---- LaTeX ----
    latex = wide.to_latex(
        escape=False,
        na_rep="--",
        column_format="l" + "c" * len(wide.columns),
    )

    tex_path = f"{out_prefix}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print("\n===== LaTeX table (copy to Overleaf) =====")
    print(latex)
    print("Saved:", long_csv, wide_csv, tex_path)


def paired_signflip_test(df: pd.DataFrame, cond: Dict, strategy_a: str, strategy_b: str,
                         n_perm: int = 10000, seed: int = 0) -> Tuple[float, float, int]:
    """
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

def cases_for_eval_set(eval_set: str) -> List[str]:
    return ["corridor", "deadend"] if eval_set == "lvl_5" else ["all"]

def main():
    df = load_all_evaluations("runs")

    main_eval_set = "lvl_5"

    for tn in TRAIN_NOISE_LEVELS:
         for case_name in cases_for_eval_set(main_eval_set):
            summ_line = plot_success_vs_eval_noise(
                df,
                train_noise=tn,
                eval_set=main_eval_set,
                case_name=case_name,
                savepath=f"fig_success_line_{main_eval_set}_{case_name}_train{tn}.png"
            )
            # --- export table for the same condition ---
            out_prefix = f"table_success_{main_eval_set}_{case_name}_train{tn}".replace(".", "p")
            make_table_for_overleaf(
                summ=summ_line,
                train_noise=tn,
                eval_set=main_eval_set,
                case_name=case_name,
                strategies=STRATEGY_ORDER,
                eval_noises=EVAL_NOISE_LEVELS,
                out_prefix=out_prefix,
            )

            plot_success_bar_by_strategy(
                df, 
                train_noise=tn, 
                eval_set=main_eval_set,
                case_name=case_name,
                savepath=f"fig_success_bar_{main_eval_set}_{case_name}_train{tn}.png"
            )


# def main():
#     df = load_all_evaluations("runs")

#     print("Loaded rows:", len(df))
#     print(df[["model_id", "strategy", "train_noise"]]
#           .drop_duplicates()
#           .sort_values(["strategy", "train_noise"]))

#     # choose eval set
#     main_eval_set = "lvl_1_5"   # lvl_1_4 lvl_5 lvl_1_5

#     for tn in TRAIN_NOISE_LEVELS:
#         plot_success_vs_eval_noise(
#             df, train_noise=tn, eval_set=main_eval_set,
#             savepath=f"fig_success_line_{main_eval_set}_train{tn}.png"
#         )
#         # plot_success_bar_by_strategy(
#         #     df, train_noise=tn, eval_set=main_eval_set,
#         #     savepath=f"fig_success_bar_{main_eval_set}_train{tn}.png"
#         # )

#     # L1–L5 vs L1–L4，在 lvl_5 set，eval_noise=0.01，train_noise=0.01
#     cond = {"train_noise": 0.01, "eval_noise": 0.01, "eval_set": "lvl_5"}
#     delta, p, n = paired_signflip_test(
#         df,
#         cond=cond,
#         strategy_a="ours_l1l5_solverl5",
#         strategy_b="naive_random"
#     )

#     print("\nPaired sign-flip test (episode-level)")
#     print("Condition:", cond)
#     print(f"Δ(success rate) = {delta:+.3f}, p = {p:.4g}, n_paired = {n}")


if __name__ == "__main__":
    main()
