import os
import re
import glob
import shutil

BASE_ROOT = "layouts_baseline"
# BASE_ROOT = "layouts_scenerio"
STAGES_ROOT = os.path.join(BASE_ROOT, "stages")


def sorted_xmls(level_dir: str):
    pattern = os.path.join(level_dir, "*.xml")
    return sorted(glob.glob(pattern))


def _safe_name(name: str) -> str:
    # avoid spaces
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def add_files_to_stage(stage_dir: str, stage_name: str, src_paths, start_idx: int):
    """
    Copy XML files into stage_dir with names that preserve:
      - global stage order index
      - stage name
      - original filename 
    """
    os.makedirs(stage_dir, exist_ok=True)
    idx = start_idx

    for src in src_paths:
        src_base = os.path.splitext(os.path.basename(src))[0]
        src_base = _safe_name(src_base)

        # Example:
        # stage3_01__baseline_lvl3_obs5_seed2002.xml
        dst_name = f"{stage_name}_{idx:02d}__{src_base}.xml"
        dst = os.path.join(stage_dir, dst_name)

        shutil.copy2(src, dst)
        print(f"[STAGE] {stage_name}: {src} -> {dst}")
        idx += 1

    return idx


def merge_all_stages_into_one(
    stages_root: str = STAGES_ROOT,
    out_dir: str = os.path.join(STAGES_ROOT, "ALL"),
    stage_order=("stage1", "stage2", "stage3", "stage4", "stage5"),
):
    """
    Merge all stage XMLs into ONE folder in order.
    """
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    idx = 1
    for stage in stage_order:
        stage_dir = os.path.join(stages_root, stage)
        files = sorted(glob.glob(os.path.join(stage_dir, "*.xml")))

        for src in files:
            base = os.path.splitext(os.path.basename(src))[0]
            base = _safe_name(base)

            # Example:
            # all_0001__stage3__stage3_01__baseline_lvl1_obs0_seed0.xml
            dst = os.path.join(out_dir, f"all_{idx:04d}__{stage}__{base}.xml")
            shutil.copy2(src, dst)
            idx += 1

    print(f"[DONE] Merged {idx-1} XMLs into: {out_dir}")


def main():
    L1_dir = os.path.join(BASE_ROOT, "L1")
    L2_dir = os.path.join(BASE_ROOT, "L2")
    L3_dir = os.path.join(BASE_ROOT, "L3")
    L4_dir = os.path.join(BASE_ROOT, "L4")
    L5_dir = os.path.join(BASE_ROOT, "L5")

    L1 = sorted_xmls(L1_dir)
    L2 = sorted_xmls(L2_dir)
    L3 = sorted_xmls(L3_dir)
    L4 = sorted_xmls(L4_dir)
    L5 = sorted_xmls(L5_dir)

    print(f"L1: {len(L1)} files")
    print(f"L2: {len(L2)} files")
    print(f"L3: {len(L3)} files")
    print(f"L4: {len(L4)} files")
    print(f"L5: {len(L5)} files")

    os.makedirs(STAGES_ROOT, exist_ok=True)

    # ---------- STAGE 1 ----------
    # Level 1: all 9 XMLs
    stage_name = "stage1"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    idx = 1
    idx = add_files_to_stage(stage_dir, stage_name, L1[:9], idx)

    # ---------- STAGE 2 ----------
    # Level 1: 9 XMLs
    # Level 2: 9 XMLs
    stage_name = "stage2"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    idx = 1
    idx = add_files_to_stage(stage_dir, stage_name, L1[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:9], idx)

    # ---------- STAGE 3 ----------
    # 5 from L1, 9 from L2, 9 from L3 repeated twice (same exact XML files)
    stage_name = "stage3"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:5], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L3[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L3[:9], idx)

    # ---------- STAGE 4 ----------
    # 4 from L1, 5 from L2, 9 from L3, 12 from L4 repeated twice (same exact XML files)
    stage_name = "stage4"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:4], idx) 
    idx = add_files_to_stage(stage_dir, stage_name, L2[:5], idx)    
    idx = add_files_to_stage(stage_dir, stage_name, L3[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L4[:12], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L4[:12], idx) 

    # ---------- STAGE 5 ----------
    # 2 from L1, 4 from L2, 7 from L3, 12 from L4, 12 from L5 repeated twice (same exact XML files)
    stage_name = "stage5"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    if os.path.exists(stage_dir):
        shutil.rmtree(stage_dir)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:2], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:4], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L3[:7], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L4[:12], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L5[:12], idx) 
    idx = add_files_to_stage(stage_dir, stage_name, L5[:12], idx)    

    print("[DONE] All stages built under:", STAGES_ROOT)

    merge_all_stages_into_one()


if __name__ == "__main__":
    main()
