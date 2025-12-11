import os
import shutil

BASE = "layouts_baseline"
STAGES_ROOT = os.path.join(BASE, "stages")

os.makedirs(STAGES_ROOT, exist_ok=True)


def list_xmls(level_folder):
    """Return a sorted list of full paths to XMLs in a level folder."""
    folder = os.path.join(BASE, level_folder)
    files = [f for f in os.listdir(folder) if f.endswith(".xml")]
    files.sort()
    return [os.path.join(folder, f) for f in files]

def add_files_to_stage(stage_dir, stage_name, src_paths, start_index):
    """
    Copy XMLs into the stage directory with new names:
    e.g. stage3_001_baseline_lvl3_obs2_seed2000.xml
    """
    idx = start_index
    for src in src_paths:
        base = os.path.basename(src)
        new_name = f"{stage_name}_{idx:03d}_{base}"
        dst = os.path.join(stage_dir, new_name)
        shutil.copy(src, dst)
        idx += 1
    return idx

def make_stage_dirs():
    for s in ["stage1", "stage2", "stage3", "stage4", "stage5"]:
        path = os.path.join(STAGES_ROOT, s)
        os.makedirs(path, exist_ok=True)


def main():
    make_stage_dirs()

    L1 = list_xmls("L1")   #  9
    L2 = list_xmls("L2")    # 9
    L3 = list_xmls("L3")    # 9
    L4 = list_xmls("L4")    #  12
    L5 = list_xmls("L5")     # 12

    # Level 1: 9 xmls
    stage_name = "stage1"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:9], idx)
    print(f"[OK] {stage_name}: {idx-1} XMLs")

    # Level 1: 9, Level 2: 9
    stage_name = "stage2"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:9], idx)
    print(f"[OK] {stage_name}: {idx-1} XMLs")

    # L1: 5, L2: 9, L3: 9 (twice → 18)
    stage_name = "stage3"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:5], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L3, idx)     
    idx = add_files_to_stage(stage_dir, stage_name, L3, idx)    
    print(f"[OK] {stage_name}: {idx-1} XMLs")

    # L1: 4, L2: 5, L3: 9, L4: 12 (twice → 24)
    stage_name = "stage4"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:4], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:5], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L3[:9], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L4, idx)   
    idx = add_files_to_stage(stage_dir, stage_name, L4, idx)       
    print(f"[OK] {stage_name}: {idx-1} XMLs")

    # L1: 2, L2: 4, L3: 7, L4: 12, L5: 12 (twice → 24)
    stage_name = "stage5"
    stage_dir = os.path.join(STAGES_ROOT, stage_name)
    idx = 1

    idx = add_files_to_stage(stage_dir, stage_name, L1[:2], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L2[:4], idx)    
    idx = add_files_to_stage(stage_dir, stage_name, L3[:7], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L4[:12], idx)
    idx = add_files_to_stage(stage_dir, stage_name, L5, idx)   
    idx = add_files_to_stage(stage_dir, stage_name, L5, idx)   
    print(f"[OK] {stage_name}: {idx-1} XMLs")


if __name__ == "__main__":
    main()
