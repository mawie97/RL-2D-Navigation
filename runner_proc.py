import os
import random
import mujoco
from mujoco import viewer
from generator_proc import ProceduralScenarioGenerator, GridSpec

if __name__ == "__main__":
    H, W = 20, 20
    root = (1, 1)
    rng = random.Random(7)

    gen = ProceduralScenarioGenerator(GridSpec(H=H, W=W))

    # Fast, scalable generation with BFS guarantees:
    grid = gen.generate_with_requirements(
        root=root,
        min_walls=80, max_walls=100,   # density band
        min_corridor=12,                # guarantee a long corridor (approx diameter >= 12)
        min_deadends=1,                 # at least 6 dead-ends
        min_deadend_depth=4,            # of depth >= 4 from root
        rng=rng,
        max_tries=200
    )

    # ASCII preview
    print("Preview:")
    for r in range(H):
        print("".join("#" if grid[r][c] else "." for c in range(W)))

    # (Connectivity already checked inside generate_with_requirements)
    xml = gen.grid_to_mujoco_xml_base_compatible(
        grid, model_name=f"connected_{H}x{W}"
    )
    path = f"scenarios/connected_{H}x{W}.xml"
    gen.write_xml(xml, path)
    print(f"Saved {path}")

    # model = mujoco.MjModel.from_xml_path(path)
    # data  = mujoco.MjData(model)
    # viewer.launch(model, data)

