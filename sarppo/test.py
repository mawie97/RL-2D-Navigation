import mujoco
import numpy as np

MODEL_PATH = "test.xml"

# ---------- load model ----------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# helpers to get indices
def joint_qpos_adr(joint_name: str) -> int:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return model.jnt_qposadr[j_id]

def actuator_for_joint(joint_name: str) -> int:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    for act_id in range(model.nu):
        if model.actuator_trnid[act_id, 0] == j_id:
            return act_id
    raise ValueError(f"No actuator found for joint {joint_name}")

# joint and actuator indices
qadr_x = joint_qpos_adr("j1")
qadr_y = joint_qpos_adr("j2")

act_x = actuator_for_joint("j1")
act_y = actuator_for_joint("j2")

# ---------- set up step test ----------
dt = model.opt.timestep
reach_tol = 0.01       # you can set 0.001 here to see the "13000" type behaviour
max_steps = 20000      # upper bound on steps
delta = np.array([0.1, 0.1], dtype=np.float64)

# initial state
mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)

start_xy = np.array([data.qpos[qadr_x], data.qpos[qadr_y]], dtype=np.float64)
target_xy = start_xy + delta

# PID plugin treats ctrl as desired joint positions
data.ctrl[act_x] = target_xy[0]
data.ctrl[act_y] = target_xy[1]

print("=== PID step test ===")
print(f"kp, kd are set in XML (pid1 instance).")
print(f"timestep = {dt}")
print(f"start_xy  = {start_xy}")
print(f"target_xy = {target_xy}")
print(f"reach_tol = {reach_tol}")
print("Running simulation...")

# direction along start -> target, for overshoot check
direction = target_xy - start_xy
dist_total = float(np.linalg.norm(direction))
if dist_total < 1e-12:
    raise RuntimeError("start and target are the same")

u = direction / dist_total  # unit direction

first_reach_step = None
max_overshoot_after_reach = 0.0

traj = []

for step in range(max_steps):
    mujoco.mj_step(model, data)

    cur_xy = np.array([data.qpos[qadr_x], data.qpos[qadr_y]], dtype=np.float64)
    traj.append(cur_xy.copy())

    dist_to_target = float(np.linalg.norm(cur_xy - target_xy))
    s = float(np.dot(cur_xy - start_xy, u))  # progress along line
    overshoot = max(0.0, s - dist_total)

    # detect first time we are at target
    if first_reach_step is None and dist_to_target <= reach_tol:
        first_reach_step = step
        print(f"\nFirst reached target within tol at step {step}, t={step * dt:.4f}s")
        print(f"pos = {cur_xy}, dist_to_target={dist_to_target:.6f}")

    # after first reach, check overshoot
    if first_reach_step is not None and step > first_reach_step:
        if overshoot > max_overshoot_after_reach:
            max_overshoot_after_reach = overshoot

    # print a few samples for inspection
    if step % 1000 == 0:
        print(f"step={step:5d} t={step * dt:7.3f}s  pos=({cur_xy[0]: .4f}, {cur_xy[1]: .4f})  "
              f"dist_to_target={dist_to_target:.6f}  overshoot={overshoot:.6f}")

# ---------- summary ----------
print("\n=== Summary ===")
print(f"Simulated time: {max_steps * dt:.3f}s ({max_steps} steps)")

if first_reach_step is None:
    print("Did NOT reach target within reach_tol in this time.")
else:
    print(f"First reach step: {first_reach_step}, t={first_reach_step * dt:.4f}s")
    print(f"Max overshoot AFTER reach (along motion direction): {max_overshoot_after_reach:.6f}")
    if max_overshoot_after_reach > 0.0:
        print("=> There IS overshoot after first reaching the target.")
    else:
        print("=> No overshoot after first reaching the target (within numerical precision).")
