# Replay script for demonstrations

# python replay_demo.py --data ./demonstrations/soccer_drill_run.npz \
#     --env LowerT1GoaliePenaltyKick-v0 --speed 4

import numpy as np
import gymnasium as gym
import time
import argparse
import inspect
import sai_mujoco.utils.v0.binding_utils as bu
import mujoco

# ==========================================
# 1. PATCHING
# ==========================================
def make_patched_method(original_method):
    def patched(self, name):
        try: return original_method(self, name)
        except ValueError:
            try: return original_method(self, f"/{name}")
            except ValueError: raise
    return patched

def apply_patches():
    TargetClass = None
    for name, obj in inspect.getmembers(bu):
        if inspect.isclass(obj) and hasattr(obj, 'body_name2id'):
            TargetClass = obj
            break
    if TargetClass:
        for m in ['body_name2id', 'geom_name2id', 'joint_name2id', 'site_name2id', 'sensor_name2id', 'actuator_name2id']:
            if hasattr(TargetClass, m): setattr(TargetClass, m, make_patched_method(getattr(TargetClass, m)))

apply_patches()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_mujoco_wrappers(env):
    base = env.unwrapped
    if hasattr(base, "model") and hasattr(base, "data"): return base.model, base.data
    if hasattr(base, "sim"): return base.sim.model, base.sim.data
    raise ValueError("Could not access MuJoCo data")

def get_raw_objects(model_wrap, data_wrap):
    m = model_wrap.ptr if hasattr(model_wrap, "ptr") else model_wrap._model if hasattr(model_wrap, "_model") else model_wrap
    d = data_wrap.ptr if hasattr(data_wrap, "ptr") else data_wrap._data if hasattr(data_wrap, "_data") else data_wrap
    return m, d

def normalize_quaternion(q):
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-6 else q

# ==========================================
# 3. REPLAY LOGIC
# ==========================================
def replay_episode(data_path, env_id, speed_multiplier):
    print(f"\n📂 Loading demonstration: {data_path}")
    try:
        demo_data = np.load(data_path)
    except FileNotFoundError: return
    
    qpos_traj = demo_data['qpos']
    n_frames_total = qpos_traj.shape[0]
    data_dim = qpos_traj.shape[1]
    
    print(f"🎮 Creating environment: {env_id}")
    env = gym.make(env_id, render_mode="human")
    env.reset()
    
    model_wrapper, data_wrapper = get_mujoco_wrappers(env)
    raw_model, raw_data = get_raw_objects(model_wrapper, data_wrapper)
    
    # OFFSET (Hardcoded based on your map_joints output)
    ROBOT_START_IDX = 9 
    ROBOT_END_IDX = ROBOT_START_IDX + 19 
    
    # Calculate Stride
    # If speed is 1, stride is 1 (play every frame). 
    # If speed is 10, stride is 10 (play every 10th frame).
    stride = int(speed_multiplier)
    if stride < 1: stride = 1
    
    print(f"⏩ Speed Multiplier: {stride}x (Rendering every {stride}th frame)")
    print(f"   Total Frames: {n_frames_total} -> Display Frames: {n_frames_total // stride}")

    print("▶️  Starting Replay...")
    
    # Loop with stride
    for i in range(0, n_frames_total, stride):
        
        # 1. Prepare Data
        frame_data = qpos_traj[i].copy()
        frame_data[3:7] = normalize_quaternion(frame_data[3:7])

        # 2. Direct Write
        raw_data.qpos[ROBOT_START_IDX : ROBOT_END_IDX] = frame_data
        raw_data.qvel[:] = 0 

        # 3. Update Visuals
        mujoco.mj_forward(raw_model, raw_data)
        env.render()
        
        # 4. Standard 60Hz Sleep
        # We keep the sleep constant but skip data frames to speed up time
        time.sleep(1/60.0)

    print("✅ Replay finished.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0")
    parser.add_argument("--speed", type=int, default=10, help="Playback speed (frame skip). Try 8 or 10 for realtime.")
    args = parser.parse_args()
    
    replay_episode(args.data, args.env, args.speed)