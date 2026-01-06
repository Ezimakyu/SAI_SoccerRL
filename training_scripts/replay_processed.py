# python replay_processed.py \
#   --processed demonstrations/walking_processed.npz \
#   --demo demonstrations/walking.npz \
#   --env LowerT1GoaliePenaltyKick-v0

# Note that the processed videos suffer from "drift", 

import numpy as np
import gymnasium as gym
import time
import argparse
import sys
import os
import inspect
import sai_mujoco.utils.v0.binding_utils as bu
import mujoco

# --- PATH SETUP ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import ENV_IDS, get_action_function

# --- PATCHING ---
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

# --- HELPER: GET MUJOCO ---
def get_mujoco_internals(env):
    base = env.unwrapped
    if hasattr(base, 'model') and hasattr(base, 'data'): return base.model, base.data
    if hasattr(base, 'sim'): return base.sim.model, base.sim.data
    if hasattr(env, 'model'): return env.model, env.data
    raise ValueError("Could not find MuJoCo internals")

def replay_processed_actions(file_path, demo_path, env_id, fps=50, debug=False):
    print(f"📂 Processed Actions: {file_path}")
    print(f"📂 Original Demo:    {demo_path}")
    
    try:
        data = np.load(file_path)
        demo_data = np.load(demo_path) 
    except FileNotFoundError:
        print("File not found!")
        return

    actions = data['actions']
    
    # Original Kinematics (Ground Truth)
    # qpos shape: [frames, 19] (Robot only)
    orig_qpos_log = demo_data['qpos'] 
    
    print(f"🎮 Env: {env_id}")
    env = gym.make(env_id, render_mode="human")
    env.reset()
    
    model, sim_data = get_mujoco_internals(env)
    
    # Unwrap SAI pointers
    if hasattr(sim_data, "ptr"): sim_data = sim_data.ptr
    elif hasattr(sim_data, "_data"): sim_data = sim_data._data
    if hasattr(model, "ptr"): model = model.ptr
    elif hasattr(model, "_model"): model = model._model

    print("⚡ Teleporting to initial state...")
    
    # Offsets
    # Env: [Ball(7) + World(2) ... Robot(19)] -> Robot starts at 9
    ROBOT_START_IDX = 9
    ROBOT_END_IDX = ROBOT_START_IDX + 19 
    
    # Teleport to Frame 0
    sim_data.qpos[ROBOT_START_IDX : ROBOT_END_IDX] = orig_qpos_log[0]
    # sim_data.qvel[:] = 0 # Zero out velocity to verify static forces first? No, keep motion.
    sim_data.qvel[-18:] = demo_data['qvel'][0][-18:]
    
    mujoco.mj_forward(model, sim_data)
    action_function = get_action_function()
    
    print("▶️  Starting Replay...")
    if debug:
        print(f"{'Frame':<6} | {'Orig Z':<8} {'Sim Z':<8} {'Diff':<8} | {'Orig Knee':<9} {'Sim Knee':<9} | {'Torque Mag':<10}")
        print("-" * 80)

    for i, raw_action in enumerate(actions):
        if i >= len(orig_qpos_log): break

        # 1. Scale Action
        torque_action = action_function(raw_action)
        
        # 2. Step Physics
        # Note: We step *before* printing the result of this frame
        _, _, terminated, truncated, _ = env.step(torque_action)
        
        # 3. DEBUG LOGGING
        if debug:
            # Current Sim State (Robot indices)
            sim_qpos = sim_data.qpos[ROBOT_START_IDX : ROBOT_END_IDX]
            
            # Target State (from Log)
            # Frame i+1 because 'action i' produces 'state i+1'
            target_idx = min(i + 1, len(orig_qpos_log) - 1)
            orig_qpos = orig_qpos_log[target_idx]
            
            # --- METRICS ---
            # Root Height (Z is index 2)
            z_orig = orig_qpos[2]
            z_sim = sim_qpos[2]
            z_diff = z_sim - z_orig
            
            # Right Knee Angle (Index 15 approx? 3 + 4 + ... let's pick index 10)
            # Assuming standard humanoid: 0-2 root pos, 3-6 root quat.
            # Joints start at 7. Let's look at index 7 (Hip) or 10 (Knee).
            # Let's just track index 7 (Left Hip Pitch usually)
            j_orig = orig_qpos[7]
            j_sim = sim_qpos[7]
            
            # Torque Magnitude (Did we request infinite force?)
            torque_mag = np.linalg.norm(torque_action)
            
            # Print
            # Mark huge errors with *
            marker = "!!!" if abs(z_diff) > 0.05 else "" 
            print(f"{i:<6} | {z_orig:<8.4f} {z_sim:<8.4f} {z_diff:<8.4f} | {j_orig:<9.4f} {j_sim:<9.4f} | {torque_mag:<10.2f} {marker}")

        env.render()
        time.sleep(1.0 / fps)
        
        if terminated or truncated:
            print(f"⏹️  Episode ended at step {i}")
            break
            
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=str, required=True)
    parser.add_argument("--demo", type=str, required=True)
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    replay_processed_actions(args.processed, args.demo, args.env, fps=20, debug=args.debug)