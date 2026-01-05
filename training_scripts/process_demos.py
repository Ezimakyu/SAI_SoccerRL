# python process_demos.py --data ./demonstrations/walking.npz --env LowerT1GoaliePenaltyKick-v0
# python process_demos.py --data ./demonstrations/jogging.npz --env LowerT1GoaliePenaltyKick-v0
# python process_demos.py --data ./demonstrations/running.npz --env LowerT1GoaliePenaltyKick-v0
# python process_demos.py --data ./demonstrations/soccer_drill_run.npz --env LowerT1GoaliePenaltyKick-v0

# process_demos.py
import numpy as np
import gymnasium as gym
import argparse
import inspect
import os
import sys
import sai_mujoco.utils.v0.binding_utils as bu
import mujoco
from scipy.signal import butter, filtfilt

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from training_scripts.main import Preprocessor
except ImportError:
    try:
        from main import Preprocessor
    except ImportError:
        print("Warning: Could not import Preprocessor. Obs will be raw.")
        Preprocessor = None

# --- CONFIGURATION ---
STATIC_SPAWN = [0.0] * 9 

# ==========================================
# 1. SIGNAL PROCESSING
# ==========================================
def low_pass_filter(data, freq=50, cutoff=6.0, order=2):
    nyquist = 0.5 * freq
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

# ==========================================
# 2. PATCHING & HELPERS
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
# 3. INFO SYNTHESIZER (The Critical Fix)
# ==========================================
def synthesize_info(model, data, robot_start_idx):
    """
    Manually constructs the exact 'info' dictionary expected by the Preprocessor.
    Calculates relative positions, velocities, and sensor data from raw physics.
    """
    # 1. Extract Robot State
    # Robot is at robot_start_idx (9). 
    # qpos: [x,y,z, qw,qx,qy,qz, ...]
    # qvel: [vx,vy,vz, wx,wy,wz, ...] (MuJoCo free joint convention)
    
    # Indices for robot root in GLOBAL qpos/qvel arrays
    r_pos_idx = robot_start_idx
    r_quat_idx = robot_start_idx + 3
    
    # Velocity is usually offset by -1 relative to qpos due to lack of W-quat
    # If robot is 2nd object (ball first), ball has 6 dof velocity.
    # So robot qvel starts at index 6.
    r_vel_idx = 6 
    
    robot_pos = data.qpos[r_pos_idx : r_pos_idx+3].copy()
    robot_quat = data.qpos[r_quat_idx : r_quat_idx+4].copy()
    
    robot_vel_all = data.qvel[r_vel_idx : r_vel_idx+6]
    robot_linvel = robot_vel_all[0:3].copy()
    robot_gyro = robot_vel_all[3:6].copy()

    # 2. Extract Ball State (Indices 0-7)
    ball_pos = data.qpos[0:3].copy()
    ball_vel = data.qvel[0:3].copy() # Linear vel

    # 3. Define Static World Objects (Goals)
    # Standard field coordinates
    goal_team_0_pos = np.array([-11.0, 0.0, 0.0]) # Left goal
    goal_team_1_pos = np.array([11.0, 0.0, 0.0])  # Right goal

    # 4. Calculate Relative Vectors (for Preprocessor)
    # Note: Preprocessor expects THESE to be rotated into robot frame later?
    # Actually, modify_state takes these and passes them.
    # Usually "rel_robot" implies (Obj_Pos - Robot_Pos).
    
    info = {
        # -- Core Robot Sensors --
        "robot_quat": robot_quat,
        "robot_gyro": robot_gyro,
        # Fake accelerometers (hard to calc without force summing)
        "robot_accelerometer": np.array([0.0, 0.0, 9.81]), 
        "robot_velocimeter": robot_linvel, 
        
        # -- Relative Goals --
        "goal_team_0_rel_robot": goal_team_0_pos - robot_pos,
        "goal_team_1_rel_robot": goal_team_1_pos - robot_pos,
        
        # -- Relative Ball --
        "ball_xpos_rel_robot": ball_pos - robot_pos,
        "ball_velp_rel_robot": ball_vel - robot_linvel, # Relative velocity
        "ball_velr_rel_robot": np.zeros(3), # Ball angular vel (ignore)
        
        # -- Goal Relative to Ball --
        "goal_team_0_rel_ball": goal_team_0_pos - ball_pos,
        "goal_team_1_rel_ball": goal_team_1_pos - ball_pos,
        
        # -- Misc --
        "player_team": np.array([0.0]), # Team 0
        "task_index": np.array([0]),    # Goalie Task
        
        # -- Missing / Optional (Safe to be zeros) --
        "goalkeeper_team_0_xpos_rel_robot": np.zeros(3),
        "goalkeeper_team_0_velp_rel_robot": np.zeros(3),
        "goalkeeper_team_1_xpos_rel_robot": np.zeros(3),
        "goalkeeper_team_1_velp_rel_robot": np.zeros(3),
        "target_xpos_rel_robot": np.zeros(3),
        "target_velp_rel_robot": np.zeros(3),
        "defender_xpos": np.zeros(8),
    }
    
    return info

# ==========================================
# 4. MAIN PROCESSOR
# ==========================================
def process_demonstration(data_path, env_id):
    print(f"\n📂 Loading: {data_path}")
    try: raw_data = np.load(data_path)
    except: 
        print("File not found.")
        return

    qpos_traj = raw_data['qpos']
    qvel_traj = raw_data['qvel']
    n_frames = qpos_traj.shape[0] - 1
    
    print(f"🎮 Env: {env_id}")
    env = gym.make(env_id, render_mode=None)
    env.reset()
    
    preprocessor = Preprocessor() if Preprocessor else None

    model_wrapper, data_wrapper = get_mujoco_wrappers(env)
    raw_model, raw_data = get_raw_objects(model_wrapper, data_wrapper)
    
    # --- FIX 1: DISABLE CONTACTS ---
    print("🔧 Disabling contacts for Inverse Dynamics...")
    contact_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    raw_model.opt.disableflags |= contact_bit

    # --- FIX 2: FILTER VELOCITY ---
    print("🌊 Applying Low-Pass Filter (6Hz)...")
    qvel_filtered = low_pass_filter(qvel_traj, freq=50, cutoff=6.0)

    ROBOT_START_IDX = 9 
    ROBOT_END_IDX = ROBOT_START_IDX + 19 
    
    actions_buffer = []
    obs_buffer = []
    next_obs_buffer = []
    rewards_buffer = []
    dones_buffer = []

    dt = raw_model.opt.timestep
    print(f"⚙️  dt: {dt} | Frames: {n_frames}")

    action_low = env.action_space.low
    action_high = env.action_space.high
    action_scale = np.where(action_high == 0, 1.0, action_high)

    for i in range(n_frames):
        # 1. Prepare State
        q_curr = qpos_traj[i].copy()
        q_curr[3:7] = normalize_quaternion(q_curr[3:7])
        v_curr = qvel_filtered[i].copy()
        v_next = qvel_filtered[i+1].copy()

        # 2. Acceleration
        q_acc = (v_next - v_curr) / dt

        # 3. FILL PHYSICS STATE
        if len(STATIC_SPAWN) == ROBOT_START_IDX:
            raw_data.qpos[:ROBOT_START_IDX] = np.array(STATIC_SPAWN)
        
        # A. Inject Robot State
        raw_data.qpos[ROBOT_START_IDX : ROBOT_END_IDX] = q_curr
        
        # --- FIX: LIFT THE ROBOT ---
        # Add 1cm (0.01m) to the Z-height of the robot root.
        # Root Position indices are [0, 1, 2] relative to robot start.
        # Z is index 2.
        z_index = ROBOT_START_IDX + 2
        raw_data.qpos[z_index] += 0.015  # Lift by 1.5cm (tweak if needed)
        # ---------------------------

        raw_data.qvel[-18:] = v_curr[-18:]
        raw_data.qacc[-18:] = q_acc[-18:]

        # 4. Inverse Dynamics
        mujoco.mj_inverse(raw_model, raw_data)
        
        # 5. Extract & Normalize Action
        action_dim = env.action_space.shape[0]
        raw_torque = raw_data.qfrc_inverse[-action_dim:].copy()
        clipped_torque = np.clip(raw_torque, action_low, action_high)
        normalized_action = clipped_torque / action_scale
        actions_buffer.append(normalized_action)

        # 6. Extract Observation
        # Enable contacts for sensor reading
        raw_model.opt.disableflags &= ~contact_bit
        mujoco.mj_forward(raw_model, raw_data)
        
        try:
            raw_obs = env.unwrapped._get_obs()
            
            if preprocessor:
                # --- FIX: SYNTHESIZE FULL INFO ---
                # We build the dictionary manually to satisfy Preprocessor requirements
                info = synthesize_info(raw_model, raw_data, ROBOT_START_IDX)
                processed_state = preprocessor.modify_state(raw_obs, info).squeeze()
            else:
                processed_state = raw_obs
                
        except Exception as e:
            print(f"Error getting obs at step {i}: {e}")
            break
            
        raw_model.opt.disableflags |= contact_bit

        obs_buffer.append(processed_state)
        rewards_buffer.append(0.0)
        dones_buffer.append(False)
        if len(obs_buffer) > 1:
            next_obs_buffer.append(processed_state)

    # --- FIX 4: TIME ALIGNMENT ---
    actions_out = np.array(actions_buffer[1:])
    obs_out = np.array(obs_buffer[:-1])
    next_obs_out = np.array(next_obs_buffer[:-1])
    rewards_out = np.array(rewards_buffer[:-1])
    dones_out = np.array(dones_buffer[:-1])
    
    final_count = len(actions_out)

    save_path = data_path.replace(".npz", "_processed.npz")
    np.savez(save_path, 
             obs=obs_out, 
             actions=actions_out, 
             next_obs=next_obs_out,
             rewards=rewards_out, 
             dones=dones_out)
             
    print(f"✅ Saved: {save_path}")
    print(f"   Samples: {final_count}")
    if final_count > 0:
        print(f"   Obs Shape: {obs_out.shape} (Expected N x 88)")
        print(f"   Action Mean: {np.mean(actions_out):.4f}")
        print(f"   Action Std:  {np.std(actions_out):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--env", type=str, default="LowerT1GoaliePenaltyKick-v0")
    args = parser.parse_args()
    process_demonstration(args.data, args.env)