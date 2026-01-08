import numpy as np
import imitate_config as C

def calculate_reward(state, next_state, action, prev_action, qpos0, prev_base_ang_vel, 
                     episode_time, true_height, prev_height, 
                     motion_loader=None, info={}):
    
    # --- 1. GET CURRENT STATE ---
    if "current_qpos" not in info:
        return 0.0, False, "No Info", {}

    current_qpos = np.array(info["current_qpos"]) 
    current_qvel = np.array(info.get("current_qvel", np.zeros_like(current_qpos)))

    # --- 2. GET REFERENCE FRAME ---
    if motion_loader is None:
        return 0.0, False, "No Loader", {}

    ref_qpos = motion_loader.get_frame(episode_time)
    ref_qvel = motion_loader.get_frame_vel(episode_time)

    # Ensure ref is array
    if not isinstance(ref_qpos, np.ndarray):
        ref_qpos = np.array(ref_qpos)
    
    # --- 3. DEBUG PRINTING (The Fix) ---
    if C.DEBUG_VALUES:
        # Only print for the first few steps of an episode or if error is huge
        # We check a global counter or just print if height is weird
        rob_z = current_qpos[C.ROOT_Z_IDX]
        ref_z = ref_qpos[C.ROOT_Z_IDX]
        
        # If we are crashing immediately, print why
        if rob_z < 0.3 or abs(rob_z - ref_z) > 0.5:
            print(f"\n[DEBUG] Time: {episode_time:.2f}")
            print(f"  Robot Z: {rob_z:.3f} | Ref Z: {ref_z:.3f}")
            print(f"  Robot Root: {current_qpos[:3]}")
            print(f"  Ref Root:   {ref_qpos[:3]}")

    # --- 4. TERMINATION ---
    terminated = False
    reason = ""
    
    # Fail if height is too low (fell over)
    # Be lenient: 0.28 might be too high if the trajectory dips low
    if current_qpos[C.ROOT_Z_IDX] < 0.25: 
        terminated = True
        reason = "Fell Over (Height)"

    # --- 5. CALCULATE TRACKING ERROR ---
    relevant_indices = [2] + list(range(3, len(current_qpos)))
    
    min_len = min(len(current_qpos), len(ref_qpos))
    valid_indices = [i for i in relevant_indices if i < min_len]

    diff = current_qpos[valid_indices] - ref_qpos[valid_indices]
    error_qpos = np.sum(np.square(diff))
    
    r_tracking_qpos = np.exp(-2.0 * error_qpos) * C.TRACKING_QPOS_W

    r_tracking_vel = 0.0
    if ref_qvel is not None and len(ref_qvel) > 0:
        min_v_len = min(len(current_qvel), len(ref_qvel))
        diff_v = current_qvel[:min_v_len] - ref_qvel[:min_v_len]
        error_vel = np.sum(np.square(diff_v))
        r_tracking_vel = np.exp(-0.1 * error_vel) * C.TRACKING_VEL_W

    # --- 6. TOTAL ---
    r_alive = C.ALIVE_W
    
    # Handle action tuple
    act_calc = action[0] if (isinstance(action, tuple) or isinstance(action, list)) else action
    r_energy = np.exp(-np.sum(np.square(act_calc)) / 1.0) * C.ENERGY_W

    total_reward = r_tracking_qpos + r_tracking_vel + r_alive + r_energy

    stats = {
        "reward_total": total_reward,
        "track_qpos": r_tracking_qpos,
        "track_vel": r_tracking_vel,
        "alive": r_alive,
        "err_qpos": error_qpos 
    }

    return total_reward, terminated, reason, stats