import numpy as np
import run_config as C

def get_phase_targets(episode_time, freq, phase_offset=0.0):
    """
    Returns the expected 'swing' state for feet based on phase.
    """
    phase = ((episode_time * freq) + phase_offset) % 1.0
    
    # 0.0-0.5: Left Swing, Right Stance
    if phase < 0.5:
        # Target: High knee lift for clearance
        swing_progress = (phase / 0.5) * np.pi
        target_left_knee = np.sin(swing_progress) * C.PHASE_KNEE_AMPLITUDE
        target_right_knee = 0.0
    else:
        # 0.5-1.0: Right Swing, Left Stance
        swing_progress = ((phase - 0.5) / 0.5) * np.pi
        target_left_knee = 0.0
        target_right_knee = np.sin(swing_progress) * C.PHASE_KNEE_AMPLITUDE
        
    return target_left_knee, target_right_knee

def calculate_reward(state, next_state, action, prev_action, qpos0, prev_base_ang_vel, 
                     episode_time, true_height, prev_height, phase_offset=0.0):
    
    # --- 1. EXTRACT STATE ---
    # Indices
    current_left_knee = next_state[C.KNEE_IDXS[0]]
    current_right_knee = next_state[C.KNEE_IDXS[1]]
    
    current_left_hip_pitch = next_state[C.HIP_PITCH_IDXS[0]]
    current_right_hip_pitch = next_state[C.HIP_PITCH_IDXS[1]]
    
    current_left_hip_roll = next_state[C.HIP_ROLL_IDXS[0]]
    current_right_hip_roll = next_state[C.HIP_ROLL_IDXS[1]]
    current_left_hip_yaw = next_state[C.HIP_YAW_IDXS[0]]
    current_right_hip_yaw = next_state[C.HIP_YAW_IDXS[1]]
    
    # Base Motion
    vel_x = next_state[33] 
    vel_y = next_state[34] 
    vel_z = (true_height - prev_height) / C.DT
    ang_vel_roll_pitch = next_state[27:29]
    ang_vel_yaw = next_state[29]
    
    # Orientation
    proj_grav = next_state[24:27] # [x, y, z]
    
    # Nominal Pose Extraction
    current_qpos = next_state[:12] 
    
    # --- 2. TERMINATION ---
    terminated = False
    reason = ""
    if true_height < 0.28: terminated = True; reason = "Fell Over (Height)"
    if np.linalg.norm(proj_grav[:2]) > C.LEAN_THRESHOLD: terminated = True; reason = "Fell Over (Tilt)"

    # --- 3. REWARD COMPONENTS ---
    r_alive = C.ALIVE_W

    # Height & Velocity
    r_height = np.exp(-np.square(true_height - C.TARGET_HEIGHT) / 0.05) * C.HEIGHT_W
    
    if vel_x > 0:
        r_run = np.exp(-np.square(vel_x - C.TARGET_VEL_X) / 0.5) * C.TRACKING_VEL_W
    else:
        r_run = vel_x * abs(C.BACKWARDS_PENALTY_W)

    r_ang_vel_yaw = np.exp(-np.square(ang_vel_yaw) / 0.5) * C.TRACKING_ANG_VEL_W

    # --- POSTURE & ORIENTATION ---
    
    # 1. Nominal Pose
    if qpos0 is not None:
        target_qpos = qpos0
        if qpos0.shape[0] > current_qpos.shape[0]:
            target_qpos = qpos0[:current_qpos.shape[0]]
            
        qpos_error = np.sum(np.square(current_qpos - target_qpos))
        r_dof_pos = np.exp(-qpos_error / 1.0) * C.DOF_POS_W
    else:
        r_dof_pos = 0.0

    # 2. Asymmetric Tilt
    r_side_tilt = -np.square(proj_grav[1]) * C.SIDE_TILT_PENALTY
    forward_tilt = proj_grav[0]
    if forward_tilt < 0:
        r_forward_tilt = forward_tilt * C.BACKWARD_TILT_PENALTY
    else:
        r_forward_tilt = np.exp(-np.square(forward_tilt - C.TARGET_FORWARD_TILT) / 0.1) * C.FORWARD_TILT_W
    r_posture = r_side_tilt + r_forward_tilt

    # --- GAIT & LEG SHAPING ---
    phase = ((episode_time * C.GAIT_FREQ) + phase_offset) % 1.0
    
    # 1. Split Stance
    if phase < 0.5:
        r_split = np.exp(-np.square((current_left_hip_pitch - current_right_hip_pitch) - C.TARGET_HIP_SPLIT) / 0.2) * C.SPLIT_STANCE_W
    else:
        r_split = np.exp(-np.square((current_right_hip_pitch - current_left_hip_pitch) - C.TARGET_HIP_SPLIT) / 0.2) * C.SPLIT_STANCE_W

    # 2. Anti-Splay
    splay_mag = np.sum(np.square([current_left_hip_roll, current_right_hip_roll, current_left_hip_yaw, current_right_hip_yaw]))
    r_hip_splay = -splay_mag * abs(C.HIP_SPLAY_PENALTY_W)

    # 3. Stance Straightening
    target_knee = current_right_knee if phase < 0.5 else current_left_knee
    r_stance_straight = np.exp(-np.square(target_knee - 0.05) / 0.1) * C.STANCE_STRAIGHT_W

    # 4. Knee Bend
    avg_knee = (np.abs(current_left_knee) + np.abs(current_right_knee)) / 2.0
    r_knee_bend = np.exp(-np.square(avg_knee - 0.5) / 0.2) * C.KNEE_BEND_W
    
    # 5. Feet Air Time (Swing Leg Reward)
    # 0.0-0.5: Left Swing. 0.5-1.0: Right Swing.
    if phase < 0.5:
        r_feet_air_time = np.exp(-np.square(current_left_knee - 1.2) / 0.5) * C.FEET_AIR_TIME_W
    else:
        r_feet_air_time = np.exp(-np.square(current_right_knee - 1.2) / 0.5) * C.FEET_AIR_TIME_W

    # 6. [NEW] Anti-Hopping (Double Air Penalty)
    # If both knees are bent > 0.5 rads (~28 deg) simultaneously, apply penalty.
    if current_left_knee > 0.5 and current_right_knee > 0.5:
        r_double_air = C.DOUBLE_AIR_PENALTY_W
    else:
        r_double_air = 0.0

    # --- STABILITY & EFFICIENCY ---
    r_lin_vel_z = np.exp(-np.square(vel_z) / 0.1) * C.LIN_VEL_Z_W
    r_ang_vel_xy = np.exp(-np.sum(np.square(ang_vel_roll_pitch)) / 1.0) * C.ANG_VEL_XY_W

    r_action_rate = np.exp(-np.sum(np.square(action - prev_action)) / 0.1) * C.ACTION_RATE_W
    r_energy = np.exp(-np.sum(np.square(action)) / 1.0) * C.ENERGY_W

    # --- 4. TOTAL ---
    upright_gate = 1.0 if true_height > 0.4 else 0.0
    
    total_reward = (
        r_alive + 
        r_height + 
        r_run * upright_gate + 
        r_ang_vel_yaw + 
        r_posture + 
        r_dof_pos + 
        r_split + 
        r_hip_splay + 
        r_stance_straight + 
        r_knee_bend +
        r_feet_air_time * upright_gate +
        r_double_air +  # <--- NEW
        r_lin_vel_z + 
        r_ang_vel_xy + 
        r_action_rate + 
        r_energy
    )

    # --- 5. STATS ---
    is_stepping = 1.0 if (vel_x > 0.1 and r_split > 0.5 and r_posture > 0.5) else 0.0

    stats = {
        "reward_total": total_reward,
        "run": r_run,
        "dof_pos": r_dof_pos,
        "split_stance": r_split,
        "hip_splay": r_hip_splay,
        "stance_straight": r_stance_straight,
        "posture": r_posture,
        "feet_air": r_feet_air_time,
        "double_air": r_double_air, # Monitor this!
        "yaw_control": r_ang_vel_yaw,
        "actual_vel_x": vel_x,
        "is_stepping": is_stepping
    }

    return total_reward, terminated, reason, stats