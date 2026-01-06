import numpy as np
import run_config as C

def get_phase_targets(episode_time, freq, phase_offset=0.0):
    """
    Calculates the target leg positions based on a phase clock.
    """
    phase = ((episode_time * freq) + phase_offset) % 1.0
    
    if phase < 0.5:
        # Left Leg Swing (Knee Up)
        swing_progress = (phase / 0.5) * np.pi 
        target_left_knee = np.sin(swing_progress) * C.PHASE_KNEE_AMPLITUDE
        target_right_knee = 0.0 
    else:
        # Right Leg Swing (Knee Up)
        swing_progress = ((phase - 0.5) / 0.5) * np.pi
        target_left_knee = 0.0 
        target_right_knee = np.sin(swing_progress) * C.PHASE_KNEE_AMPLITUDE
        
    return target_left_knee, target_right_knee

def calculate_reward(state, next_state, action, prev_action, qpos0, prev_base_ang_vel, 
                     episode_time, true_height, prev_height, phase_offset=0.0):
    
    # --- 1. EXTRACT STATE ---
    # Indices based on Config
    current_left_knee = next_state[C.KNEE_IDXS[0]]
    current_right_knee = next_state[C.KNEE_IDXS[1]]
    
    current_left_hip_pitch = next_state[C.HIP_PITCH_IDXS[0]]
    current_right_hip_pitch = next_state[C.HIP_PITCH_IDXS[1]]
    
    # [NEW] Splay indices (Roll/Yaw)
    current_left_hip_roll = next_state[C.HIP_ROLL_IDXS[0]]
    current_right_hip_roll = next_state[C.HIP_ROLL_IDXS[1]]
    current_left_hip_yaw = next_state[C.HIP_YAW_IDXS[0]]
    current_right_hip_yaw = next_state[C.HIP_YAW_IDXS[1]]
    
    projected_gravity = next_state[24:27] 
    
    vel_x = next_state[33]
    vel_y = next_state[34]
    vel_z = (true_height - prev_height) / C.DT 
    
    ang_vel_xy = np.linalg.norm(next_state[27:29])
    ang_vel_yaw = np.abs(next_state[29])

    # --- 2. TERMINATION LOGIC ---
    terminated = False
    reason = ""
    lean_magnitude = np.linalg.norm(projected_gravity[:2])

    if true_height < 0.28:
        terminated = True
        reason = "Fell Over (Height)"
    
    if lean_magnitude > C.LEAN_THRESHOLD:
        terminated = True
        reason = "Fell Over (Tilt)"

    # --- 3. REWARD COMPONENTS ---

    # A. SURVIVAL
    r_alive = C.ALIVE_W

    # B. HEIGHT
    height_error = true_height - C.TARGET_HEIGHT
    r_height = np.exp(-np.square(height_error) / 0.05) * C.HEIGHT_W

    # C. FORWARD VELOCITY
    if vel_x > 0:
        r_run = np.exp(-np.square(vel_x - C.TARGET_VEL_X) / 0.5) * C.TRACKING_VEL_W
    else:
        r_run = vel_x * abs(C.BACKWARDS_PENALTY_W)

    # D. POSTURE
    r_side_tilt = -np.square(projected_gravity[1]) * C.SIDE_TILT_PENALTY
    
    forward_tilt = projected_gravity[0]
    if forward_tilt < 0:
        r_forward_tilt = forward_tilt * C.BACKWARD_TILT_PENALTY
    else:
        error_pitch = forward_tilt - C.TARGET_FORWARD_TILT
        r_forward_tilt = np.exp(-np.square(error_pitch) / 0.1) * C.FORWARD_TILT_W
        
    r_posture = r_side_tilt + r_forward_tilt

    # E. SPLIT STANCE (Sagittal Plane Scissor)
    phase = ((episode_time * C.GAIT_FREQ) + phase_offset) % 1.0
    if phase < 0.5:
        # Left > Right
        actual_split = current_left_hip_pitch - current_right_hip_pitch
        r_split = np.exp(-np.square(actual_split - C.TARGET_HIP_SPLIT) / 0.2) * C.SPLIT_STANCE_W
    else:
        # Right > Left
        actual_split = current_right_hip_pitch - current_left_hip_pitch
        r_split = np.exp(-np.square(actual_split - C.TARGET_HIP_SPLIT) / 0.2) * C.SPLIT_STANCE_W

    # [NEW] F. ANTI-SPLAY (Constrain Hips to Forward Motion)
    # We punish Hip Roll and Hip Yaw magnitude.
    splay_magnitude = (np.square(current_left_hip_roll) + np.square(current_right_hip_roll) +
                       np.square(current_left_hip_yaw) + np.square(current_right_hip_yaw))
    r_hip_splay = -splay_magnitude * abs(C.HIP_SPLAY_PENALTY_W)

    # G. STEERING
    r_side_drift = -np.square(vel_y) * 2.0
    heading_error = np.abs(projected_gravity[1]) 
    r_heading = -(heading_error * 2.0 + np.square(heading_error) * 5.0)
    r_yaw_spin = -np.square(ang_vel_yaw) * 0.5

    # H. STABILITY
    r_lin_vel_z = -np.square(vel_z) * abs(C.LIN_VEL_Z_W)
    r_ang_vel_xy = -np.square(ang_vel_xy) * abs(C.ANG_VEL_XY_W)
    r_upright = np.exp(-np.square(lean_magnitude) / 0.1) * C.UPRIGHT_W

    # I. RHYTHM
    start_gate = np.clip(episode_time / 2.0, 0.0, 1.0)
    t_left_knee, t_right_knee = get_phase_targets(episode_time, C.GAIT_FREQ, phase_offset)
    
    diff_left = np.abs(current_left_knee) - t_left_knee
    diff_right = np.abs(current_right_knee) - t_right_knee
    error_phase = np.square(diff_left) + np.square(diff_right)
    r_phase = (np.exp(-error_phase / 0.5) * C.PHASE_W) * start_gate

    # J. STYLE
    target_knee_angle = 0.5
    avg_knee_angle = (np.abs(current_left_knee) + np.abs(current_right_knee)) / 2.0
    r_knee_bend = np.exp(-np.square(avg_knee_angle - target_knee_angle) / 0.2) * C.KNEE_BEND_W

    # K. REGULARIZATION
    r_energy = -np.sum(np.square(action)) * abs(C.ENERGY_W)
    r_action_rate = -np.sum(np.square(action - prev_action)) * abs(C.ACTION_RATE_W)

    # --- 4. TOTAL ---
    upright_gate = 1.0 if true_height > 0.4 else 0.0
    
    total_reward = (
        r_alive + 
        r_height + 
        r_posture + 
        r_split + 
        r_hip_splay +        # <--- Added here
        r_lin_vel_z + 
        r_ang_vel_xy + 
        r_heading + 
        r_side_drift + 
        r_yaw_spin +
        r_knee_bend +
        r_energy + 
        r_action_rate +
        (r_run * upright_gate) + 
        (r_phase * upright_gate)
    )

    is_stepping = (vel_x > 0.1) and (r_split > 0.5) and (r_posture > 0.5)

    # --- 5. STATS ---
    stats = {
        "reward_total": total_reward,
        "run": r_run,
        "split_stance": r_split,
        "hip_splay": r_hip_splay, # Monitor this!
        "posture": r_posture,
        "phase": r_phase,
        "knee_bend": r_knee_bend,
        "actual_vel_x": vel_x,
        "actual_height": true_height,
        "is_stepping": 1.0 if is_stepping else 0.0, # Average of this will tell you % of time spent stepping
    }

    return total_reward, terminated, reason, stats