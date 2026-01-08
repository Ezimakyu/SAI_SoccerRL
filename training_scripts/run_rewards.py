import numpy as np
import run_config as C

def get_phase_signal(episode_time, freq, phase_offset=0.0):
    """Generates the phase angle (0 to 1) for the sine wave."""
    phase = ((episode_time * freq) + phase_offset) % 1.0
    return phase

def get_rz(phase, swing_height):
    """
    Returns target Z height for the foot based on phase.
    """
    # Left Swing: 0.0 - 0.5
    if phase < 0.5:
        progress = phase / 0.5
        left_z = np.sin(progress * np.pi) * swing_height
        right_z = 0.0
    else:
        progress = (phase - 0.5) / 0.5
        left_z = 0.0
        right_z = np.sin(progress * np.pi) * swing_height
        
    return left_z, right_z

def calculate_reward(state, next_state, action, prev_action, qpos0, prev_base_ang_vel, 
                     episode_time, true_height, prev_height, phase_offset=0.0, info={}):
    
    # --- 1. EXTRACT STATE ---
    # Indices for T1 Robot
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
    proj_grav = next_state[24:27]
    current_qpos = next_state[:12] 
    
    # --- 2. TERMINATION ---
    terminated = False
    reason = ""
    if true_height < 0.28: terminated = True; reason = "Fell Over (Height)"
    if np.linalg.norm(proj_grav[:2]) > C.LEAN_THRESHOLD: terminated = True; reason = "Fell Over (Tilt)"

    # --- 3. BASE REWARDS (Survival & Walk) ---
    r_alive = C.ALIVE_W
    r_height = np.exp(-np.square(true_height - C.TARGET_HEIGHT) / 0.05) * C.HEIGHT_W
    
    # Velocity: Reward forward progress (Bulldozer logic)
    if vel_x > 0:
        r_run = np.exp(-np.square(vel_x - C.TARGET_VEL_X) / 0.5) * C.TRACKING_VEL_W
    else:
        r_run = vel_x * abs(C.BACKWARDS_PENALTY_W)
    
    # [NEW] Forward Momentum (Linear Ramp)
    # Good for acceleration and transitions.
    # Logic: Reward velocity, but cap it at the target so it doesn't sprint forever.
    # If vel_x is 0.5, reward is 0.5 * W.
    # If vel_x is 1.5, reward is 1.5 * W.
    velocity_clipped = np.clip(vel_x, 0, C.TARGET_VEL_X)
    r_momentum = velocity_clipped * C.FORWARD_MOMENTUM_W

    r_ang_vel_yaw = np.exp(-np.square(ang_vel_yaw) / 0.5) * C.TRACKING_ANG_VEL_W

    # --- 4. POSTURE ---
    if qpos0 is not None:
        target_qpos = qpos0.copy()
        if target_qpos.shape[0] > current_qpos.shape[0]:
            target_qpos = target_qpos[:current_qpos.shape[0]]
            
        # Crouch Hack: We overwrite the knee targets in qpos0 to be slightly bent (0.1)
        # This prevents the robot from trying to lock knees perfectly straight.
        target_qpos[C.KNEE_IDXS[0]] = 0.25
        target_qpos[C.KNEE_IDXS[1]] = 0.25

        qpos_error = np.sum(np.square(current_qpos - target_qpos))
        r_dof_pos = np.exp(-qpos_error / 1.0) * C.DOF_POS_W
    else:
        r_dof_pos = 0.0

    r_side_tilt = -np.square(proj_grav[1]) * C.SIDE_TILT_PENALTY
    forward_tilt = proj_grav[0]
    if forward_tilt < 0:
        r_forward_tilt = forward_tilt * C.BACKWARD_TILT_PENALTY
    else:
        # Target a specific forward lean for momentum
        r_forward_tilt = np.exp(-np.square(forward_tilt - C.TARGET_FORWARD_TILT) / 0.1) * C.FORWARD_TILT_W
    r_posture = r_side_tilt + r_forward_tilt

    # --- 5. GAIT ---
    phase_val = get_phase_signal(episode_time, C.GAIT_FREQ, phase_offset)
    
    # Split Stance (Pitch)
    if phase_val < 0.5:
        r_split = np.exp(-np.square((current_left_hip_pitch - current_right_hip_pitch) - C.TARGET_HIP_SPLIT) / 0.2) * C.SPLIT_STANCE_W
    else:
        r_split = np.exp(-np.square((current_right_hip_pitch - current_left_hip_pitch) - C.TARGET_HIP_SPLIT) / 0.2) * C.SPLIT_STANCE_W

    # [NEW] Hip Yaw Neutrality (Knees Forward)
    # Target 0.0 radians (straight forward). 
    # Any deviation > 0.2 rads gets penalized heavily.
    yaw_error = np.square(current_left_hip_yaw) + np.square(current_right_hip_yaw)
    r_hip_yaw = np.exp(-yaw_error / 0.1) * C.HIP_YAW_W

    # Stance Width (Roll) - Sumo Stability
    width_error = np.square(np.abs(current_left_hip_roll) - C.TARGET_HIP_WIDTH) + \
                  np.square(np.abs(current_right_hip_roll) - C.TARGET_HIP_WIDTH)
    r_stance_width = np.exp(-width_error / 0.01) * C.STANCE_WIDTH_W

    # --- Phase-Dependent Knee Logic (The Fix) ---
    # Determine Stance vs Swing leg based on phase
    if phase_val < 0.5:
        swing_knee = current_left_knee
        stance_knee = current_right_knee
    else:
        swing_knee = current_right_knee
        stance_knee = current_left_knee

    # 1. Stance Straight Reward (Support the weight)
    # Target 0.1 rads (approx 5 degrees) - straight but not hyperextended
    r_stance_straight = np.exp(-np.square(stance_knee - 0.1) / 0.1) * C.STANCE_STRAIGHT_W

    # 2. Swing Bend Reward (Clear the ground!)
    # Target 1.0 rads (approx 60 degrees) - high bend to prevent toe stubbing
    r_swing_bend = np.exp(-np.square(swing_knee - 1.0) / 0.2) * C.KNEE_BEND_W
    
    # Phase & Air Time Trajectory
    target_left_z, target_right_z = get_rz(phase_val, C.MAX_FOOT_HEIGHT)
    est_left_z = (current_left_knee / 1.5) * C.MAX_FOOT_HEIGHT
    est_right_z = (current_right_knee / 1.5) * C.MAX_FOOT_HEIGHT
    
    error_phase_traj = np.square(est_left_z - target_left_z) + np.square(est_right_z - target_right_z)
    r_feet_phase = np.exp(-error_phase_traj / 0.01) * C.FEET_PHASE_W 

    if phase_val < 0.5:
        r_feet_air_time = np.exp(-np.square(current_left_knee - 1.2) / 0.5) * C.FEET_AIR_TIME_W
    else:
        r_feet_air_time = np.exp(-np.square(current_right_knee - 1.2) / 0.5) * C.FEET_AIR_TIME_W

    # Anti-Hopping
    if est_left_z > 0.05 and est_right_z > 0.05:
        r_double_air = C.DOUBLE_AIR_PENALTY_W
    else:
        r_double_air = 0.0
        
    r_feet_slip = 0.0 

    # --- 6. KICKING & PRECISION ---
    r_ball_approach = 0.0
    r_ball_vel = 0.0
    r_precision_shot = 0.0
    
    if "ball_xpos_rel_robot" in info and "target_xpos_rel_robot" in info:
        ball_rel = np.array(info["ball_xpos_rel_robot"]).flatten()
        dist_to_ball = np.linalg.norm(ball_rel)
        
        # 1. Approach
        r_ball_approach = np.exp(-dist_to_ball / 1.0) * C.BALL_APPROACH_W
        
        # 2. Velocity
        ball_vel = np.zeros(3)
        if "ball_velp_rel_robot" in info:
            ball_vel = np.array(info["ball_velp_rel_robot"]).flatten()
        ball_speed = np.linalg.norm(ball_vel)
        r_ball_vel = ball_speed * C.BALL_VEL_W
        
        # 3. Precision
        target_rel = np.array(info["target_xpos_rel_robot"]).flatten()
        target_dist = np.linalg.norm(target_rel)
        
        if target_dist > 0.01 and ball_speed > 0.01:
            target_dir = target_rel / target_dist
            ball_dir = ball_vel / ball_speed
            alignment = np.dot(ball_dir, target_dir)
            if alignment > 0:
                r_precision_shot = (ball_speed * alignment) * C.TARGET_PRECISION_W
    
    # --- STABILITY & EFFICIENCY ---
    r_lin_vel_z = np.exp(-np.square(vel_z) / 0.1) * C.LIN_VEL_Z_W
    r_ang_vel_xy = np.exp(-np.sum(np.square(ang_vel_roll_pitch)) / 1.0) * C.ANG_VEL_XY_W
    r_action_rate = np.exp(-np.sum(np.square(action - prev_action)) / 0.1) * C.ACTION_RATE_W
    r_energy = np.exp(-np.sum(np.square(action)) / 1.0) * C.ENERGY_W

    # --- TOTAL ---
    upright_gate = 1.0 if true_height > 0.4 else 0.0
    
    total_reward = (
        r_alive + 
        r_height + 
        r_run * upright_gate + 
        r_momentum * upright_gate +
        r_ang_vel_yaw + 
        r_posture + 
        r_dof_pos + 
        r_split + 
        r_stance_width +   
        r_hip_yaw +
        r_stance_straight + 
        r_swing_bend +       # Using the new phase-dependent bend
        r_feet_air_time * upright_gate +
        r_feet_phase * upright_gate + 
        r_feet_slip +                 
        r_double_air +  
        r_ball_approach + 
        r_ball_vel +       
        r_precision_shot + 
        r_lin_vel_z + 
        r_ang_vel_xy + 
        r_action_rate + 
        r_energy
    )

    stats = {
        "reward_total": total_reward,
        "run": r_run,
        "momentum": r_momentum,
        "dof_pos": r_dof_pos,
        "split_stance": r_split,
        "stance_width": r_stance_width,
        "hip_yaw": r_hip_yaw,
        "stance_straight": r_stance_straight,
        "swing_bend": r_swing_bend, 
        "posture": r_posture,
        "feet_air": r_feet_air_time,
        "feet_phase": r_feet_phase, 
        "double_air": r_double_air, 
        "ball_app": r_ball_approach,
        "ball_vel": r_ball_vel,
        "precision": r_precision_shot,
        "yaw_control": r_ang_vel_yaw,
        "actual_vel_x": vel_x,
    }

    return total_reward, terminated, reason, stats