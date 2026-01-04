# Calculations using the reward

import numpy as np
import balance_config as C

def calculate_reward(state, next_state, action, prev_action, qpos0):
    """
    Calculates the reward, termination status, and debug stats.
    
    Args:
        state: The state BEFORE the step (used for prev velocities).
        next_state: The state AFTER the step.
        action: The action taken.
        prev_action: The action taken in the previous step.
        qpos0: The initial reset pose (for pose regularization).
        
    Returns:
        reward (float): Total reward.
        terminated (bool): Whether the episode failed.
        termination_reason (str or None): Reason for failure.
        stats (dict): Breakdown of reward components for logging.
    """
    
    # --- State Extraction ---
    # Previous state info (for gradients/progress)
    prev_gravity_vec = state[24:27]
    prev_tilt = float(np.linalg.norm(prev_gravity_vec[:2]))
    prev_base_ang_vel = state[27:30]
    prev_ang_mag = float(np.linalg.norm(prev_base_ang_vel))
    prev_base_lin_vel = state[33:36]

    # Current state info
    gravity_vec = next_state[24:27]
    tilt = float(np.linalg.norm(gravity_vec[:2]))
    base_lin_vel = next_state[33:36]
    base_ang_vel = next_state[27:30]
    ang_mag = float(np.linalg.norm(base_ang_vel))
    robot_qvel = next_state[12:24]
    joint_qvel = robot_qvel[6:] if robot_qvel.shape[0] >= 7 else robot_qvel
    joint_pos = next_state[0:12] if next_state.shape[0] >= 12 else np.zeros(12, dtype=np.float32)
    
    # --- Urgency / Regime Scaling ---
    fall_urgency_now = float(np.clip(
        ang_mag * C.URGENCY_W_ANG + C.URGENCY_W_TILT * tilt, 
        0.0, C.URGENCY_MAX
    ))
    penalty_scale = 1.0 / (1.0 + fall_urgency_now)

    # --- 1. Positive Rewards ---
    
    # Survival & Upright
    r_survival = C.SURVIVAL_W
    r_upright = C.UPRIGHT_W * np.exp(-C.UPRIGHT_EXP_SCALE * tilt)
    if tilt > C.RECOVERY_TILT:
        r_upright *= C.UPRIGHT_BOOST_WHEN_TILTED

    # Recovery Progress
    r_progress = 0.0
    if tilt >= C.STABLE_TILT:
        tilt_progress = (prev_tilt - tilt)
        angvel_progress = (prev_ang_mag - ang_mag)
        r_progress = C.TILT_PROGRESS_W * float(tilt_progress) + C.ANGVEL_DAMP_W * float(angvel_progress)

    # Early Reaction (Move fast if falling starts)
    vel_mag = float(np.linalg.norm(joint_qvel))
    r_early = 0.0
    if tilt < C.STABLE_TILT and ang_mag > C.EARLY_REACT_ANG_THRESH:
        r_early = C.EARLY_REACT_W * vel_mag

    # CoM Acceleration (Reverse direction)
    r_com = 0.0
    if tilt > C.RECOVERY_TILT:
        dot_prod = float(np.dot(base_lin_vel[:2], prev_base_lin_vel[:2]))
        r_com = C.COM_ACCEL_W * max(0.0, -dot_prod)

    # Push Up (Extend bent knees)
    r_push = 0.0
    avg_knee_pos = float(np.mean(joint_pos[C.KNEE_LOCAL_IDXS]))
    if avg_knee_pos > 0.35: 
        avg_knee_vel = float(np.mean(robot_qvel[C.KNEE_LOCAL_IDXS]))
        if avg_knee_vel < 0: # Extension
            r_push = C.PUSH_UP_W * (-avg_knee_vel)
            
    # Quiet Stance (Settle down)
    r_quiet = 0.0
    if tilt < C.STABLE_TILT and ang_mag < 0.5:
        all_vel_sq = float(np.sum(np.square(robot_qvel)))
        r_quiet = C.QUIET_STANCE_W * np.exp(-1.0 * all_vel_sq)

    # Asymmetry
    r_asym = 0.0
    if fall_urgency_now > C.ASYM_URGENCY_THRESHOLD and action.shape[0] % 2 == 0:
        half = action.shape[0] // 2
        left = action[:half]
        right = action[half:]
        r_asym = C.ASYM_W * float(np.linalg.norm(left - right))

    r_base = 0.0
    if tilt > C.STABLE_TILT:
        # 1. Calculate raw velocity magnitude of legs
        leg_vel_mag = float(np.linalg.norm(robot_qvel[6:]))
        
        # 2. Apply weight
        raw_base_reward = C.BASE_WIDENING_W * leg_vel_mag
        
        # 3. CRITICAL: Clamp the reward! 
        # It should never exceed ~0.5 per step, otherwise it becomes the main goal.
        r_base = min(0.5, raw_base_reward)

    # --- 2. Penalties (Costs) ---

    # Drift
    c_drift = C.DRIFT_W * float(np.sum(np.square(base_lin_vel[:2])))

    # Joint Velocity
    excess_vel = max(0.0, vel_mag - C.VEL_THRESHOLD)
    c_vel = min(1.0, C.VEL_W * (excess_vel**2))

    # Angular Acceleration
    ang_acc = (base_ang_vel - prev_base_ang_vel) / C.DT
    c_ang_acc = min(1.0, C.ANG_ACC_W * float(np.sum(np.square(ang_acc))))

    # Knee Flexion Preference
    c_knee = 0.0
    for i in C.KNEE_LOCAL_IDXS:
        if 0 <= i < joint_pos.shape[0]:
            c_knee += float((joint_pos[i] - C.KNEE_REST_ANGLE) ** 2)
    c_knee = C.KNEE_W * c_knee

    # Knee Hyper-flexion Limit
    c_knee_hyper = 0.0
    for i in C.KNEE_LOCAL_IDXS:
        if 0 <= i < joint_pos.shape[0]:
            # Squared penalty grows fast, but weight needs to be high
            excess = max(0.0, float(joint_pos[i]) - C.KNEE_MAX_PREF)
            c_knee_hyper += excess**2
    c_knee_hyper = min(2.0, C.KNEE_HYPER_W * c_knee_hyper) # Allow higher max penalty

    # [NEW] Anti-Squat Penalty (Double Knee Bend)
    # Check if BOTH knees are bent beyond threshold
    c_squat = 0.0
    if len(C.KNEE_LOCAL_IDXS) >= 2:
        # Assuming index 0 is left, 1 is right in the config list
        k1_idx, k2_idx = C.KNEE_LOCAL_IDXS[0], C.KNEE_LOCAL_IDXS[1]
        k1_pos = float(joint_pos[k1_idx])
        k2_pos = float(joint_pos[k2_idx])
        
        if k1_pos > C.DOUBLE_KNEE_BEND_THRESHOLD and k2_pos > C.DOUBLE_KNEE_BEND_THRESHOLD:
            # Penalize the magnitude of the bend
            excess_squat = (k1_pos - C.DOUBLE_KNEE_BEND_THRESHOLD) + (k2_pos - C.DOUBLE_KNEE_BEND_THRESHOLD)
            c_squat = C.DOUBLE_KNEE_BEND_W * (excess_squat ** 2)

    # Pose Regularization
    qpos = next_state[:12]
    c_pose = C.POSE_W * float(np.sum(np.square(qpos - qpos0)))

    # Control Smoothness
    c_ctrl = C.CTRL_W * float(np.sum(np.square(action)))
    c_smooth = C.SMOOTH_W * float(np.sum(np.square(action - prev_action)))

    # Ankle/Hip Stability
    c_ankle_stab = 0.0
    c_hip_stab = 0.0
    if robot_qvel.shape[0] >= 12:
        ankle_vels = robot_qvel[C.ANKLE_LOCAL_IDXS]
        # raw calculation
        raw_ankle_cost = C.ANKLE_STAB_W * float(np.sum(np.square(ankle_vels)))
        # CLAMPING: Penalty cannot exceed 1.0 per step
        c_ankle_stab = min(1.0, raw_ankle_cost)
        
        hip_vels = robot_qvel[C.HIP_PITCH_IDXS]
        raw_hip_cost = C.HIP_STAB_W * float(np.sum(np.square(hip_vels)))
        # CLAMPING
        c_hip_stab = min(1.0, raw_hip_cost)
        
        # Unlock slightly during high urgency
        if tilt >= C.STABLE_TILT:
             c_ankle_stab *= C.ANKLE_EMERGENCY_SCALE
             c_hip_stab *= C.ANKLE_EMERGENCY_SCALE
    
    # [NEW] Leg Lift Penalty (Anti-Stork)
    # Penalize Hip Pitch if it exceeds the threshold (lifting leg too high)
    # Indices 0 and 6 are Hip Pitch
    c_leg_lift = 0.0
    for i in C.LEG_LIFT_IDX:
        if 0 <= i < joint_pos.shape[0]:
            # Usually positive hip pitch = lifting leg up. Check sign in your sim, 
            # but abs() is safer to prevent backward kicks too.
            excess_lift = max(0.0, abs(float(joint_pos[i])) - C.LEG_LIFT_THRESHOLD)
            c_leg_lift += excess_lift ** 2
    c_leg_lift = C.LEG_LIFT_W * c_leg_lift

    # [NEW] Leg Twist Penalty (Anti-Twist)
    # Penalize Hip Yaw (indices 2 and 8)
    c_leg_twist = 0.0
    for i in C.LEG_TWIST_IDX:
        if 0 <= i < joint_pos.shape[0]:
            excess_twist = max(0.0, abs(float(joint_pos[i])) - C.LEG_TWIST_THRESHOLD)
            c_leg_twist += excess_twist ** 2
    c_leg_twist = C.LEG_TWIST_W * c_leg_twist

    # --- 3. Scaling & Adjustments ---
    
    if tilt >= C.STABLE_TILT:
        c_vel *= 0.2
        c_ang_acc *= 0.2
        c_pose *= C.POSE_RECOVERY_SCALE
        c_knee *= 0.2
        c_knee_hyper *= 0.2
        # Unlock stabilizers during emergency
        c_ankle_stab *= C.ANKLE_EMERGENCY_SCALE
        c_hip_stab *= C.ANKLE_EMERGENCY_SCALE

    # Apply urgency scaling to all motion penalties
    c_vel *= penalty_scale
    c_ang_acc *= penalty_scale
    c_ctrl *= penalty_scale
    c_smooth *= penalty_scale
    c_pose *= penalty_scale
    c_knee *= penalty_scale
    c_knee_hyper *= penalty_scale

    # --- 4. Termination Logic ---
    terminated = False
    termination_reason = None
    
    if tilt > C.FALL_TILT:
        reward = -2.0
        terminated = True
        termination_reason = "fell_tilt"
    elif float(np.linalg.norm(base_lin_vel[:2])) > C.DRIFT_SPEED_XY:
        reward = -2.0
        terminated = True
        termination_reason = "drift_xy"
    else:
        # Final Summation
        reward = (
            r_survival + r_upright + r_asym + r_progress + r_early + r_com + r_push + r_quiet + r_base
            - c_drift - c_vel - c_ang_acc - c_knee - c_knee_hyper - c_pose
            - c_ctrl - c_smooth - c_ankle_stab - c_hip_stab - c_leg_lift - c_leg_twist - c_squat
        )

    # --- 5. Stats for Logging ---
    # We populate this dict so the main loop can accumulate it easily
    stats = {
        "survival": r_survival,
        "base_widening": r_base,
        "upright": r_upright,
        "asym": r_asym,
        "progress": r_progress,
        "early_react": r_early,
        "com_accel": r_com,
        "push_up": r_push,
        "quiet_stance": r_quiet,
        "drift": -c_drift,
        "vel": -c_vel,
        "ang_acc": -c_ang_acc,
        "knee": -c_knee,
        "knee_hyper": -c_knee_hyper,
        "pose": -c_pose,
        "ctrl": -c_ctrl,
        "smooth": -c_smooth,
        "ankle_stab": -c_ankle_stab,
        "hip_stab": -c_hip_stab,
        "leg_lift": -c_leg_lift,
        "leg_twist": -c_leg_twist
    }

    return reward, terminated, termination_reason, stats