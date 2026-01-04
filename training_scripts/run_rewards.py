# run_rewards.py
import numpy as np
import run_config as C

def calculate_reward(state, next_state, action, prev_action, qpos0):
    
    # --- State Extraction ---
    # Previous state info
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
    
    # --- Urgency Scaling ---
    fall_urgency_now = float(np.clip(
        ang_mag * C.URGENCY_W_ANG + C.URGENCY_W_TILT * tilt, 
        0.0, C.URGENCY_MAX
    ))
    penalty_scale = 1.0 / (1.0 + fall_urgency_now)

    # --- 1. Gait Analysis (The New Logic) ---
    
    # Calculate aggregate activity (velocity) for each leg
    # We use norm of joint velocities for hip/knee/ankle
    left_leg_vel = float(np.linalg.norm(robot_qvel[C.LEFT_LEG_IDXS]))
    right_leg_vel = float(np.linalg.norm(robot_qvel[C.RIGHT_LEG_IDXS]))

    # A. Alternation Reward (Velocity Asymmetry)
    # We want |L - R| to be high. One moves, one holds.
    r_alt = C.ALT_W * abs(left_leg_vel - right_leg_vel)

    # B. Scissor Reward (Position Asymmetry)
    # We want |L_Hip - R_Hip| to be high. Stride length.
    # Hip Pitch indices are 0 and 6
    left_hip = joint_pos[0]
    right_hip = joint_pos[6]
    r_scissor = C.SCISSOR_W * abs(left_hip - right_hip)

    # C. Double Support Penalty
    # We don't want both active (hopping) or both quiet (standing)
    c_double = C.DOUBLE_SUPPORT_W * min(left_leg_vel, right_leg_vel)

    # --- 2. Positive Rewards ---
    
    r_survival = C.SURVIVAL_W

    r_upright = C.UPRIGHT_W * np.exp(-C.UPRIGHT_EXP_SCALE * tilt)
    if tilt > C.RECOVERY_TILT:
        r_upright *= C.UPRIGHT_BOOST_WHEN_TILTED

    r_progress = 0.0
    if tilt >= C.STABLE_TILT:
        tilt_progress = (prev_tilt - tilt)
        angvel_progress = (prev_ang_mag - ang_mag)
        r_progress = C.TILT_PROGRESS_W * float(tilt_progress) + C.ANGVEL_DAMP_W * float(angvel_progress)

    # Push Up (Reduced weight)
    r_push = 0.0
    avg_knee_pos = float(np.mean(joint_pos[C.KNEE_LOCAL_IDXS]))
    if avg_knee_pos > 0.35: 
        avg_knee_vel = float(np.mean(robot_qvel[C.KNEE_LOCAL_IDXS]))
        if avg_knee_vel < 0:
            r_push = C.PUSH_UP_W * (-avg_knee_vel)
            
    # Quiet Stance
    r_quiet = 0.0
    if tilt < C.STABLE_TILT and ang_mag < 0.5:
        all_vel_sq = float(np.sum(np.square(robot_qvel)))
        r_quiet = C.QUIET_STANCE_W * np.exp(-1.0 * all_vel_sq)

    # --- The Velocity Gate ---
    vel_vec = next_state[33:36]
    ball_vec = next_state[48:51]
    dist_to_ball = float(np.linalg.norm(ball_vec))
    ball_dir = ball_vec / (dist_to_ball + 1e-6)
    
    vel_towards_ball = float(np.dot(vel_vec, ball_dir))
    rewardable_vel = min(vel_towards_ball, C.TARGET_RUN_SPEED)
    r_run = C.RUN_VEL_W * max(0.0, rewardable_vel)

    # CRITICAL: Gating Logic
    # If the most active leg is moving slower than THRESH,
    # it implies the robot is sliding/skating/statue-ing.
    # Crush the run reward to 20%.
    max_leg_activity = max(left_leg_vel, right_leg_vel)
    if max_leg_activity < C.WALK_ACT_THRESH:
        r_run *= 0.2

    # --- 3. Penalties ---

    c_drift = C.DRIFT_W * float(np.sum(np.square(base_lin_vel[:2])))

    vel_mag = float(np.linalg.norm(joint_qvel))
    excess_vel = max(0.0, vel_mag - C.VEL_THRESHOLD)
    c_vel = min(1.0, C.VEL_W * (excess_vel**2))

    ang_acc = (base_ang_vel - prev_base_ang_vel) / C.DT
    c_ang_acc = min(1.0, C.ANG_ACC_W * float(np.sum(np.square(ang_acc))))

    c_knee = 0.0
    for i in C.KNEE_LOCAL_IDXS:
        if 0 <= i < joint_pos.shape[0]:
            c_knee += float((joint_pos[i] - C.KNEE_REST_ANGLE) ** 2)
    c_knee = C.KNEE_W * c_knee

    c_knee_hyper = 0.0
    for i in C.KNEE_LOCAL_IDXS:
        if 0 <= i < joint_pos.shape[0]:
            excess = max(0.0, float(joint_pos[i]) - C.KNEE_MAX_PREF)
            c_knee_hyper += excess**2
    c_knee_hyper = min(2.0, C.KNEE_HYPER_W * c_knee_hyper)

    # Relaxed Anti-Squat
    c_squat = 0.0
    if len(C.KNEE_LOCAL_IDXS) >= 2:
        k1_idx, k2_idx = C.KNEE_LOCAL_IDXS[0], C.KNEE_LOCAL_IDXS[1]
        k1_pos = float(joint_pos[k1_idx])
        k2_pos = float(joint_pos[k2_idx])
        if k1_pos > C.DOUBLE_KNEE_BEND_THRESHOLD and k2_pos > C.DOUBLE_KNEE_BEND_THRESHOLD:
            excess_squat = (k1_pos - C.DOUBLE_KNEE_BEND_THRESHOLD) + (k2_pos - C.DOUBLE_KNEE_BEND_THRESHOLD)
            c_squat = C.DOUBLE_KNEE_BEND_W * (excess_squat ** 2)

    qpos = next_state[:12]
    c_pose = C.POSE_W * float(np.sum(np.square(qpos - qpos0)))
    c_ctrl = C.CTRL_W * float(np.sum(np.square(action)))
    c_smooth = C.SMOOTH_W * float(np.sum(np.square(action - prev_action)))

    c_ankle_stab = 0.0
    c_hip_stab = 0.0
    if robot_qvel.shape[0] >= 12:
        ankle_vels = robot_qvel[C.ANKLE_LOCAL_IDXS]
        raw_ankle_cost = C.ANKLE_STAB_W * float(np.sum(np.square(ankle_vels)))
        c_ankle_stab = min(1.0, raw_ankle_cost)
        hip_vels = robot_qvel[C.HIP_PITCH_IDXS]
        raw_hip_cost = C.HIP_STAB_W * float(np.sum(np.square(hip_vels)))
        c_hip_stab = min(1.0, raw_hip_cost)

    c_leg_lift = 0.0
    for i in C.LEG_LIFT_IDX:
        if 0 <= i < joint_pos.shape[0]:
            excess_lift = max(0.0, abs(float(joint_pos[i])) - C.LEG_LIFT_THRESHOLD)
            c_leg_lift += excess_lift ** 2
    c_leg_lift = C.LEG_LIFT_W * c_leg_lift

    c_leg_twist = 0.0
    for i in C.LEG_TWIST_IDX:
        if 0 <= i < joint_pos.shape[0]:
            excess_twist = max(0.0, abs(float(joint_pos[i])) - C.LEG_TWIST_THRESHOLD)
            c_leg_twist += excess_twist ** 2
    c_leg_twist = C.LEG_TWIST_W * c_leg_twist

    # --- Scaling ---
    # Relax constraints when walking (high activity) or tilted
    is_walking = max_leg_activity > C.WALK_ACT_THRESH
    
    if tilt >= C.STABLE_TILT or is_walking:
        c_vel *= 0.2
        c_ang_acc *= 0.2
        c_pose *= C.POSE_RECOVERY_SCALE
        c_knee *= 0.2
        c_knee_hyper *= 0.2
        c_ankle_stab *= C.ANKLE_EMERGENCY_SCALE
        c_hip_stab *= C.ANKLE_EMERGENCY_SCALE
        # Special: If walking, forgive leg lift slightly
        if is_walking:
             c_leg_lift *= 0.2

    c_vel *= penalty_scale
    c_ang_acc *= penalty_scale
    c_ctrl *= penalty_scale
    c_smooth *= penalty_scale
    c_pose *= penalty_scale
    c_knee *= penalty_scale
    c_knee_hyper *= penalty_scale

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
        reward = (
            r_survival + r_upright + r_progress + r_push + r_quiet + r_run 
            + r_alt + r_scissor # <--- Added new rewards
            - c_double          # <--- Added new penalty
            - c_drift - c_vel - c_ang_acc - c_knee - c_knee_hyper - c_squat - c_pose
            - c_ctrl - c_smooth - c_ankle_stab - c_hip_stab - c_leg_lift - c_leg_twist
        )

    stats = {
        "survival": r_survival,
        "run": r_run,
        "alt": r_alt,
        "scissor": r_scissor,
        "double_supp": -c_double,
        "upright": r_upright,
        "push_up": r_push,
        "leg_twist": -c_leg_twist,
        "leg_lift": -c_leg_lift,
        "squat": -c_squat,
        "drift": -c_drift,
        "pose": -c_pose,
    }

    return reward, terminated, termination_reason, stats