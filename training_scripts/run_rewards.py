import numpy as np
import run_config as C

def get_phase_targets(episode_time, freq):
    """
    Returns the target Knee Angles based on a 1.5Hz clock.
    0 = Straight Leg, 1.0 = Bent Knee (Swing)
    """
    # Phase moves from 0.0 to 1.0 repeatedly
    phase = (episode_time * freq) % 1.0
    
    # Left Leg swings during first half (0.0 - 0.5)
    # Right Leg swings during second half (0.5 - 1.0)
    
    # We use a sine wave to make the motion smooth
    # sin(0) = 0, sin(pi/2) = 1, sin(pi) = 0
    
    if phase < 0.5:
        # Left Leg Swing Phase
        # Map 0.0-0.5 to 0-PI for sine wave
        swing_progress = (phase / 0.5) * np.pi 
        target_left_knee = np.sin(swing_progress) * C.PHASE_KNEE_AMPLITUDE
        target_right_knee = 0.0 # Right leg should be straight (Stance)
    else:
        # Right Leg Swing Phase
        swing_progress = ((phase - 0.5) / 0.5) * np.pi
        target_left_knee = 0.0 # Left leg should be straight (Stance)
        target_right_knee = np.sin(swing_progress) * C.PHASE_KNEE_AMPLITUDE
        
    return target_left_knee, target_right_knee

def calculate_reward(state, next_state, action, prev_action, qpos0, prev_base_ang_vel, episode_time, true_height):
    """
    Calculates reward with Phase-Based Gait enforcement.
    """
    
    # --- 1. EXTRACT STATE ---
    # Based on your indices, state 0-11 are likely Joint Positions
    # 3 = Left Knee, 9 = Right Knee
    current_left_knee = next_state[3]
    current_right_knee = next_state[9]
    
    # Gravity Vector (approx indices 24-26 usually)
    # If standard MuJoCo: Z-axis of body frame in world
    projected_gravity = next_state[24:27] 
    
    # CALCULATE LEAN (Pitch)
    # If gravity.x is large positive/negative, we are leaning forward/back
    # "Zombie Lean" usually means leaning BACK (positive pitch in some frames)
    # We just penalize deviation from vertical generally.
    lean_magnitude = np.linalg.norm(projected_gravity[:2])

    vel_x = next_state[33] # Velocimeter X
    vel_y = next_state[34] # Velocimeter Y

    # --- 2. TERMINATION ---
    terminated = False
    reason = ""
    
    # Use True Height from Physics
    if true_height < 0.25:
        terminated = True
        reason = "Fell Over (Height)"
    
    # Tilt Fail
    if lean_magnitude > C.FALL_TILT:
        terminated = True
        reason = "Fell Over (Tilt)"

    # --- 3. REWARDS ---

    # A. SURVIVAL
    r_survival = C.SURVIVAL_W

    # B. HEIGHT (Target 0.62m)
    height_error = true_height - C.TARGET_HEIGHT
    r_height = np.exp(-np.square(height_error) / 0.05) * C.HEIGHT_W

    # C. TRACKING VELOCITY (Target 1.5 m/s)
    vel_error = vel_x - C.TARGET_VEL_X
    r_run = np.exp(-np.square(vel_error) / 0.5) * C.TRACKING_VEL_W

    # D. PHASE REWARD (The Zombie Fix)
    # ---------------------------------------------------------------------
    # 1. Get where the knees SHOULD be right now
    t_left, t_right = get_phase_targets(episode_time, C.GAIT_FREQ)
    
    # 2. Compare with where knees ARE
    # Note: Knees usually bend negatively or positively depending on model.
    # We take absolute value to be safe, assuming 0 is straight.
    diff_left = np.abs(current_left_knee) - t_left
    diff_right = np.abs(current_right_knee) - t_right
    
    # 3. Gaussian Reward for matching the rhythm
    error_phase = np.square(diff_left) + np.square(diff_right)
    r_phase = np.exp(-error_phase / 0.5) * C.PHASE_W
    # ---------------------------------------------------------------------

    # E. REGULARIZATION
    r_energy = -np.sum(np.square(action)) * C.ENERGY_W
    
    # Penalize Leaning Back specifically if possible, otherwise general upright
    r_upright = np.exp(-np.square(lean_magnitude) / 0.1) * C.UPRIGHT_W

    # --- 4. TOTAL ---
    # Gate the Run reward: No points for sliding on butt
    upright_gate = 1.0 if true_height > 0.4 else 0.0
    
    total_reward = (
        r_survival +
        r_height +
        (r_run * upright_gate) +
        (r_phase * upright_gate) + # Only pay for rhythm if standing
        r_upright +
        r_energy
    )

    stats = {
        "reward_total": total_reward,
        "r_height": r_height,
        "r_run": r_run,
        "r_phase": r_phase,
        "actual_vel_x": vel_x,
        "actual_height": true_height,
        "target_knee_L": t_left
    }

    return total_reward, terminated, reason, stats