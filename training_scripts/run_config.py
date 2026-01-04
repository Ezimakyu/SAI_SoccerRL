# balance_config.py

# --- TRAINING HYPERPARAMETERS ---
TIMESTEPS = 400_000
BATCH_SIZE = 256
REPLAY_SIZE = 1_000_000
MAX_EPISODE_STEPS = 300
WARMUP_STEPS = 10_000
UPDATE_INTERVAL = 50
UPDATES_PER_INTERVAL = 50

# --- CONTROL / ACTION CONFIG ---
POLICY_ACTION_CLIP = 0.35
DYNAMIC_CLIP_W = 0.80
PERTURB_PROB = 0.02
PERTURB_POLICY_STD = 0.05
PERTURB_TORQUE_STD_FRAC = 0.15

# --- URGENCY & TILT REGIMES ---
STABLE_TILT = 0.05
RECOVERY_TILT = 0.20
UPRIGHT_BOOST_WHEN_TILTED = 1.0
URGENCY_MAX = 3.0
URGENCY_W_TILT = 1.0
URGENCY_W_ANG = 1.0
URGENCY_W_ANG_ACC = 0.5
DT = 0.02

# --- REWARD WEIGHTS ---
SURVIVAL_W = 2.5
UPRIGHT_W = 0.5
UPRIGHT_EXP_SCALE = 0.1
TILT_PROGRESS_W = 1.0
ANGVEL_DAMP_W = 0.2
EARLY_REACT_W = 0.02
EARLY_REACT_ANG_THRESH = 0.2
COM_ACCEL_W = 0.1
PUSH_UP_W = 0.1
QUIET_STANCE_W = 0.5
BASE_WIDENING_W = 0.5

# --- PENALTY WEIGHTS ---
VEL_THRESHOLD = 3.0
VEL_W = 0.002
DRIFT_W = 0.2
ANG_ACC_W = 0.01
CTRL_W = 0.01
SMOOTH_W = 0.01
POSE_W = 0.2
POSE_RECOVERY_SCALE = 0.05

# --- JOINT INDICES & WEIGHTS ---
# Left Knee: 3, Right Knee: 9
KNEE_LOCAL_IDXS = [3, 9]
KNEE_REST_ANGLE = 0.25
KNEE_W = 0.05
KNEE_MAX_PREF = 0.8
KNEE_HYPER_W = 5.0

# Hip Pitch: 0, 6
HIP_PITCH_IDXS = [0, 6]
HIP_STAB_W = 0.05

# Ankles: 4, 5, 10, 11
ANKLE_LOCAL_IDXS = [4, 5, 10, 11]
ANKLE_STAB_W = 0.001
ANKLE_EMERGENCY_SCALE = 0.1

# --- LEG CONTROL ---
# Penalize lifting the thigh too high (Hip Pitch)
# Threshold 0.5 rad is approx 28 degrees. Enough to step, but penalizes high marching.
LEG_LIFT_IDX = [0, 6]  # Left Hip Pitch, Right Hip Pitch
LEG_LIFT_THRESHOLD = 1.0
LEG_LIFT_W = 0.5 

# Penalize twisting the leg (Hip Yaw) - this stops the "helicopter" recovery
# Threshold 0.2 rad is approx 11 degrees.
LEG_TWIST_IDX = [2, 8] # Left Hip Yaw, Right Hip Yaw
LEG_TWIST_THRESHOLD = 0.2
LEG_TWIST_W = 1.0

# [NEW] RUNNING REWARDS
# Reward forward velocity (X-axis).
# 1.0 means at 1m/s it gets +1.0 reward per step.
RUN_VEL_W = 6.0
TARGET_RUN_SPEED = 1.5  # Cap reward at this speed (m/s) so it doesn't sprint uncontrollably

# Penalize bending both knees simultaneously (squatting)
DOUBLE_KNEE_BEND_THRESHOLD = 0.8 # approx 45 degrees
DOUBLE_KNEE_BEND_W = 0.5

# --- GAIT / ALTERNATION CONFIG ---
# Indices for calculating leg activity
# Assuming based on your robot: 
# Left Leg: Hip(0), Knee(3), Ankle(4,5) -> [0, 3, 4, 5]
# Right Leg: Hip(6), Knee(9), Ankle(10,11) -> [6, 9, 10, 11]
LEFT_LEG_IDXS = [0, 3, 4, 5]
RIGHT_LEG_IDXS = [6, 9, 10, 11]

# 1. Alternation (Velocity Asymmetry)
# Reward for one leg moving fast while other is slow
ALT_W = 0.1

# 2. Scissor (Position Asymmetry) - NEW IDEA
# Reward for splitting the legs (one forward, one back)
# Uses Hip Pitch indices [0] and [6]
SCISSOR_W = 1.0

# 3. Double Support Penalty
# Punish if both legs are moving fast (hopping) or both slow (standing)
DOUBLE_SUPPORT_W = 0.2

# 4. Velocity Gating
# If leg velocity max is below this, running reward is crushed.
WALK_ACT_THRESH = 1.0



# --- TERMINATION ---
FALL_TILT = 0.65
DRIFT_SPEED_XY = 2.0

# --- ASYMMETRY ---
ASYM_W = 0.1
ASYM_URGENCY_THRESHOLD = 1.0

# --- LOGGING ---
LOG_WINDOW_EPISODES = 20
LOG_EVERY_EPISODES = 10
USE_WANDB = False
DEBUG_REWARDS = True
DEBUG_PRINT_INTERVAL = 100