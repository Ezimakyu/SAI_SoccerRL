import numpy as np

# --- 1. ENVIRONMENT SETTINGS ---
TIMESTEPS = 3000000          # Total training steps
MAX_EPISODE_STEPS = 500      # 10 seconds at 50Hz
DT = 0.02                    # Simulation timestep (50Hz)

# --- 2. TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256
REPLAY_SIZE = 1000000
UPDATE_INTERVAL = 1          # Train every step
UPDATES_PER_INTERVAL = 1
WARMUP_STEPS = 2000          # Random actions before training
LOG_EVERY_EPISODES = 5
LOG_WINDOW_EPISODES = 20
USE_WANDB = False            # Kept off as requested

# --- 3. REWARD WEIGHTS ---

# A. DeepMind "Locomotion" Core
# -----------------------------
TRACKING_VEL_W = 1.5         # Match target velocity
PHASE_W = 0.8                # [NEW] Strong bonus for rhythmic stepping (The "Anti-Zombie" fix)
FEET_AIR_TIME_W = 0.5        # Keep feet up during swing
FEET_HEIGHT_W = -1.0         # Don't stomp too high

# --- HEIGHT CONTROL ---
TARGET_HEIGHT = 0.62         # Desired height of torso (meters)
HEIGHT_W = 3.0               # Strong bonus for being near this height

# B. Safety & Regularization
# -----------------------------
SURVIVAL_W = 0.2            # Small bonus for staying alive
UPRIGHT_W = 1.0              # General penalty for tilting
LEAN_BACK_PENALTY_W = 1.0    # [NEW] Specific penalty for "Zombie" backward lean
DRIFT_W = -0.5               # Penalty for moving sideways
ENERGY_W = -0.001            # Penalty for high torque

# C. Smoothness
# -----------------------------
SMOOTH_W = -0.1              # Penalty for changing actions too fast (Jitter)
ANG_ACC_W = -0.05            # Penalty for jerky body rotation

# D. Legacy Heuristics (Disabled)
# -----------------------------
SCISSOR_W = 0.0
ALT_W = 0.0
STILL_PENALTY_W = 0.5        # Penalty for standing still
GAIT_PHASE_W = 0.0           # Replaced by new PHASE_W logic

# --- 4. GAIT PARAMETERS ---
GAIT_FREQ = 1.5              # 1.5 Hz stepping frequency
PHASE_KNEE_AMPLITUDE = 1.5   # [NEW] Target knee bend (radians) during swing phase
TARGET_VEL_X = 1.5           # Target running speed (m/s)

# Safety Limits
UPRIGHT_THRESHOLD = 0.2      # Rads from vertical considered "perfect"
FALLING_THRESHOLD = 0.7      # Rads from vertical considered "fallen"
FALL_TILT = 0.8              # Terminal condition

# --- 5. INDICES (Updated based on check_indices.py) ---
# 3 = Left Knee, 9 = Right Knee
# 0 = Left Hip Y, 6 = Right Hip Y
KNEE_IDXS = [3, 9]           
HIP_PITCH_IDXS = [0, 6]      
KNEE_LOCAL_IDXS = [3, 9]     # Kept for backward compatibility if needed

# --- 6. ACTION PROCESSING ---
POLICY_ACTION_CLIP = 1.0     
PERTURB_PROB = 0.0           
PERTURB_POLICY_STD = 0.1
DEBUG_REWARDS = True         
DEBUG_PRINT_INTERVAL = 5