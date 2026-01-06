import numpy as np

# --- 1. ENVIRONMENT SETTINGS ---
TIMESTEPS = 5000000          
MAX_EPISODE_STEPS = 500      # 10 seconds at 50Hz
DT = 0.02                    # Simulation timestep (50Hz)

# --- 2. TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 256
REPLAY_SIZE = 1000000
UPDATE_INTERVAL = 50          
UPDATES_PER_INTERVAL = 50
WARMUP_STEPS = 5000          
LOG_EVERY_EPISODES = 5
LOG_WINDOW_EPISODES = 50
SAVE_EVERY_EPISODES = 50     
USE_WANDB = False            

# --- 3. REWARD WEIGHTS ---

# A. The Core Drivers
TRACKING_VEL_W = 4.0         
PHASE_W = 1.2                
ALIVE_W = 5.0                

# B. Stability & Posture
TARGET_FORWARD_TILT = 0.4    # ~23 deg forward lean
FORWARD_TILT_W = 3.0         
BACKWARD_TILT_PENALTY = 6.0  
SIDE_TILT_PENALTY = 1.0      

UPRIGHT_W = 1.0              
LIN_VEL_Z_W = -0.1           
ANG_VEL_XY_W = -0.02         

# C. Leg Configuration
TARGET_HIP_SPLIT = 0.6       
SPLIT_STANCE_W = 3.0         

# [NEW] Anti-Splay (Constrain hips to forward motion only)
# Penalize Hip Roll (1, 7) and Yaw (2, 8) if they deviate from 0.
HIP_SPLAY_PENALTY_W = -2.0   

# D. Height & Feet
TARGET_HEIGHT = 0.62         
HEIGHT_W = 3.0               
FEET_AIR_TIME_W = 1.0        

# E. Style
KNEE_BEND_W = 0.8            

# F. Efficiency
ENERGY_W = -0.001            
ACTION_RATE_W = -0.05        

# --- 4. GAIT & PHYSICS ---
GAIT_FREQ = 1.5              
PHASE_KNEE_AMPLITUDE = 1.4   
TARGET_VEL_X = 1.5           
LEAN_THRESHOLD = 0.6         

# --- 5. CHECKPOINT ---
CHECKPOINT_FILENAME = "sac_run_checkpoint_spamtrain9.pth"

# --- 6. INDICES ---
# Y-axis = Pitch (Swinging forward/backward)
HIP_PITCH_IDXS = [0, 6]      # hip_y_left (0), hip_y_right (6)
KNEE_IDXS      = [3, 9]      # knee_y_left (3), knee_y_right (9)

# X-axis = Roll (Splaying legs outward/inward)
HIP_ROLL_IDXS  = [1, 7]      # hip_x_left (1), hip_x_right (7)

# Z-axis = Yaw (Twisting the leg)
HIP_YAW_IDXS   = [2, 8]      # hip_z_left (2), hip_z_right (8) 

# --- 7. DEBUG ---
POLICY_ACTION_CLIP = 1.0     
DEBUG_REWARDS = True

# --- 8. PENALTIES ---
BACKWARDS_PENALTY_W = -2.0