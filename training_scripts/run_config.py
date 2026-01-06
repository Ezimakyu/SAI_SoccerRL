import numpy as np

# --- 1. ENVIRONMENT SETTINGS ---
TIMESTEPS = 10000000         # DeepMind runs are long
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
SAVE_EVERY_EPISODES_LONGER = 300
USE_WANDB = False            

# --- 3. REWARD WEIGHTS (Hybrid: DeepMind Stability + Your Shaping) ---

# A. The Core Drivers
TRACKING_VEL_W = 5.0         # Your aggressive velocity weight
TRACKING_ANG_VEL_W = 1.0     # Penalize spinning
PHASE_W = 3.0                
ALIVE_W = 5.0                

# B. DeepMind Regularization (The New Stabilizers)
DOF_POS_W = 1.0              # Penalizes deviation from default "Standing Pose" (qpos0)
ORN_W = 1.0                  # Gravity alignment
LIN_VEL_Z_W = 0.5            # Penalize vertical bouncing
ANG_VEL_XY_W = 0.3           # Penalize body wobble

# C. Your "Anti-Zombie" Posture Shaping
TARGET_FORWARD_TILT = 0.4    
FORWARD_TILT_W = 3.0         
BACKWARD_TILT_PENALTY = 6.0  
SIDE_TILT_PENALTY = 0.3      

# D. Leg Configuration & Constraints
TARGET_HIP_SPLIT = 0.6       
SPLIT_STANCE_W = 3.0         

# Explicit Splay Penalty
HIP_SPLAY_PENALTY_W = -3.0   

# Stance Straightening
STANCE_STRAIGHT_W = 1.0      
KNEE_BEND_W = 1.0            

# E. Height & Feet
TARGET_HEIGHT = 0.62         
HEIGHT_W = 3.0               
FEET_AIR_TIME_W = 1.5       # Reward for lifting feet during swing

# [DEEPMIND ADDITIONS] 
# Added these so we can implement the full DeepMind logic in the next step.
MAX_FOOT_HEIGHT = 0.12       # [FIX] Added missing constant (Target swing height in meters)
FEET_PHASE_W = 1.0           # Rewards tracking the specific sine-wave arc (smooth landing)
FEET_SLIP_W = -0.25          # Penalizes feet sliding on the ground (traction)

# [NEW] Anti-Hopping: Heavy penalty if both feet are off ground (knees bent)
DOUBLE_AIR_PENALTY_W = -10.0 

# F. Efficiency & Smoothness
ENERGY_W = 0.005             
ACTION_RATE_W = 0.2          

# --- 4. GAIT & PHYSICS ---
GAIT_FREQ = 1.5              
PHASE_KNEE_AMPLITUDE = 1.4   
TARGET_VEL_X = 1.5           
LEAN_THRESHOLD = 0.6         

# --- 5. CHECKPOINT ---
CHECKPOINT_FILENAME = "sac_run_checkpoint_base1.pth"

# --- 6. INDICES (Your Specific Mapping) ---
HIP_PITCH_IDXS = [0, 6]      
HIP_ROLL_IDXS  = [1, 7]      
HIP_YAW_IDXS   = [2, 8]      
KNEE_IDXS      = [3, 9]           

# --- 7. DEBUG ---
POLICY_ACTION_CLIP = 1.0     
DEBUG_REWARDS = True

# --- 8. PENALTIES ---
BACKWARDS_PENALTY_W = -3.0