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
SAVE_EVERY_EPISODES_LONGER = 200
USE_WANDB = False            

# --- 3. REWARD WEIGHTS ---

# [NEW] Knees Forward (Stop the twisting)
# Penalize Hip Yaw if it rotates away from 0.0
HIP_YAW_W = 3.0

# [NEW] Momentum Builder
# Rewards raw forward velocity linearly.
# Helps the robot find the gradient to start moving from a standstill.
FORWARD_MOMENTUM_W = 4.0

# [NEW] KICKING & PRECISION WEIGHTS
# 1. Approach: Reward getting closer to the ball (Guide rail)
BALL_APPROACH_W = 2.0 

# 2. Bulldozer: Reward raw ball velocity (Task 1/2)
BALL_VEL_W = 1.0

# 3. Precision: Reward ball moving TOWARDS the target (Task 3)
# This is the "Sniper" reward. High payoff for accuracy.
TARGET_PRECISION_W = 5.0

# A. The Core Drivers
TRACKING_VEL_W = 8.0         
TRACKING_ANG_VEL_W = 2.0     
PHASE_W = 4.0                
ALIVE_W = 5.0                

# B. DeepMind Regularization
DOF_POS_W = 3.0              
ORN_W = 1.0                  
LIN_VEL_Z_W = 1.5            
ANG_VEL_XY_W = 0.7           

# C. Posture Shaping
TARGET_FORWARD_TILT = 0.3
FORWARD_TILT_W = 3.0         
BACKWARD_TILT_PENALTY = 9.0  
SIDE_TILT_PENALTY = 1.0      

# D. Leg Configuration
TARGET_HIP_SPLIT = 0.7       
SPLIT_STANCE_W = 4.0         

# [FIX] Replaced negative Splay penalty with positive Width target
# This prevents the robot from crossing legs (tripping) by encouraging a 0.15 rad spread.
STANCE_WIDTH_W = 2.0         
TARGET_HIP_WIDTH = 0.30      

# Stance Straightening
STANCE_STRAIGHT_W = 0.1      
KNEE_BEND_W = 3.0      

# E. Height & Feet
TARGET_HEIGHT = 0.58         # Height is 0.62, a little less for allowing bend-down   
HEIGHT_W = 3.0               
FEET_AIR_TIME_W = 1.0        # Reward for lifting feet during swing

# DeepMind Additions
MAX_FOOT_HEIGHT = 0.12       
FEET_PHASE_W = 5.0           
FEET_SLIP_W = -0.25          

# Anti-Hopping
DOUBLE_AIR_PENALTY_W = -5.0 

# F. Efficiency
ENERGY_W = 0.005             
ACTION_RATE_W = 0.2          

# --- 4. GAIT & PHYSICS ---
GAIT_FREQ = 3.0               
PHASE_KNEE_AMPLITUDE = 1.0   
TARGET_VEL_X = 1.0           
LEAN_THRESHOLD = 0.5         

# --- 5. CHECKPOINT ---
CHECKPOINT_FILENAME = "sac_run_checkpoint_base7.pth"

# --- 6. INDICES ---
HIP_PITCH_IDXS = [0, 6]      
HIP_ROLL_IDXS  = [1, 7]      
HIP_YAW_IDXS   = [2, 8]      
KNEE_IDXS      = [3, 9]           

# --- 7. DEBUG ---
POLICY_ACTION_CLIP = 1.0     
DEBUG_REWARDS = True

# --- 8. PENALTIES ---
BACKWARDS_PENALTY_W = -6.0