import numpy as np
import os

# --- 1. ENVIRONMENT ---
TIMESTEPS = 200000         
MAX_EPISODE_STEPS = 1000     # Allow long playbacks
DT = 0.02                    

# --- 2. TRAINING ---
BATCH_SIZE = 256
REPLAY_SIZE = 1000000
UPDATE_INTERVAL = 50          
UPDATES_PER_INTERVAL = 50
WARMUP_STEPS = 5000          
LOG_EVERY_EPISODES = 20
SAVE_EVERY_EPISODES = 20
SAVE_EVERY_EPISODES_LONGER = 60
CHECKPOINT_FILENAME = "sac_imitate_checkpoint_base1.pth"
USE_WANDB = False 

# --- 3. REWARDS ---
# High reward for matching the reference pose
TRACKING_QPOS_W = 10.0       
TRACKING_VEL_W = 1.0         # Reward matching velocity (if qvel exists)

# Stability helpers
ALIVE_W = 5.0                
ENERGY_W = 0.001  

# --- 5. DATASET CONFIG ---
# List of .npz files to cycle through

# --- 4. DATASET ---
# Folder containing your .npz files
TRAJECTORY_DIR = "./training_scripts/demonstrations" 

# If empty, script will auto-find *.npz in TRAJECTORY_DIR
TRAJECTORY_FILES = [
    "jogging.npz", 
    "running.npz"
    # "training_scripts/demonstrations/walk.npz", 
] 

# Curriculum: How many steps to train on a file before swapping
STEPS_PER_TRAJECTORY = 20000 

# --- 5. INDICES ---
# Standard MuJoCo Humanoid/Booster indices
# 0-2: Root Pos (x,y,z)
# 3-7: Root Quat (w,x,y,z)
# 7+: Joint Angles
ROOT_Z_IDX = 2
JOINT_START_IDX = 7

# --- 6. INDICES ---
# Standard MuJoCo Humanoid indices (Adjust if your XML differs)
# 0-2: Root Pos (x,y,z)
# 3-6: Root Quat (w,x,y,z)
# 7+: Joint Angles
ROOT_POS_IDXS = [0, 1, 2]
ROOT_QUAT_IDXS = [3, 4, 5, 6]
JOINT_START_IDX = 7

# --- 5. INITIALIZATION ---
# [CRITICAL] Set this to True to spawn robot in the ref pose
INIT_FROM_TRAJECTORY = True 
# Randomly start somewhat into the file (0.0 to 0.9 of duration)
RANDOM_START_PHASE = True

# Debug
DEBUG_REWARDS = True
DEBUG_VALUES = False     # [NEW] Print raw values to console to diagnose crashes