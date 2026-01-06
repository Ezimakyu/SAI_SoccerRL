import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from sai_rl import SAIClient

import sys
import os

# Add root directory to path to find sai_patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Allow importing from current directory when running as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sai_patch
except ImportError:
    pass

from sac import SAC
from training import training_loop

# Define environment IDs for the 3 tasks
ENV_IDS = [
    "LowerT1GoaliePenaltyKick-v0",
    "LowerT1ObstaclePenaltyKick-v0",
    "LowerT1KickToTarget-v0"
]

NUM_TIMESTEPS = 2000000

class MultiTaskEnv(gym.Env):
    def __init__(self, env_ids):
        self.envs = [gym.make(id) for id in env_ids]
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.current_env_idx = 0

    def reset(self, **kwargs):
        self.current_env_idx = np.random.randint(len(self.envs))
        obs, info = self.envs[self.current_env_idx].reset(**kwargs)
        info['task_index'] = self.current_env_idx
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.envs[self.current_env_idx].step(action)
        info['task_index'] = self.current_env_idx
        return obs, reward, terminated, truncated, info

    def close(self):
        for env in self.envs:
            env.close()

## Initialize the SAI client
# Moved to main execution block to avoid running on import

## Make the environment (Use MultiTaskEnv for training)
# Moved to main execution block

class Preprocessor():
    def get_task_onehot(self, info, batch_size):
        idx = info.get('task_index', None)
        
        if idx is None:
            if "defender_xpos" in info:
                idx = 1 # LowerT1ObstaclePenaltyKick-v0
            elif "target_xpos_rel_robot" in info and "goalkeeper_team_0_xpos_rel_robot" not in info:
                idx = 2 # LowerT1KickToTarget-v0
            else:
                idx = 0 # LowerT1GoaliePenaltyKick-v0 (Default)

        # Force conversion to a flat array and grab the first element safely
        try:
            raw_idx = np.array(idx).reshape(-1)[0]
            safe_idx = int(raw_idx)
        except:
            safe_idx = 0
        
        # Create a batch of one-hot vectors
        vecs = np.zeros((batch_size, 3), dtype=np.float32)
        if 0 <= safe_idx < 3:
            vecs[:, safe_idx] = 1.0
        return vecs

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        # q is (batch, 4), v is (batch, 3)
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.sum(q_vec * v, axis=1, keepdims=True) * 2.0)    
        return a - b + c 

    def safe_get(self, info, key, target_len, batch_size):
        # Initialize zero array for the full batch
        val = np.zeros((batch_size, target_len), dtype=np.float32) 
        if key in info:
            data = np.array(info[key])
            
            # Ensure data is 2D (batch, features)
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            # If info provides 1 row but we have 16 envs, tile it
            if data.shape[0] == 1 and batch_size > 1:
                data = np.tile(data, (batch_size, 1))
            
            # Copy data into our pre-allocated zero array
            limit_rows = min(batch_size, data.shape[0])
            limit_cols = min(target_len, data.shape[1])
            val[:limit_rows, :limit_cols] = data[:limit_rows, :limit_cols]
            
        return val

    def modify_state(self, obs, info):
        # Ensure obs is 2D (batch, features)
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
            
        batch_size = obs.shape[0]

        # 1. Task One-Hot
        task_onehot = self.get_task_onehot(info, batch_size)
        
        # 2. Variable info keys (Goalkeepers, targets, etc.)
        goalkeeper_0_xpos = self.safe_get(info, "goalkeeper_team_0_xpos_rel_robot", 3, batch_size)
        goalkeeper_0_velp = self.safe_get(info, "goalkeeper_team_0_velp_rel_robot", 3, batch_size)
        goalkeeper_1_xpos = self.safe_get(info, "goalkeeper_team_1_xpos_rel_robot", 3, batch_size)
        goalkeeper_1_velp = self.safe_get(info, "goalkeeper_team_1_velp_rel_robot", 3, batch_size)
        target_xpos = self.safe_get(info, "target_xpos_rel_robot", 3, batch_size)
        target_velp = self.safe_get(info, "target_velp_rel_robot", 3, batch_size)
        defender_xpos = self.safe_get(info, "defender_xpos", 8, batch_size)

        # 3. Core Robot State
        robot_qpos = obs[:, :12]
        robot_qvel = obs[:, 12:24]
        
        # Helper to ensure info keys are 2D
        def get_info_2d(key, width):
            if key in info:
                d = np.array(info[key])
                if len(d.shape) == 1: d = d.reshape(1, -1)
                if d.shape[0] == 1 and batch_size > 1: d = np.tile(d, (batch_size, 1))
                return d
            return np.zeros((batch_size, width), dtype=np.float32)

        quat = get_info_2d("robot_quat", 4)
        base_ang_vel = get_info_2d("robot_gyro", 3)
        accel = get_info_2d("robot_accelerometer", 3)
        velocimeter = get_info_2d("robot_velocimeter", 3)
        
        # Gravity Projection (Batch Aware)
        gravity_base = np.tile(np.array([0.0, 0.0, -1.0]), (batch_size, 1))
        project_gravity = self.quat_rotate_inverse(quat, gravity_base)
        
        # Always-present relative vectors
        goal0_rel_robot = get_info_2d("goal_team_0_rel_robot", 3)
        goal1_rel_robot = get_info_2d("goal_team_1_rel_robot", 3)
        goal0_rel_ball = get_info_2d("goal_team_0_rel_ball", 3)
        goal1_rel_ball = get_info_2d("goal_team_1_rel_ball", 3)
        ball_rel_robot = get_info_2d("ball_xpos_rel_robot", 3)
        ball_velp_robot = get_info_2d("ball_velp_rel_robot", 3)
        ball_velr_robot = get_info_2d("ball_velr_rel_robot", 3)
        player_team = get_info_2d("player_team", 1)

        # 4. [NEW] Phase Signal (DeepMind Input)
        # We need 'episode_time' in info to calculate this. 
        # If missing, it defaults to 0.0 (static phase).
        gait_freq = 1.5 # Must match run_config.py
        episode_time = 0.0
        
        if "episode_time" in info:
            et = info["episode_time"]
            try:
                # Handle scalar or single-element array
                episode_time = float(np.mean(et)) 
            except:
                episode_time = 0.0

        phase = (episode_time * gait_freq) % 1.0
        rads = phase * 2 * np.pi
        
        # DeepMind feeds [cos(phase), sin(phase)]
        phase_signal = np.array([np.cos(rads), np.sin(rads)], dtype=np.float32)
        phase_signal = np.tile(phase_signal, (batch_size, 1))

        # 5. Horizontal Stack
        obs_final = np.hstack((
            robot_qpos, 
            robot_qvel,
            project_gravity,
            base_ang_vel,
            accel,
            velocimeter,
            goal0_rel_robot, 
            goal1_rel_robot, 
            goal0_rel_ball, 
            goal1_rel_ball, 
            ball_rel_robot, 
            ball_velp_robot, 
            ball_velr_robot, 
            player_team, 
            goalkeeper_0_xpos, 
            goalkeeper_0_velp, 
            goalkeeper_1_xpos, 
            goalkeeper_1_velp, 
            target_xpos, 
            target_velp, 
            defender_xpos,
            task_onehot,
            phase_signal # <--- Added DeepMind Phase Signal
        ))

        return obs_final.astype(np.float32)

# --- DYNAMIC FEATURE CALCULATION ---
# Moved to main execution block

## Create the model
# Moved to main execution block

## Define an action function
def get_action_function():
    """
    Returns the action scaling function.
    Hardcodes limits to avoid dependency on global 'action_space'.
    """
    # Define limits inside the function (Self-contained)
    # These match the standard Booster/Humanoid torque limits
    low = np.array([-45.0] * 12, dtype=np.float32) # Adjust if your env uses different limits
    high = np.array([45.0] * 12, dtype=np.float32)
    
    def action_function(policy):
        # Policy is usually tanh output [-1, 1]
        expected_bounds = [-1.0, 1.0]
        
        # Normalize to [0, 1]
        action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
        
        # Clip strictly
        bounded_percent = np.minimum(np.maximum(action_percent, 0.0), 1.0)
        
        # Scale to Torque
        return low + (high - low) * bounded_percent
    return action_function

# What to submit to SAI
# def action_function(policy):
#     """
#     Self-contained torque scaler for submission.
#     Handles both single inputs and vectorized batches (e.g., 16 parallel robots).
#     Maps neural net output [-1, 1] to torque limits [-45, 45].
#     """
#     # Physical limits found from environment check
#     low = -45.0
#     high = 45.0
    
#     # 1. Ensure input is a numpy array
#     policy = np.array(policy)
    
#     # 2. Map policy (-1 to 1) to 0 to 1
#     # This works for both scalars, vectors, and matrices (batching)
#     action_percent = (policy - (-1.0)) / (1.0 - (-1.0))
    
#     # 3. Strictly clip to [0, 1] range
#     # np.clip handles multi-dimensional arrays automatically
#     bounded_percent = np.clip(action_percent, 0.0, 1.0)
    
#     # 4. Scale to torque: -45 + (90 * percent)
#     # Result will have the same shape as the input 'policy'
#     return low + (high - low) * bounded_percent

## Train the model
if __name__ == "__main__":
    # Initialize the SAI client
    sai = SAIClient(
        comp_id="lower-t1-penalty-kick-goalie",
        api_key="sai_ddqEmPy1JIeQoGSI72BcdGUePbVdYtSj"
    )

    # Make the environment
    env = MultiTaskEnv(ENV_IDS)

    # Create a dummy environment just to check the observation size
    dummy_obs, dummy_info = env.reset()
    dummy_preprocessor = Preprocessor()

    # Run the preprocessor once to see exactly how big the output is
    processed_state = dummy_preprocessor.modify_state(dummy_obs, dummy_info)
    REAL_N_FEATURES = processed_state.shape[1] 

    print(f"Calculated Input Features: {REAL_N_FEATURES}")

    # Create the model
    # model = SAC(
    #     n_features=REAL_N_FEATURES,  
    #     action_space=env.action_space, 
    #     neurons=[400, 300], 
    #     activation_function=F.relu,
    #     learning_rate=0.0001,
    # )

    # Create action function closure
    action_function = get_action_function(env.action_space)

    training_loop(env, model, action_function, Preprocessor, timesteps=NUM_TIMESTEPS)

    ## Watch & Benchmark
    # Only initialize SAIClient if we have credentials and want to upload/verify
    try:
        print("Initializing SAIClient for watch/benchmark...")
        # sai client already initialized above
        
        ## Watch
        sai.watch(model, action_function, Preprocessor)

        ## Benchmark the model locally
        sai.benchmark(model, action_function, Preprocessor)
    except Exception as e:
        print(f"\nSkipping SAI watch/benchmark due to error (likely missing API key): {e}")
        print("Training completed successfully. Models are saved locally.")