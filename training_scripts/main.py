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
    def get_task_onehot(self, info):
        idx = info.get('task_index', None)
        
        if idx is None:
            if "defender_xpos" in info:
                idx = 1 # LowerT1ObstaclePenaltyKick-v0
            elif "target_xpos_rel_robot" in info and "goalkeeper_team_0_xpos_rel_robot" not in info:
                idx = 2 # LowerT1KickToTarget-v0
            else:
                idx = 0 # LowerT1GoaliePenaltyKick-v0 (Default)

        if hasattr(idx, 'item'):
             idx = idx.item() if np.ndim(idx) == 0 else (idx[0] if len(idx) > 0 else 0)
        
        vec = np.zeros(3)
        safe_idx = int(idx)
        if 0 <= safe_idx < 3:
            vec[safe_idx] = 1.0
        return vec

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def safe_get(self, info, key, target_len):
        val = np.zeros((1, target_len)) 
        if key in info:
            data = info[key]
            data = np.array(data).flatten()
            current_len = len(data)
            limit = min(target_len, current_len)
            val[0, :limit] = data[:limit]
        return val

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        # Ensure dimensions for always-present keys
        keys_to_expand = [
            "robot_quat", "robot_gyro", "robot_accelerometer", "robot_velocimeter",
            "goal_team_0_rel_robot", "goal_team_1_rel_robot",
            "goal_team_0_rel_ball", "goal_team_1_rel_ball",
            "ball_xpos_rel_robot", "ball_velp_rel_robot", "ball_velr_rel_robot",
            "player_team"
        ]
        for k in keys_to_expand:
            if k in info and len(info[k].shape) == 1:
                info[k] = np.expand_dims(info[k], axis=0)

        # Variable keys
        goalkeeper_0_xpos = self.safe_get(info, "goalkeeper_team_0_xpos_rel_robot", 3)
        goalkeeper_0_velp = self.safe_get(info, "goalkeeper_team_0_velp_rel_robot", 3)
        goalkeeper_1_xpos = self.safe_get(info, "goalkeeper_team_1_xpos_rel_robot", 3)
        goalkeeper_1_velp = self.safe_get(info, "goalkeeper_team_1_velp_rel_robot", 3)
        target_xpos = self.safe_get(info, "target_xpos_rel_robot", 3)
        target_velp = self.safe_get(info, "target_velp_rel_robot", 3)
        defender_xpos = self.safe_get(info, "defender_xpos", 8)

        # Core State Extraction
        robot_qpos = obs[:,:12]
        robot_qvel = obs[:,12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        # Stack Construction (CRITICAL FOR INDICES)
        # 0-12: qpos
        # 12-24: qvel
        # 24-27: grav
        # 27-30: gyro
        # 30-33: accel
        # 33-36: velocimeter  <-- ROBOT VELOCITY
        # 36-39: goal0 rel robot
        # 39-42: goal1 rel robot
        # 42-45: goal0 rel ball
        # 45-48: goal1 rel ball
        # 48-51: ball rel robot <-- TARGET VECTOR
        
        obs = np.hstack((
            robot_qpos, 
            robot_qvel,
            project_gravity,
            base_ang_vel,
            info["robot_accelerometer"],
            info["robot_velocimeter"],
            info["goal_team_0_rel_robot"], 
            info["goal_team_1_rel_robot"], 
            info["goal_team_0_rel_ball"], 
            info["goal_team_1_rel_ball"], 
            info["ball_xpos_rel_robot"], 
            info["ball_velp_rel_robot"], 
            info["ball_velr_rel_robot"], 
            info["player_team"], 
            goalkeeper_0_xpos, 
            goalkeeper_0_velp, 
            goalkeeper_1_xpos, 
            goalkeeper_1_velp, 
            target_xpos, 
            target_velp, 
            defender_xpos,
            task_onehot
        ))

        return obs

# --- DYNAMIC FEATURE CALCULATION ---
# Moved to main execution block

## Create the model
# Moved to main execution block

## Define an action function
def get_action_function(action_space):
    def action_function(policy):
        expected_bounds = [-1, 1]
        action_percent = (policy - expected_bounds[0]) / (
            expected_bounds[1] - expected_bounds[0]
        )
        bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
        return (
                action_space.low
                + (action_space.high - action_space.low) * bounded_percent
        )
    return action_function


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