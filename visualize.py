# Reward breakdown is likely inaccurate
# Visualize stand: python visualize.py
# Visualize balance: python visualize.py --model_type sac --model_path training_scripts/balance_models/sac_balance_checkpoint.pth --policy_clip 0.35
# Visualize run: python visualize.py --model_type sac --model_path training_scripts/run_models/sac_run_checkpoint.pth

import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
import sys
import os
import argparse

# --- Path Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_scripts'))

from sai_rl import SAIClient
from sac import SAC
from training_scripts.main import Preprocessor, get_action_function, ENV_IDS

# --- Import Custom Reward Logic ---
import run_config as C
import run_rewards

# --- Configuration ---
DEFAULT_MODEL_PATH = "training_scripts/run_models2/sac_run_checkpoint_astep.pth"
DEFAULT_MODEL_TYPE = "sac" 
DEFAULT_TASK_INDEX = 0
DEFAULT_POLICY_CLIP = 0.35 

def get_mujoco_data(env):
    """
    Robustly finds the MuJoCo 'data' object (containing qpos/qvel) 
    by digging through Gym/Gymnasium wrappers.
    """
    if hasattr(env, "unwrapped"):
        if hasattr(env.unwrapped, "data"):
            return env.unwrapped.data
    
    curr = env
    for _ in range(10): 
        if hasattr(curr, "data") and hasattr(curr.data, "qpos"):
            return curr.data
        if hasattr(curr, "physics") and hasattr(curr.physics, "data"): 
            return curr.physics.data
        if hasattr(curr, "sim") and hasattr(curr.sim, "data"): 
            return curr.sim.data
        if hasattr(curr, "env"):
            curr = curr.env
        else:
            break
    return None

def visualize():
    parser = argparse.ArgumentParser(description="Visualize Trained Policy")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["sac"])
    parser.add_argument("--task_index", type=int, default=DEFAULT_TASK_INDEX)
    parser.add_argument("--policy_clip", type=float, default=DEFAULT_POLICY_CLIP)
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    MODEL_TYPE = args.model_type
    TASK_INDEX = args.task_index
    POLICY_CLIP = args.policy_clip
    RENDER_MODE = "human"

    # 1. Initialize Environment
    env_name = ENV_IDS[TASK_INDEX]
    print(f"Loading environment: {env_name}")
    
    try:
        import sai_mujoco
    except ImportError:
        pass

    try:
        from sai_rl import SAIClient
        sai = SAIClient(
            comp_id="lower-t1-penalty-kick-goalie",
            api_key="sai_ddqEmPy1JIeQoGSI72BcdGUePbVdYtSj" 
        )
    except Exception:
        pass
    
    env = gym.make(env_name, render_mode=RENDER_MODE)
    action_function = get_action_function()
    preprocessor = Preprocessor()

    try:
        # 3. Initialize Model
        dummy_obs, dummy_info = env.reset()
        dummy_info["task_index"] = TASK_INDEX
        
        # [CRITICAL UPDATE] Inject time for initial feature check (Phase Signal)
        dummy_info["episode_time"] = 0.0

        processed_state = preprocessor.modify_state(dummy_obs, dummy_info)
        n_features = processed_state.shape[1]

        print(f"Model Input Features: {n_features} (Includes Phase Signal)")

        if MODEL_TYPE.lower() == "sac":
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"

            model = SAC(n_features=n_features, action_space=env.action_space, device=device)
            print(f"[visualize] SAC device: {model.device}")

        # 4. Load Weights
        print(f"Loading weights from {MODEL_PATH}")
        try:
            map_location = "cpu" if model.device == "cpu" else model.device
            state_dict = torch.load(MODEL_PATH, map_location=map_location)
            model.load_state_dict(state_dict)
            model.eval()
        except FileNotFoundError:
            print(f"Error: Model file {MODEL_PATH} not found!")
            return
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return

        # 5. Run Loop
        print("Starting visualization... Press Ctrl+C to stop.")
        
        mj_data = get_mujoco_data(env)
        if mj_data is None:
            print("CRITICAL WARNING: Could not access MuJoCo data. Rewards will be broken.")
            qpos0 = np.zeros(12) 
        else:
            env.reset()
            full_qpos = mj_data.qpos.copy()
            qpos0 = full_qpos[7:] 
            print(f"Captured Nominal Pose (qpos0): {qpos0[:4]}...")

        num_episodes = 3

        for ep in range(num_episodes):
            obs, info = env.reset()
            info["task_index"] = TASK_INDEX
            
            prev_action = np.zeros(env.action_space.shape)
            prev_base_ang_vel = np.zeros(3)
            
            if mj_data:
                prev_height = mj_data.qpos[2]
            else:
                prev_height = 0.62

            episode_time = 0.0
            phase_offset = 0.0 
            stats_buffer = {}
            done = False
            step = 0

            while not done:
                # [CRITICAL UPDATE] Inject time for Preprocessor to generate Phase Signal
                info["episode_time"] = episode_time
                
                s = preprocessor.modify_state(obs, info).squeeze()

                if MODEL_TYPE.lower() == "sac":
                    raw_action = model.select_action(s, evaluate=True)
                    if POLICY_CLIP is not None and POLICY_CLIP > 0:
                        raw_action = np.clip(raw_action, -POLICY_CLIP, POLICY_CLIP)
                    policy = np.expand_dims(raw_action, axis=0)
                
                action = action_function(policy)[0].squeeze()
                next_obs, env_reward, terminated, truncated, info = env.step(action)
                
                # --- Accurate Reward Calculation ---
                if mj_data:
                    true_height = mj_data.qpos[2]
                else:
                    true_height = 0.62

                calc_reward, _, reason, stats = run_rewards.calculate_reward(
                    state=obs,
                    next_state=next_obs,
                    action=action,
                    prev_action=prev_action,
                    qpos0=qpos0,
                    prev_base_ang_vel=prev_base_ang_vel,
                    episode_time=episode_time,
                    true_height=true_height,
                    prev_height=prev_height,
                    phase_offset=phase_offset
                )

                for k, v in stats.items():
                    if k not in stats_buffer: stats_buffer[k] = []
                    stats_buffer[k].append(v)

                obs = next_obs
                prev_action = action
                prev_height = true_height
                episode_time += C.DT
                step += 1
                
                done = terminated or truncated

            print(f"\nEp {ep+1} | Steps: {step} | Reason: {reason}")
            print("   [Rewards Breakdown]")
            for k, v_list in stats_buffer.items():
                avg_val = sum(v_list) / len(v_list)
                print(f"      {k}: {avg_val:.4f}")
            print("-" * 40)

    finally:
        env.close()

if __name__ == "__main__":
    visualize()