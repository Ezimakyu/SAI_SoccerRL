
# Visualize stand: python visualize.py
# Visualize balance: python visualize.py --model_type sac --model_path training_scripts/balance_models/sac_balance_checkpoint.pth --policy_clip 0.35


import torch
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import time
import sys
import os

# Add training_scripts to python path so we can import modules from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'training_scripts'))

from sai_rl import SAIClient
# from training_scripts.td3 import TD3
from sac import SAC
from training_scripts.main import Preprocessor, get_action_function, ENV_IDS

import argparse

# Configuration
# Defaults
DEFAULT_MODEL_PATH = "training_scripts/stand_models/sac_balance_checkpoint.pth"
DEFAULT_MODEL_TYPE = "sac" 
DEFAULT_TASK_INDEX = 0
DEFAULT_POLICY_CLIP = 0.35

def visualize():
    parser = argparse.ArgumentParser(description="Visualize Trained Policy")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE, choices=["sac"], help="Model type: sac")
    parser.add_argument("--task_index", type=int, default=DEFAULT_TASK_INDEX, help="Task index (0-2)")
    parser.add_argument(
        "--policy_clip",
        type=float,
        default=DEFAULT_POLICY_CLIP,
        help="Clamp SAC policy output in [-policy_clip, +policy_clip] before scaling. "
             "This should match training (e.g. POLICY_ACTION_CLIP in train_stand.py).",
    )
    args = parser.parse_args()

    MODEL_PATH = args.model_path
    MODEL_TYPE = args.model_type
    TASK_INDEX = args.task_index
    POLICY_CLIP = args.policy_clip
    RENDER_MODE = "human"

    # 1. Initialize the Environment
    # We load a specific environment to watch, e.g., the Goalie task
    env_name = ENV_IDS[TASK_INDEX]
    print(f"Loading environment: {env_name}")
    
    # Try to import sai_mujoco to register envs locally if SAIClient fails
    try:
        import sai_mujoco
        print("[visualize] Imported sai_mujoco for local environment registration.")
    except ImportError:
        pass

    # Initialize SAIClient (Optional for local visualization if envs are registered via import)
    try:
        from sai_rl import SAIClient
        sai = SAIClient(
            comp_id="lower-t1-penalty-kick-goalie",
            api_key="sai_ddqEmPy1JIeQoGSI72BcdGUePbVdYtSj" 
        )
    except Exception as e:
        print(f"[visualize] Warning: SAIClient init failed (offline mode?): {e}")
    
    # Use sai.make_env for convenience as it handles render modes and registration well
    # Or use gym.make(env_name, render_mode=RENDER_MODE) if registered
    env = gym.make(env_name, render_mode=RENDER_MODE)
    action_function = get_action_function(env.action_space)

    # 2. Initialize Preprocessor
    preprocessor = Preprocessor()

    try:
        # 3. Initialize Model (Architecture must match training)
        # Get feature dimension dynamically to be safe
        dummy_obs, dummy_info = env.reset()
        # Mock task_index in info if missing, Preprocessor handles it but let's be explicit
        dummy_info["task_index"] = TASK_INDEX

        processed_state = preprocessor.modify_state(dummy_obs, dummy_info)
        n_features = processed_state.shape[1]

        print(f"Model Input Features: {n_features}")

        if MODEL_TYPE.lower() == "sac":
            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"

            model = SAC(
                n_features=n_features,
                action_space=env.action_space,
                device=device,
            )
            print(f"[visualize] SAC device: {model.device}")
        # elif MODEL_TYPE.lower() == "td3":
        #     model = TD3(
        #         n_features=n_features,
        #         action_space=env.action_space,
        #         neurons=[400, 300],
        #         activation_function=F.relu,
        #         learning_rate=0.0001,
        #     )
        else:
            raise ValueError(f"Unknown MODEL_TYPE={MODEL_TYPE!r}. Use 'sac'.")

        # 4. Load Weights
        print(f"Loading weights from {MODEL_PATH}")
        try:
            map_location = "cpu"
            if MODEL_TYPE.lower() == "sac":
                map_location = model.device  # type: ignore[attr-defined]
            state_dict = torch.load(MODEL_PATH, map_location=map_location)
            model.load_state_dict(state_dict)
            model.eval()  # type: ignore[attr-defined]
        except FileNotFoundError:
            print(f"Error: Model file {MODEL_PATH} not found!")
            return
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return

        # 5. Run Loop
        print("Starting visualization... Press Ctrl+C to stop.")
        num_episodes = 5

        for ep in range(num_episodes):
            obs, info = env.reset()
            info["task_index"] = TASK_INDEX  # Ensure preprocessor knows the task

            done = False
            episode_reward = 0
            step = 0

            while not done:
                # Preprocess
                s = preprocessor.modify_state(obs, info).squeeze()

                # Inference
                if MODEL_TYPE.lower() == "sac":
                    raw_action = model.select_action(s, evaluate=True)  # type: ignore[attr-defined]
                    # IMPORTANT: match training-time policy-space clamp (prevents huge torques / twitching)
                    if POLICY_CLIP is not None and POLICY_CLIP > 0:
                        raw_action = np.clip(raw_action, -POLICY_CLIP, POLICY_CLIP)
                    policy = np.expand_dims(raw_action, axis=0)
                else:
                    state_tensor = torch.from_numpy(np.expand_dims(s, axis=0))
                    with torch.no_grad():
                        policy = model(state_tensor).detach().numpy()  # type: ignore[operator]

                # Action Scaling (No noise for visualization)
                action = action_function(policy)[0].squeeze()

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                info["task_index"] = TASK_INDEX

                done = terminated or truncated
                episode_reward += reward
                step += 1

                # Optional: slow down if too fast
                # time.sleep(0.01)

            print(f"Episode {ep+1} Reward: {episode_reward:.2f}, Steps: {step}")
    finally:
        env.close()

if __name__ == "__main__":
    visualize()