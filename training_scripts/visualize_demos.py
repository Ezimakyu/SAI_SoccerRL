# python visualize_demos.py --file demonstrations/goal_kick.npz --loop

import numpy as np
import gymnasium as gym
import argparse
import time
import os
import sys
import glob

# Add training_scripts to path if needed (adjust based on your folder structure)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import get_action_function, ENV_IDS
# Optional: Try to import sai_mujoco for local env registration
try:
    import sai_mujoco
    print("[visualize_demos] Imported sai_mujoco.")
except ImportError:
    pass

# --- CONFIGURATION ---
DEFAULT_DEMO_DIR = os.path.join(os.path.dirname(__file__), "demonstrations")
DEFAULT_TASK_INDEX = 0

def visualize_demonstrations():
    parser = argparse.ArgumentParser(description="Replay .npz Demonstrations")
    parser.add_argument("--demo_dir", type=str, default=DEFAULT_DEMO_DIR, help="Folder containing .npz files")
    parser.add_argument("--file", type=str, default=None, help="Specific .npz file to play (optional)")
    parser.add_argument("--task_index", type=int, default=DEFAULT_TASK_INDEX, help="Task index (0-2) to initialize env")
    parser.add_argument("--fps", type=int, default=30, help="Playback speed (FPS)")
    parser.add_argument("--loop", action="store_true", help="Loop the specific file")
    args = parser.parse_args()

    # 1. SETUP ENVIRONMENT
    env_name = ENV_IDS[args.task_index]
    print(f"Loading Environment: {env_name}")
    
    # We use render_mode="human" to see the window
    try:
        env = gym.make(env_name, render_mode="human")
    except Exception as e:
        print(f"Error loading env: {e}")
        print("Make sure sai_mujoco is installed or env is registered.")
        return

    action_function = get_action_function(env.action_space)

    # 2. LOCATE FILES
    if args.file:
        files = [args.file]
    else:
        search_path = os.path.join(args.demo_dir, "*.npz")
        files = glob.glob(search_path)
        print(f"Found {len(files)} .npz files in {args.demo_dir}")

    if not files:
        print("No files found. Exiting.")
        return

    # 3. PLAYBACK LOOP
    print("Starting Playback... Press Ctrl+C to exit.")
    print("------------------------------------------")
    
    try:
        for file_path in files:
            # Loop check for single file mode
            while True: 
                print(f"Playing: {os.path.basename(file_path)}")
                
                try:
                    data = np.load(file_path)
                    
                    # Handle different naming conventions
                    if 'actions' in data:
                        actions = data['actions']
                    elif 'action' in data:
                        actions = data['action']
                    else:
                        print(f"Skipping {file_path}: No 'actions' key found.")
                        break
                    
                    # Reset Env
                    env.reset()
                    
                    # Play the tape
                    total_reward = 0
                    for t, raw_action in enumerate(actions):
                        # Handle shape issues (sometimes actions are [1, dim])
                        if len(raw_action.shape) > 1:
                            raw_action = raw_action.squeeze()
                        
                        # Apply Action Scaling (Important!)
                        # We assume the dataset stores 'policy' actions (unscaled).
                        # If the robot spazzes out, try removing action_function().
                        scaled_action = action_function(raw_action)
                        
                        _, reward, terminated, truncated, _ = env.step(scaled_action)
                        total_reward += reward
                        
                        # Render (Gym handles window updates)
                        env.render()
                        
                        # Speed Control
                        time.sleep(1.0 / args.fps)

                        if terminated or truncated:
                            break

                    print(f"  -> Replay Ended. Steps: {t+1} | Re-Simulated Reward: {total_reward:.2f}")

                except Exception as e:
                    print(f"Error playing {file_path}: {e}")

                # Break the while loop unless we are forced to loop
                if not args.loop:
                    break
                else:
                    print("Looping...")
                    time.sleep(0.5)

            # Pause between files
            if not args.loop:
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    visualize_demonstrations()