import os
import glob
import wandb
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys

# Allow importing from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import run_config as C       
import run_rewards as R      
from balance_utils import ReplayBuffer 
from sac import SAC
from main import Preprocessor, get_action_function, ENV_IDS, MultiTaskEnv

# --- CONFIGURATION & PATHS ---
USE_DEMONSTRATIONS = False
BC_STEPS = 50000         
BC_BATCH_SIZE = 256
LOAD_CHECKPOINT = False   
CHECKPOINT_FILENAME = C.CHECKPOINT_FILENAME # Pulled from Config

# DEMO FILES
DEMO_DIR = os.path.join(os.path.dirname(__file__), "demonstrations")
DEMO_FILES = [
    os.path.join(DEMO_DIR, "walking_processed.npz"),
]

# --- HELPER: ROBUST PHYSICS FINDER ---
def find_mujoco_objects(env_obj):
    """Recursively hunts for (model, data) objects."""
    if hasattr(env_obj, 'data') and hasattr(env_obj, 'model'):
        return env_obj.model, env_obj.data
    if hasattr(env_obj, 'sim'):
        return env_obj.sim.model, env_obj.sim.data
    if hasattr(env_obj, 'unwrapped') and env_obj.unwrapped != env_obj:
        m, d = find_mujoco_objects(env_obj.unwrapped)
        if d: return m, d
    for attr in ['env', 'envs', 'active_env', '_env', 'task_envs']:
        if hasattr(env_obj, attr):
            child = getattr(env_obj, attr)
            if isinstance(child, dict) and len(child) > 0: child = list(child.values())[0]
            elif isinstance(child, list) and len(child) > 0: child = child[0]
            if child:
                m, d = find_mujoco_objects(child)
                if d: return m, d
    return None, None

def detect_root_body(model, data):
    """Finds the index of the main torso/root body by checking initial Z-heights."""
    print("\n[Physics] Scanning bodies to find Torso/Root...")
    best_id = 1 
    max_z = -999.0
    n_bodies = getattr(model, 'nbody', 0)
    if n_bodies == 0: n_bodies = 10 
        
    for i in range(1, n_bodies):
        try:
            z_val = data.xpos[i, 2]
            if z_val > max_z and z_val < 1.5:
                max_z = z_val
                best_id = i
        except: break
    print(f"[Physics] Locked onto Body {best_id} as Root. Initial Height: {max_z:.4f} m")
    return best_id

def load_demonstrations_to_buffer(buffer, file_paths):
    total_count = 0
    print(f"[Loader] Attempting to load {len(file_paths)} demonstration files...")
    for file_path in file_paths:
        if not os.path.exists(file_path): continue
        try:
            data = np.load(file_path)
            obs, actions, rewards = data['obs'], data['actions'], data['rewards']
            next_obs, dones = data['next_obs'], data['dones']
            
            # Padding for new observation space size if necessary
            # Note: Phase signal adds 2 dims, so 87 -> 89. Adjust logic as needed.
            # For now, we assume demos might not be perfectly compatible with new Phase inputs 
            # and might need retraining or robust loading. 
            
            n_steps = min(len(obs), len(actions), len(rewards), len(next_obs), len(dones))
            for t in range(n_steps):
                buffer.push(obs[t], actions[t], rewards[t], next_obs[t], dones[t])
            total_count += n_steps
        except Exception as e:
            print(f"  [Error] Failed to load {os.path.basename(file_path)}: {e}")
    return total_count

def pretrain_actor_bc(agent, replay_buffer, steps, batch_size):
    if len(replay_buffer) < batch_size: return
    print(f"[Pre-Train] Starting Imitation Learning for {steps} steps...")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    pbar = tqdm(range(steps), desc="Imitation Learning")
    for i in pbar:
        state_batch, action_batch, _, _, _ = replay_buffer.sample(batch_size)
        state_batch = torch.FloatTensor(state_batch).to(agent.device)
        action_batch = torch.FloatTensor(action_batch).to(agent.device)
        outputs = agent.actor(state_batch)
        pred_action = outputs[2] if isinstance(outputs, tuple) and len(outputs) >= 3 else (outputs[0] if isinstance(outputs, tuple) else outputs)
        loss = criterion(pred_action, action_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("[Pre-Train] Behavior Cloning complete.")

# --- MAIN LOOP ---
def train_run():
    env = MultiTaskEnv(ENV_IDS)
    preprocessor = Preprocessor()
    dummy_obs, dummy_info = env.reset()
    
    # Inject dummy time for initial dimension check
    dummy_info["episode_time"] = 0.0
    
    mj_model, mj_data = find_mujoco_objects(env)
    if mj_data is None:
        print("\nFATAL: Could not find MuJoCo data object.")
        return
    
    root_body_id = detect_root_body(mj_model, mj_data)
    
    # Check input dims with the new Phase Signal
    processed_dummy = preprocessor.modify_state(dummy_obs, dummy_info)
    n_features = processed_dummy.shape[1]
    print(f"[Init] Model Input Features: {n_features} (Includes Phase Signal)")
    
    agent = SAC(
        n_features=n_features,
        action_space=env.action_space,
        log_std_min=-4.0,
        log_std_max=-0.5,
        alpha=0.05, 
        device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    model_dir = os.path.join(os.path.dirname(__file__), "run_models2")
    os.makedirs(model_dir, exist_ok=True)
    
    resume_path = os.path.join(model_dir, CHECKPOINT_FILENAME)
    
    loaded_checkpoint = False
    if LOAD_CHECKPOINT and os.path.exists(resume_path):
        # We need to handle potential size mismatch if loading old checkpoint without Phase
        try:
            agent.load_state_dict(torch.load(resume_path, map_location=agent.device))
            print(f"[Init] Loaded Checkpoint from {CHECKPOINT_FILENAME}")
            loaded_checkpoint = True
        except Exception as e:
            print(f"[Init] Checkpoint load failed (likely architecture mismatch due to Phase Signal): {e}")
            print("[Init] Starting Fresh with new architecture.")
            loaded_checkpoint = False
    else:
        print("[Init] Starting Fresh.")

    replay_buffer = ReplayBuffer(C.REPLAY_SIZE)
    action_function = get_action_function()

    if USE_DEMONSTRATIONS and not loaded_checkpoint:
        cnt = load_demonstrations_to_buffer(replay_buffer, DEMO_FILES)
        if cnt > 0:
            pretrain_actor_bc(agent, replay_buffer, BC_STEPS, BC_BATCH_SIZE)
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_run_seeded.pth"))

    if C.USE_WANDB:
        wandb.init(project="booster-run", name="run-v8-stable", config=vars(C))

    total_steps = 0
    episode_num = 0
    best_reward = -float("inf")
    recent_rewards = deque(maxlen=C.LOG_WINDOW_EPISODES)
    pbar = tqdm(total=C.TIMESTEPS, desc="RL Training")

    while total_steps < C.TIMESTEPS:
        obs, info = env.reset()
        
        # [CRITICAL UPDATE] Inject time for Preprocessor Phase Signal
        info["episode_time"] = 0.0 
        
        state = preprocessor.modify_state(obs, info).squeeze()
        
        # --- INITIALIZE TRACKERS ---
        episode_reward, episode_steps, done = 0.0, 0, False
        stats_agg = defaultdict(float)
        episode_reason = "Time Limit"
        
        qpos0 = state[:12].copy()
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        prev_base_ang_vel = state[27:30].copy()
        phase_offset = np.random.uniform(0.0, 1.0)

        try: current_height = mj_data.xpos[root_body_id, 2]
        except: current_height = 0.62
        prev_height = current_height 

        while not done:
            should_warmup = (not loaded_checkpoint) and (total_steps < C.WARMUP_STEPS) and (len(replay_buffer) == 0)
            action = env.action_space.sample() if should_warmup else agent.select_action(state, evaluate=False)
            action_clamped = np.clip(action, -C.POLICY_ACTION_CLIP, C.POLICY_ACTION_CLIP)
            
            next_obs, _, _, truncated, info = env.step(action_function(action_clamped))
            
            # [CRITICAL UPDATE] Pass time to Preprocessor for next_state
            episode_steps += 1
            info["episode_time"] = episode_steps * C.DT
            
            next_state = preprocessor.modify_state(next_obs, info).squeeze()

            try: current_height = mj_data.xpos[root_body_id, 2]
            except: current_height = 0.62

            reward, terminated, reason, stats = R.calculate_reward(
                state, next_state, action_clamped, prev_action, qpos0, prev_base_ang_vel, 
                episode_steps * C.DT, current_height, prev_height, phase_offset
            )

            if episode_steps >= C.MAX_EPISODE_STEPS: truncated = True
            done = terminated or truncated
            if terminated: episode_reason = reason
            episode_reward += reward
            
            if C.DEBUG_REWARDS:
                for k, v in stats.items(): stats_agg[k] += v

            replay_buffer.push(state, action, reward, next_state, done)
            
            if len(replay_buffer) > C.BATCH_SIZE and total_steps % C.UPDATE_INTERVAL == 0:
                for _ in range(C.UPDATES_PER_INTERVAL):
                    agent.update(*replay_buffer.sample(C.BATCH_SIZE))

            state = next_state
            prev_action = action_clamped
            prev_height = current_height
            total_steps += 1
            pbar.update(1)
            if done: break

        episode_num += 1
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_run_best.pth"))

        if episode_num % C.LOG_EVERY_EPISODES == 0:
            print(f"\nEp {episode_num} | Reward: {episode_reward:.2f} | Avg: {avg_reward:.2f} | Steps: {episode_steps} | Reason: {episode_reason}")
            if C.DEBUG_REWARDS and episode_steps > 0:
                print("   [Rewards Breakdown]")
                for k, v in stats_agg.items():
                    avg_val = v / episode_steps
                    if abs(avg_val) > 0.001:
                        print(f"      {k.replace('r_', '')}: {avg_val:.4f}")
            print("-" * 40)
            
            if C.USE_WANDB:
                wandb.log({"episode/reward": episode_reward, "episode/avg": avg_reward}, step=total_steps)

        if episode_num % C.SAVE_EVERY_EPISODES == 0:
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_run_checkpoint.pth"))
        if episode_num % C.SAVE_EVERY_EPISODES_LONGER == 0:
            longer_filename = f"sac_run_checkpoint{episode_num:05d}.pth"
            torch.save(agent.state_dict(), os.path.join(model_dir, longer_filename))

    env.close()

if __name__ == "__main__":
    train_run()