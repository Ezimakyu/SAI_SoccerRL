import time
import numpy as np
import torch
import glob
import os
import sys
import wandb
from collections import deque, defaultdict
from tqdm import tqdm

# --- IMPORTS FROM LOCAL MODULES ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sac import SAC 
from main import MultiTaskEnv, ENV_IDS, Preprocessor, get_action_function
from balance_utils import ReplayBuffer

# --- CONFIGS ---
import imitate_config as C
import imitate_rewards as R

# --- HELPERS ---
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

class MotionLoader:
    def __init__(self, path):
        print(f"[Loader] Loading trajectory: {os.path.basename(path)}")
        self.data = np.load(path)
        
        if "qpos" in self.data:
            self.qpos_traj = self.data["qpos"]
        else:
            if "obs" in self.data: 
                self.qpos_traj = self.data["obs"] 
            else:
                raise ValueError(f"File {path} does not contain 'qpos' key!")

        self.qvel_traj = self.data.get("qvel", None)
        self.frequency = float(self.data["frequency"]) if "frequency" in self.data else 50.0
        self.dt = 1.0 / self.frequency
        self.num_frames = self.qpos_traj.shape[0]
        self.duration = self.num_frames * self.dt

    def get_frame(self, episode_time):
        idx = int((episode_time % self.duration) / self.dt)
        idx = min(idx, self.num_frames - 1)
        qpos = self.qpos_traj[idx].copy()
        if len(qpos) >= 7:
            norm = np.linalg.norm(qpos[3:7])
            if norm > 1e-6: qpos[3:7] /= norm
        return qpos

    def get_frame_vel(self, episode_time):
        if self.qvel_traj is None: return None
        idx = int((episode_time % self.duration) / self.dt)
        idx = min(idx, self.num_frames - 1)
        return self.qvel_traj[idx].copy()

# --- MAIN LOOP ---
def main():
    # 1. SETUP ENVIRONMENT
    env = MultiTaskEnv(ENV_IDS)
    
    mj_model, mj_data = find_mujoco_objects(env)
    if mj_data is None:
        print("\nFATAL: Could not find MuJoCo data object.")
        return

    # Initialize Preprocessor
    dummy_obs, dummy_info = env.reset()
    dummy_info["episode_time"] = 0.0
    preprocessor = Preprocessor()
    processed_state = preprocessor.modify_state(dummy_obs, dummy_info)
    n_features = processed_state.shape[1]
    
    action_function = get_action_function()
    
    # 2. SETUP AGENT
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    agent = SAC(
        n_features=n_features,
        action_space=env.action_space,
        log_std_min=-4.0,
        log_std_max=-0.5,
        neurons=[400, 300],
        alpha=0.05,
        device=device,
    )
    
    replay_buffer = ReplayBuffer(C.REPLAY_SIZE)

    # 3. SETUP DIRS & LOGGING
    model_dir = os.path.join(os.path.dirname(__file__), "models_imitate")
    os.makedirs(model_dir, exist_ok=True)

    if C.USE_WANDB:
        wandb.init(project="booster-imitate", name="imitate-v1", config=vars(C))

    # 4. SETUP TRAJECTORIES
    if not C.TRAJECTORY_FILES:
        search_path = os.path.join(C.TRAJECTORY_DIR, "*.npz")
        traj_files = glob.glob(search_path)
    else:
        traj_files = [os.path.join(C.TRAJECTORY_DIR, f) for f in C.TRAJECTORY_FILES]
    
    if not traj_files:
        print(f"ERROR: No .npz files found in {C.TRAJECTORY_DIR}")
        return

    traj_files.sort()
    current_traj_idx = 0
    motion_loader = MotionLoader(traj_files[0])
    steps_on_current_traj = 0

    print(f"Starting Imitation Training on {device}...")
    print(f"Features: {n_features} | Actions: {env.action_space.shape[0]}")

    # 5. TRAINING LOOP
    total_steps = 0
    episode_num = 0
    pbar = tqdm(total=C.TIMESTEPS, desc="Imitation Training")

    while total_steps < C.TIMESTEPS:
        # Reset Env
        raw_obs, info = env.reset()
        
        # --- [NEW] INITIALIZATION LOGIC ---
        # 1. Determine Start Time
        if C.RANDOM_START_PHASE:
            # Start anywhere in the first 90% of motion
            start_time = np.random.uniform(0, motion_loader.duration * 0.9)
        else:
            start_time = 0.0
            
        current_time = start_time

        # 2. Force Robot to Spawn in Reference Pose
        if C.INIT_FROM_TRAJECTORY:
            try:
                init_qpos = motion_loader.get_frame(start_time)
                init_qvel = motion_loader.get_frame_vel(start_time)
                
                # Copy ref pose into MuJoCo data
                # Handle dimension mismatch safely
                min_q = min(len(mj_data.qpos), len(init_qpos))
                mj_data.qpos[:min_q] = init_qpos[:min_q]
                
                if init_qvel is not None:
                    min_v = min(len(mj_data.qvel), len(init_qvel))
                    mj_data.qvel[:min_v] = init_qvel[:min_v]
                else:
                    mj_data.qvel[:] = 0.0
                
                # [CRITICAL] Forward dynamics to update sensors/collision
                # If we don't do this, the first observation will be stale (T-Pose)
                # find_mujoco_objects returns (model, data). 
                # We typically need the simulator or function to step. 
                # Usually env.sim.forward() or mujoco.mj_forward(model, data).
                # Since we extracted raw data, we can try accessing the function via env.
                
                # Try standard gym mujoco forward
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'model'):
                     import mujoco
                     mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
                elif hasattr(env, 'envs'):
                     # MultiTaskEnv case
                     active_env = env.envs[env.current_env_idx]
                     if hasattr(active_env.unwrapped, 'model'):
                         import mujoco
                         mujoco.mj_forward(active_env.unwrapped.model, active_env.unwrapped.data)

            except Exception as e:
                print(f"Init Warning: {e}")

        # --- END INITIALIZATION ---

        # Update Info for Preprocessor
        info["episode_time"] = current_time
        phase_offset = np.random.uniform(0.0, 1.0)
        
        # Get State (Now reflecting the forced pose)
        state = preprocessor.modify_state(raw_obs, info).squeeze()
        
        episode_reward = 0
        done = False
        episode_steps = 0
        episode_reason = "Time Limit"
        stats_agg = defaultdict(float)
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

        while not done:
            # A. Select Action
            should_warmup = (total_steps < C.WARMUP_STEPS) and (len(replay_buffer) == 0)
            
            if should_warmup:
                raw_action = env.action_space.sample()
                action_clamped = np.clip(raw_action, -1.0, 1.0)
            else:
                action_clamped = agent.select_action(state, evaluate=False)
            
            scaled_action = action_function(action_clamped)

            # B. Step
            next_raw_obs, _, _, truncated, info = env.step(scaled_action)
            
            episode_steps += 1
            info["episode_time"] = current_time + C.DT # Use tracked time
            next_state = preprocessor.modify_state(next_raw_obs, info).squeeze()

            # C. Extract Physics
            try:
                raw_qpos = mj_data.qpos.flat[:].copy()
                raw_qvel = mj_data.qvel.flat[:].copy()
            except Exception as e:
                raw_qpos = np.zeros(12 + 7)
                raw_qvel = np.zeros(12 + 6)

            # Get Targets
            target_qpos = motion_loader.get_frame(current_time)
            target_qvel = motion_loader.get_frame_vel(current_time)

            reward_info = {
                "current_qpos": raw_qpos,
                "current_qvel": raw_qvel,
                "target_qpos": target_qpos,
                "target_qvel": target_qvel,
                "phase_offset": phase_offset
            }
            reward_info.update(info)

            # D. Reward
            reward, terminated, reason, stats = R.calculate_reward(
                state, next_state, action_clamped, 
                prev_action, raw_qpos, None, 
                current_time, 0.0, 0.0, 
                motion_loader=motion_loader, 
                info=reward_info
            )
            
            if episode_steps >= C.MAX_EPISODE_STEPS: 
                truncated = True
            
            done = terminated or truncated
            if terminated: episode_reason = reason

            # E. Store
            replay_buffer.push(state, action_clamped, reward, next_state, done)

            if len(replay_buffer) > C.BATCH_SIZE and total_steps % C.UPDATE_INTERVAL == 0:
                for _ in range(C.UPDATES_PER_INTERVAL):
                    agent.update(*replay_buffer.sample(C.BATCH_SIZE))

            state = next_state
            prev_action = action_clamped
            episode_reward += reward
            total_steps += 1
            steps_on_current_traj += 1
            current_time += C.DT
            
            if C.DEBUG_REWARDS:
                for k, v in stats.items(): stats_agg[k] += v

            pbar.update(1)

            # F. Curriculum Switch
            if steps_on_current_traj >= C.STEPS_PER_TRAJECTORY:
                current_traj_idx = (current_traj_idx + 1) % len(traj_files)
                new_file = traj_files[current_traj_idx]
                motion_loader = MotionLoader(new_file)
                steps_on_current_traj = 0

            if done: break

        # Log
        episode_num += 1
        if episode_num % C.LOG_EVERY_EPISODES == 0:
            print(f"\nEp {episode_num} | Reward: {episode_reward:.2f} | Steps: {episode_steps} | Reason: {episode_reason}")
            if C.DEBUG_REWARDS and episode_steps > 0:
                print("   [Rewards Breakdown]")
                for k, v in stats_agg.items():
                    avg_val = v / episode_steps
                    if abs(avg_val) > 0.001:
                        print(f"      {k}: {avg_val:.4f}")
            
            if C.USE_WANDB:
                wandb.log({"episode/reward": episode_reward, "episode/steps": episode_steps}, step=total_steps)

        # Save
        if episode_num % C.SAVE_EVERY_EPISODES == 0:
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_imitate_checkpoint.pth"))
        if episode_num % C.SAVE_EVERY_EPISODES_LONGER == 0:
            longer_filename = f"sac_imitate_checkpoint{episode_num:05d}.pth"
            torch.save(agent.state_dict(), os.path.join(model_dir, longer_filename))

    env.close()

if __name__ == "__main__":
    main()