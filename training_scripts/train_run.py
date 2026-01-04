# train_run.py
import os
import wandb
from collections import deque, defaultdict
import numpy as np
import torch
from tqdm import tqdm
import sys

# Allow importing from current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- CHANGED IMPORTS ---
import run_config as C       # Points to run_config.py
import run_rewards as R      # Points to run_rewards.py
from balance_utils import ReplayBuffer # Re-use the utility class (no need to rename)
from sac import SAC
from main import Preprocessor, get_action_function, ENV_IDS, MultiTaskEnv

# --- OPTIONAL: SAI IMPORTS ---
try:
    import sai_mujoco
    print("[train_run] Imported sai_mujoco for local environment registration.")
except ImportError:
    pass

try:
    from sai_rl import SAIClient
except Exception as e:
    print(f"[train_run] Warning: SAIClient init failed: {e}")


def train_run():
    # 1. SETUP ENVIRONMENT
    env = MultiTaskEnv(ENV_IDS)
    preprocessor = Preprocessor()
    
    dummy_obs, dummy_info = env.reset()
    n_features = preprocessor.modify_state(dummy_obs, dummy_info).shape[1]

    agent = SAC(
        n_features=n_features,
        action_space=env.action_space,
        log_std_min=-4.0,
        log_std_max=-0.5,
        alpha=0.03,
        alpha_decay=0.9999,
        alpha_min=0.01,
        device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    
    print(f"[train_run] SAC device: {agent.device}")
    
    # Load PREVIOUS balance checkpoint if you have a good one (Optional but recommended)
    resume_ckpt_path = os.path.join(os.path.dirname(__file__), "run_models", "sac_run_statue.pth")
    if os.path.exists(resume_ckpt_path):
        agent.load_state_dict(torch.load(resume_ckpt_path, map_location=agent.device))
        
        # 2. FORCE ENTROPY RESET (The "Re-Explore" Fix)
        # Your SAC implementation stores alpha as a float.
        # We reset it to a higher value (e.g., 0.1 or 0.2) to make the policy
        # "curious" again, so it will try leaning forward to find the running reward.
        target_reset_alpha = 0.2
        
        # Overwrite the internal variable
        agent.alpha = target_reset_alpha
        
        # CRITICAL: Reset the decay step counter so it doesn't instantly decay back down
        if hasattr(agent, '_update_steps'):
            agent._update_steps = 0
            
        print(f"[train_run] *** ENTROPY RESET: Alpha forced to {agent.alpha} for re-exploration ***")
    # else:
    # print("[train_run] STARTING FRESH (No checkpoint loaded).")

    replay_buffer = ReplayBuffer(C.REPLAY_SIZE)
    action_function = get_action_function(env.action_space)

    # 2. LOGGING SETUP
    if C.USE_WANDB:
        wandb.init(
            project="booster-run",  # Changed project name
            name="run-v1",
            config={k: v for k, v in vars(C).items() if not k.startswith('__')}
        )

    # Changed save directory to 'run_models'
    model_dir = os.path.join(os.path.dirname(__file__), "run_models")
    os.makedirs(model_dir, exist_ok=True)

    # 3. TRAINING LOOP
    total_steps = 0
    episode_num = 0
    best_episode_reward = -float("inf")
    recent_episode_rewards = deque(maxlen=C.LOG_WINDOW_EPISODES)
    last_logged_episode = -1
    
    pbar = tqdm(total=C.TIMESTEPS)

    while total_steps < C.TIMESTEPS:
        obs, info = env.reset()
        state = preprocessor.modify_state(obs, info).squeeze()
        
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        episode_reward_stats = defaultdict(float)
        
        qpos0 = state[:12].copy()
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)
        prev_base_ang_vel = state[27:30].copy()

        while not done and total_steps < C.TIMESTEPS:
            # A. ACTION SELECTION
            if total_steps < C.WARMUP_STEPS:
                with torch.no_grad():
                    action = agent.select_action(state, evaluate=False)
                action = action + np.random.normal(0.0, 0.1, size=action.shape)
            else:
                action = agent.select_action(state)

            if np.random.rand() < C.PERTURB_PROB:
                action = action + np.random.normal(0.0, C.PERTURB_POLICY_STD, size=action.shape)

            # B. DYNAMIC CLIPPING (URGENCY)
            cur_gravity = state[24:27]
            cur_tilt = float(np.linalg.norm(cur_gravity[:2]))
            cur_ang = state[27:30]
            
            ang_acc_est = float(np.linalg.norm(cur_ang - prev_base_ang_vel)) / C.DT
            
            fall_urgency_pre = float(np.clip(
                np.linalg.norm(cur_ang) * C.URGENCY_W_ANG + 
                C.URGENCY_W_TILT * cur_tilt + 
                C.URGENCY_W_ANG_ACC * ang_acc_est, 
                0.0, C.URGENCY_MAX
            ))
            
            dynamic_clip = float(C.POLICY_ACTION_CLIP + C.DYNAMIC_CLIP_W * fall_urgency_pre)
            dynamic_clip = min(dynamic_clip, 1.0)

            action = np.clip(action, -dynamic_clip, dynamic_clip)
            scaled_action = action_function(action)
            
            # C. EXTERNAL DISTURBANCE
            if np.random.rand() < C.PERTURB_PROB:
                torque_std = C.PERTURB_TORQUE_STD_FRAC * (env.action_space.high - env.action_space.low)
                scaled_action = scaled_action + np.random.normal(0.0, torque_std, size=scaled_action.shape)
                scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)

            # D. ENV STEP
            next_obs, _, _, truncated, info = env.step(scaled_action)
            next_state = preprocessor.modify_state(next_obs, info).squeeze()

            # E. REWARD CALCULATION (Uses run_rewards.py)
            reward, terminated, reason, stats = R.calculate_reward(
                state, next_state, action, prev_action, qpos0
            )

            episode_steps += 1
            if episode_steps >= C.MAX_EPISODE_STEPS:
                truncated = True

            done = bool(terminated or truncated)
            episode_reward += float(reward)

            if C.DEBUG_REWARDS:
                for k, v in stats.items():
                    episode_reward_stats[k] += v

            replay_buffer.push(state, action, reward, next_state, done)
            
            prev_action = action
            prev_base_ang_vel = cur_ang.copy()
            state = next_state
            total_steps += 1
            pbar.update(1)

            # Gradient Step
            q_loss = 0.0
            pi_loss = 0.0
            if total_steps > C.WARMUP_STEPS and len(replay_buffer) > C.BATCH_SIZE:
                if total_steps % C.UPDATE_INTERVAL == 0:
                    for _ in range(C.UPDATES_PER_INTERVAL):
                        q_loss, pi_loss = agent.update(*replay_buffer.sample(C.BATCH_SIZE))
            
            if C.USE_WANDB and total_steps > C.WARMUP_STEPS and total_steps % C.UPDATE_INTERVAL == 0:
                wandb.log({
                   "train/q_loss": q_loss,
                   "train/pi_loss": pi_loss,
                }, step=total_steps)

        # --- EPISODE END LOGIC ---
        episode_num += 1
        
        # 1. Debug Print
        if C.DEBUG_REWARDS and (episode_num % C.DEBUG_PRINT_INTERVAL == 0):
            print(f"\n[DEBUG] Episode {episode_num} Reward Breakdown (Total: {episode_reward:.2f}):")
            sorted_stats = sorted(episode_reward_stats.items(), key=lambda x: abs(x[1]), reverse=True)
            for k, v in sorted_stats:
                sign = "+" if v >= 0 else ""
                print(f"  {k:<15} : {sign}{v:.2f}")
            print("-" * 40)
            
            if C.USE_WANDB:
                 debug_dict = {f"rewards/{k}": v for k, v in episode_reward_stats.items()}
                 wandb.log(debug_dict, step=total_steps)

        # 2. Stats Tracking
        recent_episode_rewards.append(episode_reward)
        
        if len(recent_episode_rewards) > 0:
            r_min = float(np.min(recent_episode_rewards))
            r_median = float(np.median(recent_episode_rewards))
            r_avg = float(np.mean(recent_episode_rewards))
            r_max = float(np.max(recent_episode_rewards))
        else:
            r_min = r_median = r_avg = r_max = episode_reward

        # 3. Save Best
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_run_best.pth"))

        # 4. Periodic Log
        if episode_num > 0 and (episode_num % C.LOG_EVERY_EPISODES == 0) and (episode_num != last_logged_episode):
            tqdm.write(
                f"[run] Ep {episode_num} | "
                f"Min: {r_min:6.2f} | "
                f"Med: {r_median:6.2f} | "
                f"Avg: {r_avg:6.2f} | "
                f"Max: {r_max:6.2f} | "
                f"Steps: {episode_steps}"
            )
            
            if C.USE_WANDB:
                wandb.log({
                    "episode/reward_min": r_min,
                    "episode/reward_median": r_median,
                    "episode/reward_avg": r_avg,
                    "episode/reward_max": r_max,
                    "episode/length": episode_steps,
                }, step=total_steps)
            last_logged_episode = episode_num

        if episode_num % 50 == 0:
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_run_checkpoint.pth"))

    torch.save(agent.state_dict(), os.path.join(model_dir, "sac_run_final.pth"))
    print("[train_run] Training complete.")
    if C.USE_WANDB:
        wandb.finish()
    env.close()


if __name__ == "__main__":
    train_run()