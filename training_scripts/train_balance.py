import os
import random
import wandb
from collections import deque

import numpy as np
import torch
from tqdm import tqdm

import sys

# Allow importing from current directory when running as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sac import SAC
from main import Preprocessor, get_action_function, ENV_IDS, MultiTaskEnv

# Try to import sai_mujoco to register envs locally if SAIClient fails
try:
    import sai_mujoco
    print("[train_balance] Imported sai_mujoco for local environment registration.")
except ImportError:
    pass

# Initialize SAIClient (Optional for local training if envs are registered via import)
try:
    from sai_rl import SAIClient
    sai = SAIClient(
        comp_id="lower-t1-penalty-kick-goalie",
        api_key="sai_ddqEmPy1JIeQoGSI72BcdGUePbVdYtSj"
    )
except Exception as e:
    print(f"[train_balance] Warning: SAIClient init failed (offline mode?): {e}")

# --- STAGE 1.5: BALANCE RECOVERY ---
# Goal: start from standing checkpoint and learn to recover from perturbations without falling.

# Training config
TIMESTEPS = 400_000
BATCH_SIZE = 256
REPLAY_SIZE = 1_000_000
MAX_EPISODE_STEPS = 300  # shorter episodes encourage frequent falls + recovery learning
WARMUP_STEPS = 10_000
UPDATE_INTERVAL = 50       # Run gradient updates every N steps
UPDATES_PER_INTERVAL = 50  # Number of gradient updates to run per interval

# Policy-space action handling (should be consistent with visualization)
POLICY_ACTION_CLIP = 0.35
# Dynamic action clip unlocks larger leg/hip motion during fast falls
DYNAMIC_CLIP_W = 0.80  # Increased from 0.50 to unlock more range during urgency

# Perturbations
# IMPORTANT: make disturbances external (applied after scaling) so the policy learns to counteract them,
# not to output noisy actions itself.
PERTURB_PROB = 0.02
PERTURB_POLICY_STD = 0.05  # small exploration-like jitter in policy space
PERTURB_TORQUE_STD_FRAC = 0.15  # fraction of actuator range (applied in env action space)

# Tilt-gated shaping (stable vs recovery regimes)
STABLE_TILT = 0.15
RECOVERY_TILT = 0.35
UPRIGHT_BOOST_WHEN_TILTED = 1.5

# Fall urgency (unlock motion when falling fast)
URGENCY_MAX = 3.0
URGENCY_W_TILT = 1.0  # Increased from 0.5 to react to tilt sooner
URGENCY_W_ANG = 1.0
URGENCY_W_ANG_ACC = 0.5  # New: predictive urgency from angular acceleration

# Reward weights / thresholds
SURVIVAL_W = 0.5   # Increased from 0.15 to prioritize not falling
UPRIGHT_W = 0.5

# Reward "progress" during recovery: explicitly reward reducing tilt / reducing fall rate
TILT_PROGRESS_W = 1.0
ANGVEL_DAMP_W = 0.2

# Early reaction: reward motion if tilt is small but angular velocity is high (breaks "freeze")
EARLY_REACT_W = 0.1  # Increased from 0.05
EARLY_REACT_ANG_THRESH = 0.2

# CoM Acceleration: reward reversing direction when tilted
COM_ACCEL_W = 0.1

# Bounded joint-velocity penalty: small motion OK, excessive motion punished
VEL_THRESHOLD = 3.0
VEL_W = 0.01       # Decreased from 0.05 to allow more recovery motion

# Pose regularization (prevents the policy from "choosing" a weird crouched / folded posture as a default)
# We gate this by tilt: strong when stable, weak when recovering.
POSE_W = 0.2
POSE_RECOVERY_SCALE = 0.05  # make pose constraints much looser during recovery

# Drift suppression (IMU velocimeter)
DRIFT_W = 0.2

# Angular acceleration smoothness (encourage corrective but smooth motion)
DT = 0.02  # approximate; env dt not exposed in processed state
ANG_ACC_W = 0.05

# Mild knee compliance bias (only works if these indices correspond to knees)
# In this codebase, `state[0:12]` is robot_qpos.
# XML Order from booster_lower_t1.xml:
#  0: Left_Hip_Pitch, 1: Left_Hip_Roll, 2: Left_Hip_Yaw, 3: Left_Knee_Pitch, 4: Left_Ankle_Pitch, 5: Left_Ankle_Roll
#  6: Right_Hip_Pitch, 7: Right_Hip_Roll, 8: Right_Hip_Yaw, 9: Right_Knee_Pitch, 10: Right_Ankle_Pitch, 11: Right_Ankle_Roll
#
# Previously we sliced state[7:12] which made Index 9 appear as local index 2.
# Now we use the full state[0:12] and target indices [3, 9].
KNEE_LOCAL_IDXS = [3, 9]     # Left Knee, Right Knee
KNEE_REST_ANGLE = 0.25       # straighter (was 0.5)
KNEE_W = 0.05                # stronger bias (was 0.02)

# Prevent "one knee absorbs everything" solutions (hard hyper-flexion)
KNEE_MAX_PREF = 1.2       # radians; above this gets penalized (tune 1.0–1.6)
KNEE_HYPER_W = 0.10

# Push up reward: encourage extending knees when bent
PUSH_UP_W = 0.2

# Ankle stiffness/damping reward to prevent flailing
# Indices for Ankles (Pitch/Roll)
# Left: 4, 5; Right: 10, 11
ANKLE_LOCAL_IDXS = [4, 5, 10, 11]
# ANKLE_STAB_W = 0.1  # Penalize high velocity in ankles specifically
# ANKLE_EMERGENCY_SCALE = 0.1  # How much to relax stabilization during emergency (0.0=free, 1.0=same)

# Emergency Step Reward: Reward rapid leg motion ONLY when falling fast
# EMERGENCY_STEP_W = 0.05
# EMERGENCY_ANG_VEL_THRESH = 1.5

# Knee index probe (recommended once to identify which joint_pos indices correspond to knees)
KNEE_PROBE = True
KNEE_PROBE_EPISODES = 5
KNEE_PROBE_PRINT_EVERY_STEPS = 0  # 0 disables per-step prints; set e.g. 25 if you want


# Termination
FALL_TILT = 0.65
DRIFT_SPEED_XY = 2.0

# Optional: encourage asymmetry only during high urgency (stepping is asymmetric)
ASYM_W = 0.02
ASYM_URGENCY_THRESHOLD = 1.0

# Control / smoothness
CTRL_W = 0.01
SMOOTH_W = 0.01

# Logging
LOG_WINDOW_EPISODES = 20
LOG_EVERY_EPISODES = 10


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done),
        )

    def __len__(self):
        return len(self.buffer)


def train_balance():
    env = MultiTaskEnv(ENV_IDS)
    preprocessor = Preprocessor()

    dummy_obs, dummy_info = env.reset()
    n_features = preprocessor.modify_state(dummy_obs, dummy_info).shape[1]

    agent = SAC(
        n_features=n_features,
        action_space=env.action_space,
        # Slightly wider than pure standing; recovery needs motion
        log_std_min=-4.0,
        log_std_max=-0.5,
        alpha=0.03,
        alpha_decay=0.9999,
        alpha_min=0.01,
        device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(f"[train_balance] SAC device: {agent.device}")

    # Load standing checkpoint as initialization
    stand_ckpt_path = os.path.join(os.path.dirname(__file__), "stand_models", "sac_stand_checkpoint.pth")
    # if os.path.exists(stand_ckpt_path):
    #     print(f"[train_balance] Loading standing checkpoint: {stand_ckpt_path}")
    #     agent.load_state_dict(torch.load(stand_ckpt_path, map_location=agent.device))
    # else:
    #     print(f"[train_balance] WARNING: stand checkpoint not found at: {stand_ckpt_path}")
    print("[train_balance] STARTING FRESH (No checkpoint loaded) to test pure balance learning.")

    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    action_function = get_action_function(env.action_space)

    # Initialize WandB
    wandb.init(
        project="booster-balance",
        name="balance-recovery",
        config={
            "timesteps": TIMESTEPS,
            "batch_size": BATCH_SIZE,
            "policy_clip": POLICY_ACTION_CLIP,
            "stable_tilt": STABLE_TILT,
            "recovery_tilt": RECOVERY_TILT,
            "push_up_w": PUSH_UP_W,
            # "ankle_stab_w": ANKLE_STAB_W,
            # "emergency_step_w": EMERGENCY_STEP_W,
        }
    )

    total_steps = 0
    episode_num = 0
    best_episode_reward = -float("inf")
    recent_episode_rewards = deque(maxlen=LOG_WINDOW_EPISODES)
    last_logged_episode = -1
    pbar = tqdm(total=TIMESTEPS)

    model_dir = os.path.join(os.path.dirname(__file__), "balance_models")
    os.makedirs(model_dir, exist_ok=True)

    while total_steps < TIMESTEPS:
        obs, info = env.reset()
        state = preprocessor.modify_state(obs, info).squeeze()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        # Reference pose (qpos) at reset. In this codebase, state[:12] corresponds to robot_qpos.
        qpos0 = state[:12].copy()

        # Knee probe bookkeeping
        joint_pos0 = state[0:12].copy() if state.shape[0] >= 12 else np.zeros(12, dtype=np.float32)
        joint_min = joint_pos0.copy()
        joint_max = joint_pos0.copy()
        joint_abs_delta_sum = np.zeros_like(joint_pos0)

        prev_base_ang_vel = state[27:30].copy()
        prev_base_lin_vel = state[33:36].copy()
        prev_action = np.zeros(env.action_space.shape, dtype=np.float32)

        if KNEE_PROBE and episode_num < KNEE_PROBE_EPISODES:
            print(f"\n[knee_probe] episode {episode_num} reset joint_pos (state[0:12]): {joint_pos0}")

        while not done and total_steps < TIMESTEPS:
            # Policy-space action in [-1, 1]
            if total_steps < WARMUP_STEPS:
                # FIX: Use the agent's policy even during warmup, just with more noise
                # This fills buffer with "standing-like" data, not random spasms.
                with torch.no_grad():
                     action = agent.select_action(state, evaluate=False)
                action = action + np.random.normal(0.0, 0.1, size=action.shape)
            else:
                action = agent.select_action(state)

            # Small policy-space perturbation (kept small to avoid teaching "self-noise")
            if np.random.rand() < PERTURB_PROB:
                action = action + np.random.normal(0.0, PERTURB_POLICY_STD, size=action.shape)

            # Dynamic clip based on *current* fall urgency (before stepping)
            # Small tilt => keep actions small; fast fall => unlock leg/hip range.
            cur_gravity = state[24:27]
            cur_tilt = float(np.linalg.norm(cur_gravity[:2]))
            cur_ang = state[27:30]
            
            # Predictive urgency: use angular acceleration to unlock sooner
            # (cur_ang - prev_base_ang_vel) is change over last step
            ang_acc_est = float(np.linalg.norm(cur_ang - prev_base_ang_vel)) / DT
            
            fall_urgency_pre = float(
                np.clip(
                    np.linalg.norm(cur_ang) * URGENCY_W_ANG + 
                    URGENCY_W_TILT * cur_tilt + 
                    URGENCY_W_ANG_ACC * ang_acc_est, 
                    0.0, URGENCY_MAX
                )
            )
            dynamic_clip = float(POLICY_ACTION_CLIP + DYNAMIC_CLIP_W * fall_urgency_pre)
            # IMPORTANT: policy outputs are designed for [-1, 1]. Anything beyond just saturates
            # the action scaler and can produce max torques (looks like violent knee bends).
            dynamic_clip = min(dynamic_clip, 1.0)

            # Clip and scale
            action = np.clip(action, -dynamic_clip, dynamic_clip)
            scaled_action = action_function(action)

            # External disturbance in actuator space (simulated push / torque disturbance)
            if np.random.rand() < PERTURB_PROB:
                torque_std = PERTURB_TORQUE_STD_FRAC * (env.action_space.high - env.action_space.low)
                scaled_action = scaled_action + np.random.normal(0.0, torque_std, size=scaled_action.shape)
                scaled_action = np.clip(scaled_action, env.action_space.low, env.action_space.high)

            next_obs, _, terminated, truncated, info = env.step(scaled_action)
            next_state = preprocessor.modify_state(next_obs, info).squeeze()

            # --- BALANCE RECOVERY REWARD ---
            gravity_vec = next_state[24:27]
            tilt = float(np.linalg.norm(gravity_vec[:2]))
            prev_gravity_vec = state[24:27]
            prev_tilt_for_progress = float(np.linalg.norm(prev_gravity_vec[:2]))

            # Survival + upright
            survival_reward = SURVIVAL_W
            upright_reward = UPRIGHT_W * (1.0 - tilt)
            if tilt > RECOVERY_TILT:
                upright_reward *= UPRIGHT_BOOST_WHEN_TILTED

            # Drift (IMU velocimeter)
            base_lin_vel = next_state[33:36]
            drift_cost = DRIFT_W * float(np.sum(np.square(base_lin_vel[:2])))

            # Bounded joint velocity penalty (allow recovery motion)
            robot_qvel = next_state[12:24]
            joint_qvel = robot_qvel[6:] if robot_qvel.shape[0] >= 7 else robot_qvel
            vel_mag = float(np.linalg.norm(joint_qvel))
            excess_vel = max(0.0, vel_mag - VEL_THRESHOLD)
            vel_cost = min(1.0, VEL_W * (excess_vel**2))

            # Angular acceleration cost (smooth corrections)
            base_ang_vel = next_state[27:30]
            prev_ang_mag = float(np.linalg.norm(prev_base_ang_vel))
            ang_mag = float(np.linalg.norm(base_ang_vel))
            ang_acc = (base_ang_vel - prev_base_ang_vel) / DT
            ang_acc_cost = min(1.0, ANG_ACC_W * float(np.sum(np.square(ang_acc))))
            prev_base_ang_vel = base_ang_vel.copy()

            # Knee flex preference (soft; depends on correct indexing)
            joint_pos = next_state[0:12] if next_state.shape[0] >= 12 else np.zeros(12, dtype=np.float32)
            knee_cost = 0.0
            for i in KNEE_LOCAL_IDXS:
                if 0 <= i < joint_pos.shape[0]:
                    knee_cost += float((joint_pos[i] - KNEE_REST_ANGLE) ** 2)
            knee_cost = KNEE_W * knee_cost

            # Penalize excessive knee flexion (keeps recovery from collapsing into deep one-knee squats)
            knee_hyper_cost = 0.0
            for i in KNEE_LOCAL_IDXS:
                if 0 <= i < joint_pos.shape[0]:
                    excess = max(0.0, float(joint_pos[i]) - KNEE_MAX_PREF)
                    knee_hyper_cost += excess**2
            knee_hyper_cost = min(1.0, KNEE_HYPER_W * knee_hyper_cost)

            # Pose regularization: keep joint positions close to reset pose when stable.
            # This helps avoid "always crouch/always fold" solutions while still allowing recovery motion.
            qpos = next_state[:12]
            pose_cost = POSE_W * float(np.sum(np.square(qpos - qpos0)))

            # Knee probe stats (which indices actually move like knees?)
            if KNEE_PROBE and episode_num < KNEE_PROBE_EPISODES:
                joint_min = np.minimum(joint_min, joint_pos)
                joint_max = np.maximum(joint_max, joint_pos)
                joint_abs_delta_sum += np.abs(joint_pos - joint_pos0)
                if KNEE_PROBE_PRINT_EVERY_STEPS and (episode_steps % KNEE_PROBE_PRINT_EVERY_STEPS == 0):
                    print(f"[knee_probe] ep {episode_num} step {episode_steps} joint_pos={joint_pos}")

            # Control / smoothness
            ctrl_cost = CTRL_W * float(np.sum(np.square(action)))
            smooth_cost = SMOOTH_W * float(np.sum(np.square(action - prev_action)))
            prev_action = action

            # Fall urgency from next_state (tilt + ang vel) => scale penalties down when falling fast
            fall_urgency_now = float(
                np.clip(np.linalg.norm(base_ang_vel) * URGENCY_W_ANG + URGENCY_W_TILT * tilt, 0.0, URGENCY_MAX)
            )
            penalty_scale = 1.0 / (1.0 + fall_urgency_now)

            # Tilt regimes: stable => prefer quiet; unstable => allow motion
            if tilt >= STABLE_TILT:
                vel_cost *= 0.2
                ang_acc_cost *= 0.2
                pose_cost *= POSE_RECOVERY_SCALE
                knee_cost *= 0.2
                knee_hyper_cost *= 0.2

            # Urgency-modulate all motion penalties (unlock legs during fast falls)
            vel_cost *= penalty_scale
            ang_acc_cost *= penalty_scale
            ctrl_cost *= penalty_scale
            smooth_cost *= penalty_scale
            pose_cost *= penalty_scale
            knee_cost *= penalty_scale
            knee_hyper_cost *= penalty_scale

            # Optional asymmetry reward when urgency is high (stepping tends to be asymmetric)
            asym_reward = 0.0
            if fall_urgency_now > ASYM_URGENCY_THRESHOLD and action.shape[0] % 2 == 0:
                half = action.shape[0] // 2
                left = action[:half]
                right = action[half:]
                asym_reward = ASYM_W * float(np.linalg.norm(left - right))

            # Early reaction reward: if tilt is small but ang vel is high, reward joint velocity
            early_react_reward = 0.0
            if tilt < STABLE_TILT and ang_mag > EARLY_REACT_ANG_THRESH:
                early_react_reward = EARLY_REACT_W * vel_mag

            # CoM Accel Reward: reward reversing direction when tilted
            # We want to reward if current velocity (base_lin_vel) opposes previous velocity (prev_base_lin_vel)
            com_accel_reward = 0.0
            if tilt > RECOVERY_TILT:
                 dot_prod = float(np.dot(base_lin_vel[:2], prev_base_lin_vel[:2]))
                 # If dot_prod is negative, we reversed.
                 com_accel_reward = COM_ACCEL_W * max(0.0, -dot_prod)

            # Push up reward: if knees are bent, reward extending them (negative velocity)
            push_up_reward = 0.0
            # joint_pos/robot_qvel are 12-dim arrays
            avg_knee_pos = float(np.mean(joint_pos[KNEE_LOCAL_IDXS]))
            if avg_knee_pos > 0.35:  # If bent significantly
                avg_knee_vel = float(np.mean(robot_qvel[KNEE_LOCAL_IDXS]))
                # Positive angle = flexion, so negative velocity = extension
                if avg_knee_vel < 0:
                    push_up_reward = PUSH_UP_W * (-avg_knee_vel)

            # Ankle Stabilization: penalize high velocity in ankles to prevent "flailing"
            # We want them to act as a solid base, not loose appendages.
            ankle_stab_cost = 0.0
            emergency_step_reward = 0.0
            
            # if robot_qvel.shape[0] >= 12:
            #     ankle_vels = robot_qvel[ANKLE_LOCAL_IDXS]
            #     ankle_stab_cost = ANKLE_STAB_W * float(np.sum(np.square(ankle_vels)))
            #     
            #     # Unlock ankles slightly during high urgency to allow stepping/balance recovery
            #     if tilt >= STABLE_TILT:
            #          ankle_stab_cost *= ANKLE_EMERGENCY_SCALE
            #
            #     # Emergency Step Reward: If falling fast (high angular velocity), reward leg velocity 
            #     # to encourage a quick step/reaction.
            #     if ang_mag > EMERGENCY_ANG_VEL_THRESH:
            #         # Sum of squared velocities of all leg joints
            #         leg_vel_sq = float(np.sum(np.square(robot_qvel[0:12])))
            #         emergency_step_reward = EMERGENCY_STEP_W * leg_vel_sq

            # Termination (tilt/drift)
            termination_reason = None
            if tilt > FALL_TILT:
                reward = -2.0
                terminated = True
                termination_reason = "fell_tilt"
            elif float(np.linalg.norm(base_lin_vel[:2])) > DRIFT_SPEED_XY:
                reward = -2.0
                terminated = True
                termination_reason = "drift_xy"
            else:
                # Explicit recovery progress shaping:
                # reward getting less tilted and reducing angular speed (fall rate).
                tilt_progress = (prev_tilt_for_progress - tilt)
                angvel_progress = (prev_ang_mag - ang_mag)
                progress_reward = 0.0
                if tilt >= STABLE_TILT:
                    progress_reward = TILT_PROGRESS_W * float(tilt_progress) + ANGVEL_DAMP_W * float(angvel_progress)

                reward = (
                    survival_reward
                    + upright_reward
                    + asym_reward
                    + progress_reward
                    + early_react_reward
                    + com_accel_reward
                    + push_up_reward
                    + emergency_step_reward
                    - drift_cost
                    - vel_cost
                    - ang_acc_cost
                    - knee_cost
                    - knee_hyper_cost
                    - pose_cost
                    - ctrl_cost
                    - smooth_cost
                    - ankle_stab_cost
                )

            # Enforce shorter episodes
            episode_steps += 1
            if episode_steps >= MAX_EPISODE_STEPS:
                truncated = True

            done = bool(terminated or truncated)
            episode_reward += float(reward)

            replay_buffer.push(state, action, reward, next_state, done)
            prev_base_lin_vel = base_lin_vel.copy()
            state = next_state
            total_steps += 1
            pbar.update(1)

            # Optimizing: Batch updates to reduce Python/MPS context switching overhead
            # Only start training after warmup
            q_loss = 0.0
            pi_loss = 0.0
            if total_steps > WARMUP_STEPS and len(replay_buffer) > BATCH_SIZE:
                if total_steps % UPDATE_INTERVAL == 0:
                    for _ in range(UPDATES_PER_INTERVAL):
                        q_loss, pi_loss = agent.update(*replay_buffer.sample(BATCH_SIZE))
            
            # WandB logging (step-wise, but maybe throttle to avoid overhead)
            # Log every UPDATE_INTERVAL steps when we actually update?
            # Or just rely on the episode loop for main metrics. 
            # We can log losses if they were updated.
            if total_steps > WARMUP_STEPS and total_steps % UPDATE_INTERVAL == 0:
                wandb.log({
                   "train/q_loss": q_loss,
                   "train/pi_loss": pi_loss,
                   "train/alpha": agent.alpha,
                }, step=total_steps)

        episode_num += 1

        if KNEE_PROBE and (episode_num - 1) < KNEE_PROBE_EPISODES:
            steps = max(1, episode_steps)
            joint_range = joint_max - joint_min
            avg_abs_delta = joint_abs_delta_sum / steps
            # Heuristic: MuJoCo knee pitch is usually ~[0, 2.3] (mostly non-negative, large positive flexion).
            knee_like = [
                int(i)
                for i in range(joint_pos0.shape[0])
                if (joint_min[i] > -0.05) and (joint_max[i] > 0.8)
            ]
            print(
                f"[knee_probe] episode {episode_num - 1} summary:\n"
                f"  joint_min={joint_min}\n"
                f"  joint_max={joint_max}\n"
                f"  range={joint_range}\n"
                f"  avg_abs_delta_from_reset={avg_abs_delta}\n"
                f"  knee_like_candidates (min>-0.05 and max>0.8): {knee_like}\n"
                f"  suggestion: pick knee indices as the 1-2 dims with largest range/avg_abs_delta\n"
            )
        recent_episode_rewards.append(float(episode_reward))
        avg_recent_reward = float(np.mean(recent_episode_rewards)) if len(recent_episode_rewards) > 0 else float(episode_reward)

        # Save best
        if episode_reward > best_episode_reward:
            best_episode_reward = float(episode_reward)
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_balance_best.pth"))

        # Periodic logs
        if episode_num > 0 and (episode_num % LOG_EVERY_EPISODES == 0) and (episode_num != last_logged_episode):
            tqdm.write(
                f"[balance] Ep {episode_num} | "
                f"Avg({len(recent_episode_rewards)}): {avg_recent_reward:.2f} | "
                f"BestEp: {best_episode_reward:.2f} | "
                f"EpSteps: {episode_steps} | "
                f"TotalSteps: {total_steps}"
            )
            wandb.log({
                "episode/reward": episode_reward,
                "episode/length": episode_steps,
                "episode/avg_reward": avg_recent_reward,
                "episode/best_reward": best_episode_reward,
            }, step=total_steps)
            last_logged_episode = episode_num

        # Periodic checkpoint
        if episode_num % 50 == 0:
            torch.save(agent.state_dict(), os.path.join(model_dir, "sac_balance_checkpoint.pth"))

    torch.save(agent.state_dict(), os.path.join(model_dir, "sac_balance_final.pth"))
    print("[train_balance] Training complete.")
    wandb.finish()
    env.close()


if __name__ == "__main__":
    train_balance()


