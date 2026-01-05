# get_spawn.py
import gymnasium as gym
import numpy as np
import inspect
import sai_mujoco.utils.v0.binding_utils as bu

# --- PATCHING (Required to load the env) ---
def make_patched_method(original_method):
    def patched(self, name):
        try: return original_method(self, name)
        except ValueError:
            try: return original_method(self, f"/{name}")
            except ValueError: raise
    return patched

TargetClass = None
for name, obj in inspect.getmembers(bu):
    if inspect.isclass(obj) and hasattr(obj, 'body_name2id'):
        TargetClass = obj
        break
if TargetClass:
    for m in ['body_name2id', 'geom_name2id', 'joint_name2id', 'site_name2id', 'sensor_name2id', 'actuator_name2id']:
        if hasattr(TargetClass, m): setattr(TargetClass, m, make_patched_method(getattr(TargetClass, m)))

# --- SCRIPT ---
ENV_ID = "LowerT1GoaliePenaltyKick-v0"
print(f"Resetting {ENV_ID}...")

env = gym.make(ENV_ID)
env.reset()

# Unwrap to get raw data
base = env.unwrapped
if hasattr(base, "model"): model, data = base.model, base.data
elif hasattr(base, "sim"): model, data = base.sim.model, base.sim.data
else: raise ValueError("Could not access MuJoCo")

# Get raw pointers if needed (SAI wrappers)
if hasattr(data, "ptr"): data = data.ptr
elif hasattr(data, "_data"): data = data._data

# The robot starts at index 9. Everything before that is World/Ball.
print("\n--- COPY THESE NUMBERS ---")
ball_spawn = data.qpos[:9].copy()
print(f"STATIC_SPAWN = {list(ball_spawn)}")
print("--------------------------")