import gymnasium as gym
import inspect
import sai_mujoco.utils.v0.binding_utils as bu
import mujoco

# ==========================================
# 1. FULL PATCHING SUITE
# ==========================================
def make_patched_method(original_method):
    def patched(self, name):
        try: return original_method(self, name)
        except ValueError:
            try: return original_method(self, f"/{name}")
            except ValueError: raise
    return patched

def apply_patches():
    TargetClass = None
    for name, obj in inspect.getmembers(bu):
        if inspect.isclass(obj) and hasattr(obj, 'body_name2id'):
            TargetClass = obj
            break
    
    if TargetClass:
        # We must patch ALL of these to prevent init crashes
        methods = ['body_name2id', 'geom_name2id', 'joint_name2id', 
                   'site_name2id', 'sensor_name2id', 'actuator_name2id']
        for m in methods:
            if hasattr(TargetClass, m):
                setattr(TargetClass, m, make_patched_method(getattr(TargetClass, m)))

apply_patches()

# ==========================================
# 2. MAPPING LOGIC
# ==========================================
def print_structure(env_id="LowerT1GoaliePenaltyKick-v0"):
    print(f"🗺️  MAPPING JOINTS FOR: {env_id}")
    try:
        env = gym.make(env_id, render_mode="human")
        env.reset()
    except Exception as e:
        print(f"❌ Init failed: {e}")
        return

    # Extract raw model pointer
    base = env.unwrapped
    if hasattr(base, "model"): wrapper = base.model
    elif hasattr(base, "sim"): wrapper = base.sim.model
    else: return
    
    raw_model = wrapper.ptr if hasattr(wrapper, 'ptr') else wrapper._model if hasattr(wrapper, '_model') else wrapper

    print(f"\n{'ID':<5} | {'QPOS ADR':<10} | {'JOINT NAME'}")
    print("-" * 50)
    
    n_joints = raw_model.njnt
    
    for i in range(n_joints):
        name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        qpos_adr = raw_model.jnt_qposadr[i]
        print(f"{i:<5} | {qpos_adr:<10} | {name}")

    print("-" * 50)

if __name__ == "__main__":
    print_structure()