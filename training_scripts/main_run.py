import sys
import os

# Add root directory to path to find sai_patch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Add the current directory to path so imports work when running from inside training_scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import sai_patch
    print("[main_run] sai_patch imported.")
except ImportError:
    print("[main_run] Warning: sai_patch not found.")

from sai_rl import SAIClient

# Import the new training loop
from train_run import train_run


if __name__ == "__main__":
    # Optional: initialize SAI (not required for local training loop)
    try:
        _sai = SAIClient(
            comp_id="lower-t1-penalty-kick-goalie",
            api_key="sai_ddqEmPy1JIeQoGSI72BcdGUePbVdYtSj",
        )
    except Exception:
        print("SAIClient init failed (ok for local training).")

    print("Starting Stage 2 Training: Velocity/Run (SAC)")
    train_run()
    print("Run training finished.")