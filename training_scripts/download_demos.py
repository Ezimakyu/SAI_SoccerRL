# download from huggingface: https://huggingface.co/datasets/SaiResearch/booster_dataset/tree/main/soccer/booster_lower_t1

import os
import shutil
from huggingface_hub import snapshot_download

# --- CONFIGURATION ---
REPO_ID = "SaiResearch/booster_dataset"
REPO_SUBFOLDER = "soccer/booster_lower_t1"  # The specific folder you linked
TARGET_DIR = os.path.join(os.path.dirname(__file__), "demonstrations")

def download_and_flatten():
    print(f"--- Downloading demonstrations from {REPO_ID} ---")
    
    # 1. Download the specific subfolder
    # This maintains the folder structure temporarily (e.g. demonstrations/soccer/booster_lower_t1/...)
    download_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{REPO_SUBFOLDER}/*.npz",  # Only download .npz files
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False  # Download actual files, not symlinks
    )
    
    print("\n--- Flattening directory structure ---")
    
    # 2. Move files from the nested folders to the root of TARGET_DIR
    moved_count = 0
    for root, dirs, files in os.walk(TARGET_DIR):
        for file in files:
            if file.endswith(".npz"):
                source_path = os.path.join(root, file)
                dest_path = os.path.join(TARGET_DIR, file)
                
                # Only move if it's not already in the right place
                if source_path != dest_path:
                    shutil.move(source_path, dest_path)
                    moved_count += 1

    # 3. Cleanup empty nested folders
    # (We remove the 'soccer' folder and anything inside it that is now empty)
    nested_root = os.path.join(TARGET_DIR, "soccer")
    if os.path.exists(nested_root):
        shutil.rmtree(nested_root)

    print(f"Success! {moved_count} files moved to: {TARGET_DIR}")
    print(f"You can now run: python visualize_demos.py")

if __name__ == "__main__":
    download_and_flatten()