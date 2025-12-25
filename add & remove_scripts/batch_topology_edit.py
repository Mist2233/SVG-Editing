import os
import argparse
import subprocess
from pathlib import Path
import sys
import time

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    return True

def process_folder(folder_path, mode):
    print(f"\n{'='*50}")
    print(f"Processing Folder: {folder_path} (Mode: {mode})")
    print(f"{'='*50}")
    
    # 1. Find SVG and PNG
    svgs = list(folder_path.glob("*.svg"))
    pngs = list(folder_path.glob("*.png"))
    
    # Filter out previous results
    svgs = [s for s in svgs if "fixed" not in s.name and "pruning" not in s.name and "spawning" not in s.name]
    
    if not svgs or not pngs:
        print(f"Skipping {folder_path}: Missing raw SVG or PNG")
        return

    raw_svg = svgs[0]
    target_png = pngs[0]
    
    print(f"Raw SVG: {raw_svg.name}")
    print(f"Target PNG: {target_png.name}")
    
    # 2. Clean SVG
    print(f"\n[Step 1] Cleaning SVG...")
    clean_cmd = f"python clean_svg.py \"{raw_svg}\""
    if not run_command(clean_cmd):
        return
        
    fixed_svg = folder_path / f"{raw_svg.stem}_fixed.svg"
    if not fixed_svg.exists():
        print(f"Error: Cleaned SVG not found at {fixed_svg}")
        return
    print(f"Cleaned SVG: {fixed_svg.name}")

    # 3. Execute Mode
    
    # --- Mode: REMOVE (Pruning) ---
    if mode == "remove":
        print(f"\n[Step 2] Running Smart Pruning (Iterative LOO)...")
        timestamp = int(time.time())
        output_name = f"{raw_svg.stem}_final_pruned_{timestamp}"
        # Pruning: Smart Prune Pipeline
        prune_cmd = (
            f"python smart_prune.py \"{fixed_svg}\" \"{target_png}\" "
            f"\"output/{output_name}.svg\""
        )
        run_command(prune_cmd)
        
        # Move output to folder
        src_svg = Path(f"output/{output_name}.svg")
        src_png = Path(f"output/{output_name}.png")
        if src_svg.exists():
            dest_svg = folder_path / f"{output_name}.svg"
            os.rename(src_svg, dest_svg)
            print(f"Result saved to {dest_svg}")
        if src_png.exists():
            dest_png = folder_path / f"{output_name}.png"
            os.rename(src_png, dest_png)

    # --- Mode: ADD (Spawning) ---
    elif mode == "add":
        print(f"\n[Step 2] Running Dynamic Spawning (Add)...")
        timestamp = int(time.time())
        output_name = f"{raw_svg.stem}_final_spawned_{timestamp}"
        # Spawning: 200 iter, spawn every 50
        spawn_cmd = (
            f"python dynamic_spawning.py \"{target_png}\" "
            f"--init_svg \"{fixed_svg}\" "
            f"--output_name \"{output_name}\" "
            f"--num_iter 200 "
            f"--spawn_interval 50 "
            f"--max_paths 1000 "
            f"--use_mse"
        )
        run_command(spawn_cmd)
        
        # Move output to folder
        src_svg = Path(f"output/{output_name}.svg")
        src_png = Path(f"output/{output_name}.png")
        if src_svg.exists():
            dest_svg = folder_path / f"{output_name}.svg"
            os.rename(src_svg, dest_svg)
            print(f"Result saved to {dest_svg}")
        if src_png.exists():
            dest_png = folder_path / f"{output_name}.png"
            os.rename(src_png, dest_png)

    # --- Mode: MODIFY (Remove + Add) ---
    elif mode == "modify":
        # Step A: Prune (Remove old stuff) - Use Smart Prune Logic
        print(f"\n[Step 2A] Running Smart Pruning (Removing old content)...")
        timestamp = int(time.time())
        prune_name = f"{raw_svg.stem}_inter_pruned_{timestamp}"
        prune_cmd = (
            f"python smart_prune.py \"{fixed_svg}\" \"{target_png}\" "
            f"\"output/{prune_name}.svg\""
        )
        if not run_command(prune_cmd):
            return

        # Intermediate file
        inter_svg = Path(f"output/{prune_name}.svg")
        if not inter_svg.exists():
            print("Error: Pruning failed, cannot proceed to spawning.")
            return
            
        # Step B: Spawn (Add new stuff)
        print(f"\n[Step 2B] Running Dynamic Spawning (Adding new content)...")
        final_name = f"{raw_svg.stem}_final_modified_{timestamp}"
        spawn_cmd = (
            f"python dynamic_spawning.py \"{target_png}\" "
            f"--init_svg \"{inter_svg}\" "
            f"--output_name \"{final_name}\" "
            f"--num_iter 200 "
            f"--spawn_interval 50 "
            f"--max_paths 1000 "
            f"--use_mse"
        )
        run_command(spawn_cmd)
        
        # Move final output to folder
        src_svg = Path(f"output/{final_name}.svg")
        src_png = Path(f"output/{final_name}.png")
        if src_svg.exists():
            dest_svg = folder_path / f"{final_name}.svg"
            os.rename(src_svg, dest_svg)
            print(f"Result saved to {dest_svg}")
        if src_png.exists():
            dest_png = folder_path / f"{final_name}.png"
            os.rename(src_png, dest_png)

def main():
    root_dir = Path("edit_pair/add & remove")
    if not root_dir.exists():
        print(f"Error: Directory {root_dir} not found.")
        return

    # Define Tasks
    # 1, 2: Remove
    # 3: Modify
    # 4, 5: Add
    
    tasks = {
        "1": "remove",
        "2": "remove",
        "3": "modify",
        "4": "add",
        "5": "add"
    }
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", help="Run specific folders (e.g. 1 3)")
    args = parser.parse_args()
    
    target_folders = args.only if args.only else tasks.keys()
    
    for folder_name in target_folders:
        if folder_name not in tasks:
            print(f"Skipping unknown folder: {folder_name}")
            continue
            
        folder_path = root_dir / folder_name
        if not folder_path.exists():
            print(f"Folder not found: {folder_path}")
            continue
            
        mode = tasks[folder_name]
        process_folder(folder_path, mode)

if __name__ == "__main__":
    main()
