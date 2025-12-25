import pydiffvg
import torch
import skimage.io
import skimage.transform
import argparse
import os
import glob
import sys
from pathlib import Path
import numpy as np

def optimize_pair(svg_path, png_path, output_path, num_iter=200, max_res=256, log_interval=20, optimize_points=False):
    print(f"Processing: {svg_path} -> {output_path}")
    
    # 1. Init GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
        
    # 2. Load Target Image (Robust Normalization)
    target_np = skimage.io.imread(png_path)
    
    # Resize for speed if too large
    h_orig, w_orig = target_np.shape[0], target_np.shape[1]
    if max_res > 0 and (h_orig > max_res or w_orig > max_res):
        scale = max_res / max(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        print(f"  [Speedup] Resizing target from {w_orig}x{h_orig} to {new_w}x{new_h}")
        target_np = skimage.transform.resize(target_np, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(target_np.dtype)
    
    # Convert to Tensor and Normalize
    target = torch.from_numpy(target_np).to(torch.float32)
    
    # Check value range: if image is 0-255, divide by 255. If 0-1, keep as is.
    if target.max() > 1.05:
        target = target / 255.0
        
    target = target.to(device)
    
    # Handle Image Channels
    # If RGB, add Alpha=1
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )
    
    h, w = target.shape[0], target.shape[1]
    
    # 3. Load SVG
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    
    # -----------------------------------------------------------
    # Auto-Scale SVG to match Target PNG resolution (Small Res)
    # -----------------------------------------------------------
    target_h, target_w = target.shape[0], target.shape[1]
    
    if canvas_width != target_w or canvas_height != target_h:
        print(f"  [Auto-Scale] Resizing SVG from {canvas_width}x{canvas_height} to {target_w}x{target_h}")
        scale_x = target_w / canvas_width
        scale_y = target_h / canvas_height
        
        for path in shapes:
            path.points[:, 0] *= scale_x
            path.points[:, 1] *= scale_y
            path.stroke_width *= (scale_x + scale_y) / 2.0
            
        canvas_width = target_w
        canvas_height = target_h
    # -----------------------------------------------------------

    # Move to GPU
    for path in shapes:
        path.points = path.points.to(device)
        path.stroke_width = path.stroke_width.to(device)
        
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)
            
    # 4. Optimizer Setup
    points_vars = []
    color_vars = []
    
    # Only optimize points if explicitly requested (usually False for color tasks)
    for path in shapes:
        if optimize_points:
            path.points.requires_grad = True
            points_vars.append(path.points)
        else:
            path.points.requires_grad = False
        
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
        if group.stroke_color is not None:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
            
    # Optimizer settings
    params_list = [{"params": color_vars, "lr": 0.05}] # High LR for color
    if optimize_points and len(points_vars) > 0:
        params_list.append({"params": points_vars, "lr": 0.1})
        
    optimizer = torch.optim.Adam(params_list)
    
    # 5. Loop
    print(f"  Start optimizing (Color Only)...")
    
    # Create debug directory
    debug_dir = Path("output/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    for t in range(num_iter):
        optimizer.zero_grad()
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # Save Init and Target for debugging (at iter 0)
        if t == 0:
            pydiffvg.imwrite(img.cpu(), str(debug_dir / f"{Path(output_path).stem}_init.png"), gamma=1.0)
            pydiffvg.imwrite(target.cpu(), str(debug_dir / f"{Path(output_path).stem}_target.png"), gamma=1.0)
        
        # Simple MSE Loss
        loss = (img - target).pow(2).mean()
        
        loss.backward()
        
        # ---------------------------------------------------------------
        # CRITICAL FIX: Lock Alpha Channel (Gradient = 0)
        # We only want to optimize RGB, not Opacity/Alpha.
        # This prevents transparent shapes from appearing, or holes from filling.
        # ---------------------------------------------------------------
        for group in shape_groups:
            if group.fill_color is not None and group.fill_color.requires_grad:
                group.fill_color.grad[3] = 0.0
            if group.stroke_color is not None and group.stroke_color.requires_grad:
                group.stroke_color.grad[3] = 0.0
        # ---------------------------------------------------------------
        
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)
                
        if t % log_interval == 0:
            print(f"    Iter {t}: Loss = {loss.item():.6f}")
            
    # 6. Save
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch process SVG color refinement V2")
    parser.add_argument("--root", default="edit_pair/color", help="Root directory to search")
    parser.add_argument("--iter", type=int, default=200, help="Number of iterations")
    parser.add_argument("--only", nargs="+", help="Specific folders to run (e.g. 1 5 7)")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Directory {root_path} not found.")
        return

    subdirs = [x for x in root_path.iterdir() if x.is_dir()]
    
    # Filter specific folders if requested
    if args.only:
        target_names = set(args.only)
        subdirs = [d for d in subdirs if d.name in target_names]
        print(f"Filtered to run only: {args.only}")
    
    print(f"Found {len(subdirs)} subdirectories to process.")
    
    for d in subdirs:
        svgs = list(d.glob("*.svg"))
        # Exclude previous results (ending in -m.svg or -v2.svg)
        svgs = [s for s in svgs if not s.name.endswith("-m.svg") and not s.name.endswith("-v2.svg")]
        pngs = list(d.glob("*.png"))
        
        if len(svgs) == 0 or len(pngs) == 0:
            print(f"Skipping {d}: Missing SVG or PNG")
            continue
            
        svg_file = svgs[0]
        png_file = pngs[0]
        
        # Output name
        output_name = f"{svg_file.stem}-v2.svg"
        output_path = d / output_name
        
        try:
            # optimize_points=False ensures we only change colors!
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter, optimize_points=False)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
