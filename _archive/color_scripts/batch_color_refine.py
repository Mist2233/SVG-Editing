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

def optimize_pair(svg_path, png_path, output_path, num_iter=50, max_res=256, log_interval=10):
    print(f"Processing: {svg_path} -> {output_path}")
    
    # 1. Init GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
        
    # 2. Load Target Image
    target_np = skimage.io.imread(png_path)
    
    # Resize for speed if too large
    h_orig, w_orig = target_np.shape[0], target_np.shape[1]
    if max_res > 0 and (h_orig > max_res or w_orig > max_res):
        scale = max_res / max(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        print(f"  [Speedup] Resizing target from {w_orig}x{h_orig} to {new_w}x{new_h}")
        target_np = skimage.transform.resize(target_np, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(target_np.dtype)
    
    target = torch.from_numpy(target_np).to(torch.float32) / 255.0
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
            # Scale control points
            path.points[:, 0] *= scale_x
            path.points[:, 1] *= scale_y
            
            # Scale stroke width (approximate with average scale)
            path.stroke_width *= (scale_x + scale_y) / 2.0
            
        # Update canvas size to match target
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
            
    # 4. Optimizer
    points_vars = []
    color_vars = []
    
    # Optimize both points and colors for best fit
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
        if group.stroke_color is not None:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
            
    # Lower LR for points to maintain structure, higher for color
    optimizer = torch.optim.Adam([
        {"params": points_vars, "lr": 0.5},
        {"params": color_vars, "lr": 0.05}
    ])
    
    # 5. Loop
    for t in range(num_iter):
        optimizer.zero_grad()
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # Simple MSE Loss
        loss = (img - target).pow(2).mean()
        
        loss.backward()
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)
                
        if t % log_interval == 0:
            print(f"  Iter {t}: Loss = {loss.item():.6f}")
            
    # 6. Save
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch process SVG color refinement")
    parser.add_argument("--root", default="edit_pair/color", help="Root directory to search")
    parser.add_argument("--iter", type=int, default=200, help="Number of iterations")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Directory {root_path} not found.")
        return

    # Find all subdirectories
    subdirs = [x for x in root_path.iterdir() if x.is_dir()]
    
    print(f"Found {len(subdirs)} subdirectories in {root_path}")
    
    for d in subdirs:
        # Find SVG (exclude already modified ones ending in -m.svg)
        svgs = list(d.glob("*.svg"))
        svgs = [s for s in svgs if not s.name.endswith("-m.svg")]
        
        # Find PNG
        pngs = list(d.glob("*.png"))
        
        if len(svgs) == 0 or len(pngs) == 0:
            print(f"Skipping {d}: Missing SVG or PNG")
            continue
            
        # Take the first pair found
        svg_file = svgs[0]
        png_file = pngs[0]
        
        output_name = f"{svg_file.stem}-m.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
