import pydiffvg
import torch
import skimage.io
import argparse
import os
import sys
from pathlib import Path
import numpy as np

# 强制 stdout 实时输出
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def optimize_pair(svg_path, png_path, output_path, num_iter=200):
    print(f"Processing: {svg_path} -> {output_path}")
    
    # 1. Init GPU
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
        
    # 2. Load Target Image
    target_np = skimage.io.imread(png_path)
    target = torch.from_numpy(target_np).to(torch.float32)
    if target.max() > 1.05:
        target = target / 255.0
    target = target.to(device)
    
    if target.shape[2] == 3:
        target = torch.cat([target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2)
    
    # 3. Load SVG
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    
    # Auto-Scale
    target_h, target_w = target.shape[0], target.shape[1]
    if canvas_width != target_w or canvas_height != target_h:
        scale_x = target_w / canvas_width
        scale_y = target_h / canvas_height
        for path in shapes:
            path.points[:, 0] *= scale_x
            path.points[:, 1] *= scale_y
            path.stroke_width *= (scale_x + scale_y) / 2.0
        canvas_width = target_w
        canvas_height = target_h

    # Move to GPU & Fix Geometry
    for path in shapes:
        path.points = path.points.to(device)
        path.points.requires_grad = False # 锁死形状
        
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width = path.stroke_width.to(device)
        else:
            path.stroke_width = torch.tensor(path.stroke_width).to(device)
        path.stroke_width.requires_grad = False # 锁死描边宽度

    # 4. Optimizer Setup (Force Solid Alpha = 1.0)
    rgb_optim_vars = []
    
    for group in shape_groups:
        # --- Fill Color ---
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
            current_rgb = group.fill_color.data[:3].clone()
            
            # [核心修改] 强制 Alpha = 1.0
            # 不再使用 group.fill_color.data[3].clone()
            forced_alpha = torch.tensor(1.0).to(device)
            
            rgb_var = current_rgb.clone().detach().requires_grad_(True)
            
            group.custom_rgb_var = rgb_var
            group.fixed_alpha = forced_alpha
            
            rgb_optim_vars.append(rgb_var)
            
        # --- Stroke Color ---
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)
            current_rgb = group.stroke_color.data[:3].clone()
            
            # [核心修改] 强制 Alpha = 1.0
            forced_stroke_alpha = torch.tensor(1.0).to(device)
            
            rgb_var = current_rgb.clone().detach().requires_grad_(True)
            
            group.custom_stroke_rgb_var = rgb_var
            group.fixed_stroke_alpha = forced_stroke_alpha
            
            rgb_optim_vars.append(rgb_var)
            
    # 只优化 RGB
    optimizer = torch.optim.Adam(rgb_optim_vars, lr=0.05)

    # Background for composition
    background_color = torch.tensor([1.0, 1.0, 1.0]).to(device)
    
    # 5. Loop
    print(f"--> Start Solid-Color Refine ({num_iter} iterations)...")
    
    for t in range(num_iter):
        optimizer.zero_grad()
        
        # Reconstruct Colors with Forced Alpha
        for group in shape_groups:
            if hasattr(group, 'custom_rgb_var'):
                # Fill: Optimized RGB + 1.0
                group.fill_color = torch.cat([group.custom_rgb_var, group.fixed_alpha.unsqueeze(0)])
            
            if hasattr(group, 'custom_stroke_rgb_var'):
                # Stroke: Optimized RGB + 1.0
                group.stroke_color = torch.cat([group.custom_stroke_rgb_var, group.fixed_stroke_alpha.unsqueeze(0)])
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # White Background Composition
        # 即使 Alpha 是 1.0，这个公式依然适用（1.0 * Color + 0.0 * White = Color）
        # 但它能处理边缘的抗锯齿部分（Anti-aliasing），这是让边界平滑的关键
        alpha = img[:, :, 3:4]
        color = img[:, :, :3]
        img_composed = color * alpha + background_color * (1 - alpha)
        
        target_rgb = target[:, :, :3]
        if target.shape[2] == 4:
             target_alpha = target[:, :, 3:4]
             target_rgb = target[:, :, :3] * target_alpha + background_color * (1 - target_alpha)

        # MSE Loss
        loss = (img_composed - target_rgb).pow(2).mean()
        
        loss.backward()
        
        optimizer.step()
        
        # Clamp Only RGB Vars
        for var in rgb_optim_vars:
            var.data.clamp_(0.0, 1.0)
                
        if t % 20 == 0:
            print(f"    Iter {t}: Loss = {loss.item():.6f}")
            
    # 6. Save (Save the final solid state)
    for group in shape_groups:
        if hasattr(group, 'custom_rgb_var'):
            group.fill_color = torch.cat([group.custom_rgb_var, group.fixed_alpha.unsqueeze(0)])
        if hasattr(group, 'custom_stroke_rgb_var'):
            group.stroke_color = torch.cat([group.custom_stroke_rgb_var, group.fixed_stroke_alpha.unsqueeze(0)])
            
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)

def main():
    parser = argparse.ArgumentParser(description="Batch process SVG: Solid Color Only (Force Alpha=1.0)")
    parser.add_argument("--root", default="edit_pair/color", help="Root directory to search")
    parser.add_argument("--iter", type=int, default=200, help="Number of iterations")
    parser.add_argument("--only", nargs="+", help="Specific folders to run (e.g. 1 5 7)")
    args = parser.parse_args()
    
    root_path = Path(args.root)
    if not root_path.exists():
        print(f"Error: Directory {root_path} not found.")
        return

    subdirs = [x for x in root_path.iterdir() if x.is_dir()]
    
    if args.only:
        target_names = set(args.only)
        subdirs = [d for d in subdirs if d.name in target_names]
    
    print(f"Found {len(subdirs)} subdirectories to process.")
    
    for d in subdirs:
        svgs = list(d.glob("*.svg"))
        # 排除之前的尝试，只找最干净的
        valid_svgs = [s for s in svgs if "external" not in s.name and "fast" not in s.name and "fix" not in s.name and "coloronly" not in s.name and "final" not in s.name]
        
        if not valid_svgs:
             valid_svgs = [s for s in svgs if "v3" in s.name]
             
        pngs = list(d.glob("*.png"))
        
        if not valid_svgs or not pngs:
            print(f"Skipping {d}: Missing suitable SVG or PNG")
            continue
            
        svg_file = valid_svgs[0]
        png_file = pngs[0]
        output_name = f"{svg_file.stem}-solid.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
