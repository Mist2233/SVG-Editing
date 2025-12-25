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
    print(f"Processing (Constrained): {svg_path} -> {output_path}")
    
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
    try:
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    except Exception as e:
        print(f"Error parsing SVG {svg_path}: {e}")
        raise e
    
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

    # ==========================================
    # [关键步骤] 初始化：强制规则 (Enforce Rules)
    # ==========================================
    print("  [Init] Enforcing rules on Stroke and Fill...")
    
    # 1. 强制把描边变细，防止遮挡
    for path in shapes:
        path.points = path.points.to(device)
        path.points.requires_grad = False # 锁死形状
        
        # 强制重置描边宽度为 1.5
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width.data.fill_(1.5)
            path.stroke_width = path.stroke_width.to(device)
        else:
            path.stroke_width = torch.tensor(1.5).to(device)
            
        path.stroke_width.requires_grad = False # 锁死描边宽度 (配合 Init 的 1.5，这就是最强的 Constraint)

    # 2. 随机化颜色 (消灭紫色)
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
            # 随机化 RGB
            group.fill_color.data[:3] = torch.rand(3).to(device)
            # Alpha 稍后在优化器里强制设为 1.0
            
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)
            # 随机化 RGB
            group.stroke_color.data[:3] = torch.rand(3).to(device)
            # Alpha 保持原样或随机化? 随机化比较好
            group.stroke_color.data[3] = torch.rand(1).to(device)

    # 4. Optimizer Setup
    optim_vars = []
    
    for group in shape_groups:
        # --- Fill Color (主角) ---
        if group.fill_color is not None:
            current_rgb = group.fill_color.data[:3].clone()
            
            # [关键] 强制 Fill 为 100% 不透明！
            # 这样颜色会非常鲜艳，不会泛白，也不会透出背景
            group.fixed_alpha = torch.tensor(1.0).to(device)
            
            rgb_var = current_rgb.clone().detach().requires_grad_(True)
            group.custom_rgb_var = rgb_var 
            optim_vars.append(rgb_var)
            
        # --- Stroke Color (配角) ---
        if group.stroke_color is not None:
            current_rgb = group.stroke_color.data[:3].clone()
            current_alpha = group.stroke_color.data[3].clone()
            
            # [关键] Stroke 允许调节透明度
            # 这样如果优化器觉得不需要描边，可以把 alpha 降为 0
            rgb_var = current_rgb.clone().detach().requires_grad_(True)
            alpha_var = current_alpha.clone().detach().requires_grad_(True)
            
            group.custom_stroke_rgb_var = rgb_var
            group.custom_stroke_alpha_var = alpha_var
            
            optim_vars.append(rgb_var)
            optim_vars.append(alpha_var)
            
    # 优化所有颜色变量
    optimizer = torch.optim.Adam(optim_vars, lr=0.05)

    # Background for composition
    background_color = torch.tensor([1.0, 1.0, 1.0]).to(device)
    
    # 5. Loop
    print(f"--> Start Constrained Refine ({num_iter} iterations)...")
    
    for t in range(num_iter):
        optimizer.zero_grad()
        
        # Reconstruct Colors
        for group in shape_groups:
            # Fill: Optimize RGB + Fixed Alpha (1.0)
            if hasattr(group, 'custom_rgb_var'):
                group.fill_color = torch.cat([group.custom_rgb_var, group.fixed_alpha.unsqueeze(0)])
            
            # Stroke: Optimize RGB + Optimize Alpha
            if hasattr(group, 'custom_stroke_rgb_var'):
                group.stroke_color = torch.cat([group.custom_stroke_rgb_var, group.custom_stroke_alpha_var.unsqueeze(0)])
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # White Background Composition
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
        
        # Clamp All Vars
        for var in optim_vars:
            var.data.clamp_(0.0, 1.0)
            
        # [额外保险] 虽然 stroke_width 被锁死，但如果未来开启优化，这里限制它不超过 3.0
        for path in shapes:
            path.stroke_width.data.clamp_(0.5, 3.0)
                
        if t % 50 == 0:
            print(f"    Iter {t}: Loss = {loss.item():.6f}")
            
    # 6. Save
    for group in shape_groups:
        if hasattr(group, 'custom_rgb_var'):
            group.fill_color = torch.cat([group.custom_rgb_var, group.fixed_alpha.unsqueeze(0)])
        if hasattr(group, 'custom_stroke_rgb_var'):
            group.stroke_color = torch.cat([group.custom_stroke_rgb_var, group.custom_stroke_alpha_var.unsqueeze(0)])
            
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)

def main():
    parser = argparse.ArgumentParser(description="Batch process SVG: Constrained Refine (Solid Fill + Thin Stroke)")
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
        
        # 优先寻找 _fixed.svg
        fixed_svgs = [s for s in svgs if "_fixed" in s.name and "thicken" not in s.name]
        
        if fixed_svgs:
            print(f"Found cleaned SVG: {fixed_svgs[0].name}")
            svg_file = fixed_svgs[0]
        else:
            print(f"No cleaned SVG found in {d}, checking others...")
            valid_svgs = [s for s in svgs if "external" not in s.name and "fast" not in s.name and "fix" not in s.name and "coloronly" not in s.name and "solid" not in s.name and "smart" not in s.name and "lineart" not in s.name]
            if not valid_svgs:
                 valid_svgs = [s for s in svgs if "v3" in s.name]
            
            if not valid_svgs:
                print(f"Skipping {d}: No suitable SVG found")
                continue
            svg_file = valid_svgs[0]
             
        pngs = list(d.glob("*.png"))
        
        if not pngs:
            print(f"Skipping {d}: Missing PNG")
            continue
            
        png_file = pngs[0]
        
        # 输出文件名
        output_name = f"{svg_file.stem}-constrained.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
