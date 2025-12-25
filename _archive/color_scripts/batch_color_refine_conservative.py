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
    print(f"Processing (Conservative): {svg_path} -> {output_path}")
    
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
    # [Step 1] 初始化：保守策略 (Conservative Setup)
    # ==========================================
    print("  [Init] Conservative setup: Keep colors, Thin strokes, Freeze geometry.")
    
    # 1. 对描边做“抽脂手术”并冻结
    for path in shapes:
        path.points = path.points.to(device)
        path.points.requires_grad = False # 锁死形状
        
        # 强制把描边设为细线 (1.0)，露出内部填充
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width.data.fill_(1.0)
            path.stroke_width = path.stroke_width.to(device)
        else:
            path.stroke_width = torch.tensor(1.0).to(device)
            
        # [关键] 彻底锁死粗细，不许优化器动它！
        path.stroke_width.requires_grad = False
        
    # 2. 颜色变量准备 (保留原图颜色，不随机化)
    optim_vars = []
    
    for group in shape_groups:
        # --- Fill Color ---
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
            
            # [关键] 使用原图的 RGB 作为起点！(不要随机)
            current_rgb = group.fill_color.data[:3].clone()
            
            # [关键] 强制 Alpha = 1.0 (实心)
            group.fixed_alpha = torch.tensor(1.0).to(device)
            
            rgb_var = current_rgb.clone().detach().requires_grad_(True)
            group.custom_rgb_var = rgb_var 
            optim_vars.append(rgb_var)
            
        # --- Stroke Color ---
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)
            
            # 使用原图 RGB
            current_rgb = group.stroke_color.data[:3].clone()
            
            # [关键] 描边也强制实心
            group.fixed_stroke_alpha = torch.tensor(1.0).to(device)
            
            rgb_var = current_rgb.clone().detach().requires_grad_(True)
            group.custom_stroke_rgb_var = rgb_var
            
            optim_vars.append(rgb_var)
            
    # 3. 优化器
    # 学习率可以稍微大一点，因为起点已经很好了，需要快速收敛到微调后的颜色
    optimizer = torch.optim.Adam(optim_vars, lr=0.05)

    # Background
    background_color = torch.tensor([1.0, 1.0, 1.0]).to(device)
    
    # 4. Loop
    print(f"--> Start Conservative Refine ({num_iter} iterations)...")
    
    for t in range(num_iter):
        optimizer.zero_grad()
        
        # Reconstruct Colors
        for group in shape_groups:
            # Fill: Optimized RGB + Fixed Alpha (1.0)
            if hasattr(group, 'custom_rgb_var'):
                group.fill_color = torch.cat([group.custom_rgb_var, group.fixed_alpha.unsqueeze(0)])
            
            # Stroke: Optimized RGB + Fixed Alpha (1.0)
            if hasattr(group, 'custom_stroke_rgb_var'):
                group.stroke_color = torch.cat([group.custom_stroke_rgb_var, group.fixed_stroke_alpha.unsqueeze(0)])
        
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
        
        # Clamp RGB (0.0 - 1.0)
        for var in optim_vars:
            var.data.clamp_(0.0, 1.0)
            
        # 不需要 Clamp stroke_width，因为它是锁死的
                
        if t % 50 == 0:
            print(f"    Iter {t}: Loss = {loss.item():.6f}")
            
    # 5. Save
    for group in shape_groups:
        if hasattr(group, 'custom_rgb_var'):
            group.fill_color = torch.cat([group.custom_rgb_var, group.fixed_alpha.unsqueeze(0)])
        if hasattr(group, 'custom_stroke_rgb_var'):
            group.stroke_color = torch.cat([group.custom_stroke_rgb_var, group.fixed_stroke_alpha.unsqueeze(0)])
            
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)

def main():
    parser = argparse.ArgumentParser(description="Batch process SVG: Conservative Refine (No Random Init)")
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
            valid_svgs = [s for s in svgs if "external" not in s.name and "fast" not in s.name and "fix" not in s.name and "coloronly" not in s.name and "solid" not in s.name and "smart" not in s.name and "lineart" not in s.name and "constrained" not in s.name]
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
        output_name = f"{svg_file.stem}-conservative.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
