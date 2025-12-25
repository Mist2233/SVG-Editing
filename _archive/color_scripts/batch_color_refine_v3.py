import pydiffvg
import torch
import skimage.io
import skimage.transform
import argparse
import os
import glob
import sys
# 强制 stdout 实时输出，不缓冲
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import ttools.modules  # 复用 refine_pipeline 的感知损失库

def optimize_pair(svg_path, png_path, output_path, num_iter=200, max_res=256, log_interval=20, optimize_points=True):
    print(f"Processing: {svg_path} -> {output_path}")
    
    # 1. Init GPU
    print("  [Step 1] Initializing GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
    print(f"  [Step 1] GPU Ready: {device}")
        
    # 2. Init LPIPS Loss (关键：复用之前的感知损失)
    print("  [Step 2] Initializing LPIPS loss (this might take a while if downloading)...")
    try:
        # 强制设置 torch hub 目录，避免权限问题
        torch.hub.set_dir("/workspace/.cache/torch")
        perception_loss = ttools.modules.LPIPS().to(device)
        print("  [Step 2] LPIPS Loaded Successfully.")
    except Exception as e:
        print(f"  [Error] Failed to load LPIPS: {e}")
        print("  Please ensure ttools is installed. Falling back to MSE.")
        perception_loss = None

    # 3. Load Target Image
    print(f"  [Step 3] Loading Target Image: {png_path}")
    target_np = skimage.io.imread(png_path)
    
    # Resize for speed
    h_orig, w_orig = target_np.shape[0], target_np.shape[1]
    if max_res > 0 and (h_orig > max_res or w_orig > max_res):
        scale = max_res / max(h_orig, w_orig)
        new_h, new_w = int(h_orig * scale), int(w_orig * scale)
        # print(f"  [Speedup] Resizing target from {w_orig}x{h_orig} to {new_w}x{new_h}")
        target_np = skimage.transform.resize(target_np, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(target_np.dtype)
    
    # Normalize
    target = torch.from_numpy(target_np).to(torch.float32)
    if target.max() > 1.05:
        target = target / 255.0
    target = target.to(device)
    
    # Ensure Target is RGBA
    if target.shape[2] == 3:
        target = torch.cat([target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2)
    
    # -----------------------------------------------------------
    # Pre-process Target for Loss (White Background Compositing)
    # 这步是解决“背景乱涂”的关键！
    # -----------------------------------------------------------
    bg = torch.ones(target.shape[0], target.shape[1], 3, device=device)
    # Blend: RGB * Alpha + White * (1 - Alpha)
    target_rgb_blended = target[:, :, 3:4] * target[:, :, :3] + (1 - target[:, :, 3:4]) * bg
    # Format for LPIPS: NCHW
    target_permuted = target_rgb_blended.unsqueeze(0).permute(0, 3, 1, 2)
    # -----------------------------------------------------------
    
    # 4. Load SVG
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    
    # Auto-Scale SVG
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

    # Move to GPU
    for path in shapes:
        path.points = path.points.to(device)
        path.stroke_width = path.stroke_width.to(device)
    for group in shape_groups:
        if group.fill_color is not None:
            # --- 强制初始化 Alpha = 1.0 (解决颜色蒙蒙的问题) ---
            group.fill_color.data[3] = 1.0
            group.fill_color = group.fill_color.to(device)
        if group.stroke_color is not None:
            # --- 强制初始化 Alpha = 1.0 ---
            group.stroke_color.data[3] = 1.0
            group.stroke_color = group.stroke_color.to(device)
            
    # 5. Optimizer Setup (Copying from refine_pipeline.py)
    points_vars = []
    color_vars = []
    
    for path in shapes:
        path.points.requires_grad = optimize_points
        if optimize_points:
            points_vars.append(path.points)
        
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
        if group.stroke_color is not None:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
            
    # 使用 refine_pipeline.py 的参数策略
    # Points LR = 1.0 (允许形状微调) -> 修改为 0.05 (Color Refinement 任务需要保持轮廓)
    # Color LR = 0.01 (refine_pipeline用了0.01，我们也可以试0.02，先保持一致)
    params_list = [{"params": color_vars, "lr": 0.05}] # 提高颜色 LR 到 0.05，让颜色更鲜艳准确
    if optimize_points and len(points_vars) > 0:
        params_list.append({"params": points_vars, "lr": 0.05}) # 大幅降低点 LR，防止轮廓崩坏
        
    optimizer = torch.optim.Adam(params_list)
    
    # 6. Loop
    # print(f"  Start optimizing (LPIPS + White BG)...")
    
    for t in range(num_iter):
        optimizer.zero_grad()
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # --- Compositing for Loss ---
        img_rgb_blended = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * bg
        img_permuted = img_rgb_blended.unsqueeze(0).permute(0, 3, 1, 2)
        
        # --- Loss Calculation ---
        loss = 0
        if perception_loss is not None:
            lpips_loss = perception_loss(img_permuted, target_permuted)
            loss += lpips_loss
        
        # Color Mean Loss (refine_pipeline logic)
        color_loss = (img_permuted.mean() - target_permuted.mean()).pow(2)
        loss += color_loss
        
        # 如果没有 LPIPS，回退到 MSE (全图 MSE)
        if perception_loss is None:
             loss += (img_rgb_blended - target_rgb_blended).pow(2).mean() * 100
        
        loss.backward()
        
        # Gradient modifications
        # refine_pipeline 没有锁 Alpha，所以我们也不锁，让它自由学习透明度
        
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)
                
        if t % log_interval == 0:
            print(f"    Iter {t}: Loss = {loss.item():.6f}")
            
    # 7. Save
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)
    # print(f"  Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Batch process SVG color refinement V3 (LPIPS)")
    parser.add_argument("--root", default="edit_pair/color", help="Root directory to search")
    parser.add_argument("--iter", type=int, default=200, help="Number of iterations")
    parser.add_argument("--only", nargs="+", help="Specific folders to run (e.g. 1 5 7)")
    # Default is optimize points=True, same as refine_pipeline
    parser.add_argument("--fix-points", action="store_true", help="If set, do not optimize points")
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
        # Prioritize cleaned SVG (_fixed.svg)
        fixed_svgs = list(d.glob("*_fixed.svg"))
        if fixed_svgs:
            svg_file = fixed_svgs[0]
            print(f"Found cleaned SVG: {svg_file.name}")
        else:
            svgs = list(d.glob("*.svg"))
            # Exclude previous results
            svgs = [s for s in svgs if not s.name.endswith("-m.svg") and not s.name.endswith("-v2.svg") and not s.name.endswith("-v3.svg") and not s.name.endswith("-thicken.svg")]
            
            if len(svgs) == 0:
                print(f"Skipping {d}: Missing suitable SVG (looking for *_fixed.svg or original)")
                continue
            svg_file = svgs[0]

        pngs = list(d.glob("*.png"))
        
        if len(pngs) == 0:
            print(f"Skipping {d}: Missing PNG")
            continue
        png_file = pngs[0]
        output_name = f"{svg_file.stem}-v3.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), 
                          num_iter=args.iter, 
                          optimize_points=not args.fix_points)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
