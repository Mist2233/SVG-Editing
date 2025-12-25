import pydiffvg
import torch
import skimage.io
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import time
from torch.optim.lr_scheduler import LambdaLR

# 强制 stdout 实时输出
sys.stdout.reconfigure(line_buffering=True)

# 强制关闭阻塞
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# --- 借鉴自 LIVE/main.py 的 LR Decay 策略 ---
class linear_decay_lrlambda_f(object):
    def __init__(self, decay_every, decay_ratio):
        self.decay_every = decay_every
        self.decay_ratio = decay_ratio

    def __call__(self, n):
        decay_time = n // self.decay_every
        decay_step = n % self.decay_every
        lr_s = self.decay_ratio**decay_time
        lr_e = self.decay_ratio**(decay_time+1)
        r = decay_step / self.decay_every
        lr = lr_s * (1-r) + lr_e * r
        return lr

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
    
    # Handle RGBA
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

    # Move to GPU
    for path in shapes:
        path.points = path.points.to(device)
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width = path.stroke_width.to(device)
        else:
            path.stroke_width = torch.tensor(path.stroke_width).to(device)
            
    for group in shape_groups:
        if group.fill_color is not None:
            # 借鉴 user 提到的 "蒙蒙的" 问题，尝试强制初始化 Alpha 为 1.0
            # 但这里我们先保持原样，通过 Loss 来驱动
            group.fill_color = group.fill_color.to(device)
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)
            
    # 4. Optimizer Setup
    points_vars = []
    color_vars = []
    
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
            
    # 分离参数组以应用不同的 Scheduler
    # Points: High LR -> Decay
    optimizer_points = torch.optim.Adam(points_vars, lr=1.0)
    # Colors: Initial LR set to 0.0 to freeze colors initially
    optimizer_colors = torch.optim.Adam(color_vars, lr=0.01)

    # Scheduler for Points (Linear Decay)
    # 1.0 -> 0.1
    lambda_points = lambda epoch: 1.0 - 0.9 * (epoch / num_iter)
    scheduler_points = LambdaLR(optimizer_points, lr_lambda=lambda_points)

    # --- 准备白色背景用于混合 ---
    background_color = torch.tensor([1.0, 1.0, 1.0]).to(device)
    
    # 定义分阶段优化的阈值
    # 前 30% 的迭代只优化形状 (Points)，锁定颜色
    warmup_iter = int(num_iter * 0.3) 

    # 5. Loop
    print(f"--> Start External Refine ({num_iter} iterations)...")
    print(f"    Phase 1: Shape Alignment (Iter 0-{warmup_iter}) - Colors Frozen")
    print(f"    Phase 2: Color Refinement (Iter {warmup_iter}-{num_iter})")

    for t in range(num_iter):
        optimizer_points.zero_grad()
        optimizer_colors.zero_grad()
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # --- 强制背景混合 (White Background Composition) ---
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
        
        # Optimizer Step Logic
        optimizer_points.step()
        scheduler_points.step()
        
        # --- 分阶段优化逻辑 ---
        if t >= warmup_iter:
            # 只有在预热期过后，才更新颜色
            optimizer_colors.step()
        
        # Clamp colors
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)
                
        if t % 20 == 0:
            current_lr = scheduler_points.get_last_lr()[0]
            phase = "Shape Only" if t < warmup_iter else "Shape + Color"
            print(f"    Iter {t} [{phase}]: Loss = {loss.item():.6f}, Points LR = {current_lr:.4f}")

            
    # 6. Save
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)

def main():
    parser = argparse.ArgumentParser(description="Batch process SVG using LIVE-inspired scheduling")
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
        # Exclude previous results
        svgs = [s for s in svgs if "fast" not in s.name and "external" not in s.name and "-m" not in s.name and "-v" not in s.name]
        pngs = list(d.glob("*.png"))
        
        if len(svgs) == 0 or len(pngs) == 0:
            print(f"Skipping {d}: Missing SVG or PNG")
            continue
            
        svg_file = svgs[0]
        png_file = pngs[0]
        output_name = f"{svg_file.stem}-external.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
