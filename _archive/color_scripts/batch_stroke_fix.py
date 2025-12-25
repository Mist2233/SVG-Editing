import pydiffvg
import torch
import skimage.io
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

# 强制 stdout 实时输出
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

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

    # Move to GPU & PRE-OPTIMIZATION FIXES
    print("  [Init] Applying 'Stroke Explosion' Fixes...")
    
    for path in shapes:
        path.points = path.points.to(device)
        
        # --- FIX 1: Reset Stroke Width to 1.0 ---
        # 强制重置描边宽度，防止起手就是粗线条
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width.data.fill_(1.0)
        else:
            path.stroke_width = torch.tensor(1.0).to(device)
        
        # 确保 stroke_width 在 GPU 上
        if not isinstance(path.stroke_width, torch.Tensor):
             path.stroke_width = torch.tensor(path.stroke_width).to(device)
        elif path.stroke_width.device != device:
             path.stroke_width = path.stroke_width.to(device)

    for group in shape_groups:
        # --- FIX 2: Randomize Colors ---
        # 打乱颜色，杀死“紫色蒙版”的初始状态
        if group.fill_color is not None:
            # 随机颜色 (RGB) + 保持原有 Alpha (或者设为 1.0)
            original_alpha = group.fill_color.data[3].clone()
            group.fill_color.data[:3] = torch.rand(3).to(device)
            group.fill_color.data[3] = 1.0 # 强制不透明，配合白底混合
            group.fill_color = group.fill_color.to(device)
            
        if group.stroke_color is not None:
            original_alpha = group.stroke_color.data[3].clone()
            group.stroke_color.data[:3] = torch.rand(3).to(device)
            group.stroke_color.data[3] = 1.0
            group.stroke_color = group.stroke_color.to(device)

    # 4. Optimizer Setup
    points_vars = []
    color_vars = []
    stroke_vars = [] # 新增：优化描边宽度
    
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        # 允许优化描边宽度，但会在循环中严格限制它
        path.stroke_width.requires_grad = True
        stroke_vars.append(path.stroke_width)
        
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
        if group.stroke_color is not None:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
            
    # Optimizers
    optimizer_points = torch.optim.Adam(points_vars, lr=1.0)
    optimizer_colors = torch.optim.Adam(color_vars, lr=0.01)
    optimizer_strokes = torch.optim.Adam(stroke_vars, lr=0.1) # 描边宽度也可以优化

    # Scheduler for Points
    lambda_points = lambda epoch: 1.0 - 0.9 * (epoch / num_iter)
    scheduler_points = LambdaLR(optimizer_points, lr_lambda=lambda_points)

    # Background for composition
    background_color = torch.tensor([1.0, 1.0, 1.0]).to(device)
    
    # Staged Optimization
    warmup_iter = int(num_iter * 0.3) 

    # 5. Loop
    print(f"--> Start Stroke-Fix Refine ({num_iter} iterations)...")
    print(f"    Phase 1: Shape & Stroke Adjustment (Iter 0-{warmup_iter}) - Colors Frozen")
    print(f"    Phase 2: Full Refinement (Iter {warmup_iter}-{num_iter})")

    for t in range(num_iter):
        optimizer_points.zero_grad()
        optimizer_colors.zero_grad()
        optimizer_strokes.zero_grad()
        
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
        
        # Step
        optimizer_points.step()
        optimizer_strokes.step() # 始终优化描边
        scheduler_points.step()
        
        if t >= warmup_iter:
            optimizer_colors.step()
        
        # --- CLAMPS ---
        # 1. Clamp Colors
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)
        
        # 2. --- FIX 3: Clamp Stroke Widths (Strict!) ---
        for path in shapes:
            # 严格限制在 0.5 到 3.0 之间
            # 这就是防止“紫色蒙版”复发的紧箍咒
            path.stroke_width.data.clamp_(0.5, 3.0) 
                
        if t % 20 == 0:
            current_lr = scheduler_points.get_last_lr()[0]
            phase = "Shape/Stroke" if t < warmup_iter else "Full"
            # 打印一下当前的平均描边宽度，看看是否受控
            avg_stroke = torch.mean(torch.cat([s.view(-1) for s in stroke_vars])).item()
            print(f"    Iter {t} [{phase}]: Loss = {loss.item():.6f}, Avg Stroke = {avg_stroke:.2f}")
            
    # 6. Save
    pydiffvg.save_svg(output_path, canvas_width, canvas_height, shapes, shape_groups)

def main():
    parser = argparse.ArgumentParser(description="Batch process SVG with Stroke Explosion Fixes")
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
        # 排除之前生成的中间文件，只找原始的或者特定的输入
        # 这里假设我们要处理的是 v3 或者原始文件，为了保险起见，我们优先处理名字最短的 SVG（通常是原文件）或者带有 v3 的
        # 但为了避免混淆，我们尽量找不带 -external, -fast, -fix 后缀的
        valid_svgs = [s for s in svgs if "external" not in s.name and "fast" not in s.name and "fix" not in s.name]
        
        # 如果找不到纯净的，就找 v3
        if not valid_svgs:
             valid_svgs = [s for s in svgs if "v3" in s.name]
             
        pngs = list(d.glob("*.png"))
        
        if not valid_svgs or not pngs:
            print(f"Skipping {d}: Missing suitable SVG or PNG")
            continue
            
        svg_file = valid_svgs[0] # Take the first valid one
        png_file = pngs[0]
        output_name = f"{svg_file.stem}-strokefix.svg"
        output_path = d / output_name
        
        try:
            optimize_pair(str(svg_file), str(png_file), str(output_path), num_iter=args.iter)
        except Exception as e:
            print(f"Error processing {d}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
