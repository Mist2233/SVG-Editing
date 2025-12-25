import argparse
import torch
import pydiffvg
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import math

def calculate_compactness(path):
    try:
        if not path.isclosed():
            return 0.0
        length = path.length()
        if length == 0: return 0.0
        area = abs(path.area())
        return (4 * math.pi * area) / (length ** 2)
    except:
        return 0.0

def main():
    parser = argparse.ArgumentParser(description="Diagnostic Tool for SVG Pruning")
    parser.add_argument("target_path", help="Path to the target image")
    parser.add_argument("svg_path", help="Path to the SVG to analyze")
    parser.add_argument("--output_dir", default="diagnostic_output", help="Directory to save debug images")
    parser.add_argument("--image_size", type=int, default=256, help="Render resolution")
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)

    # 2. Load Target
    print(f"Loading Target: {args.target_path}")
    target = Image.open(args.target_path).convert("RGBA")
    target = target.resize((args.image_size, args.image_size), Image.BICUBIC)
    
    # Convert to Tensor (N, 4, H, W)
    target_np = np.array(target).astype(np.float32) / 255.0
    target_tensor = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(device) # [1, 4, H, W]
    
    # Compose Target on White for Loss Calculation
    bg = torch.ones(1, 3, args.image_size, args.image_size).to(device)
    target_alpha = target_tensor[:, 3:4, :, :]
    target_rgb = target_tensor[:, :3, :, :]
    target_composed = target_alpha * target_rgb + (1 - target_alpha) * bg
    
    # 3. Load SVG
    print(f"Loading SVG: {args.svg_path}")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(args.svg_path)
    
    # Scale to Image Size
    scale_x = args.image_size / canvas_width
    scale_y = args.image_size / canvas_height
    for path in shapes:
        path.points[:, 0] *= scale_x
        path.points[:, 1] *= scale_y
        path.stroke_width *= (scale_x + scale_y) / 2
        path.points = path.points.to(device)
        path.stroke_width = path.stroke_width.to(device)
        
    for group in shape_groups:
        group.fill_color = group.fill_color.to(device)

    # 4. Helper Function: Render & Compute Loss
    def compute_loss(current_shapes, current_groups):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            args.image_size, args.image_size, current_shapes, current_groups
        )
        img = pydiffvg.RenderFunction.apply(
            args.image_size, args.image_size, 2, 2, 0, None, *scene_args
        )
        
        # Compose Render on White
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * torch.ones(args.image_size, args.image_size, 3, device=device)
        img_permuted = img_rgb.permute(2, 0, 1).unsqueeze(0)
        
        # MSE Loss
        loss = (img_permuted - target_composed).pow(2).mean()
        return loss.item()

    # 5. Calculate Baseline Loss
    baseline_loss = compute_loss(shapes, shape_groups)
    print(f"\nBaseline Loss (All {len(shapes)} paths): {baseline_loss:.6f}")
    
    # 6. Rank Analysis (Leave-One-Out) - SHAPE LEVEL
    print(f"\n{'='*80}")
    print(f"{'Shape ID':<10} | {'Group ID':<10} | {'Color (RGB)':<20} | {'Loss Delta':<15} | {'Verdict'}")
    print(f"{'-'*80}")
    
    scores = []
    
    # Prepare for Visualization
    # Render full image first
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        args.image_size, args.image_size, shapes, shape_groups
    )
    img = pydiffvg.RenderFunction.apply(
        args.image_size, args.image_size, 2, 2, 0, None, *scene_args
    )
    img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * torch.ones(args.image_size, args.image_size, 3, device=device)
    img_pil = Image.fromarray((img_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = None # Default font

    # Map shape_id to group object (not tuple)
    shape_to_group_obj = {}
    for g_idx, group in enumerate(shape_groups):
        for s_idx in group.shape_ids:
            shape_to_group_obj[int(s_idx)] = group

    for i in range(len(shapes)):
        # 1. Basic Info
        if i not in shape_to_group_obj:
            continue
            
        group = shape_to_group_obj[i]
        # Find g_idx for display
        g_idx = -1
        for k, g in enumerate(shape_groups):
            if g is group:
                g_idx = k
                break
                
        original_color = group.fill_color.clone()
        color_str = f"[{original_color[0]:.2f}, {original_color[1]:.2f}, {original_color[2]:.2f}]"

        # 2. Visualization
        path = shapes[i]
        if path.points.shape[0] > 0:
            center = path.points.mean(dim=0).cpu().numpy()
            x, y = center[0], center[1]
            draw.text((x, y), str(i), font=font, fill=(255,0,0))

        # 3. Leave-One-Out Analysis (Shape Level)
        # We need to temporarily remove this shape from its group
        original_shape_ids = group.shape_ids.clone()
        
        # Remove i from shape_ids
        new_ids = [sid for sid in original_shape_ids if int(sid) != i]
        
        if len(new_ids) == 0:
            # If group becomes empty, exclude this group from rendering entirely
            # Create a temporary list of groups excluding the current one
            temp_groups = [g for k, g in enumerate(shape_groups) if k != g_idx]
            new_loss = compute_loss(shapes, temp_groups)
        else:
            # Update group shape_ids with correct dtype
            group.shape_ids = torch.tensor(new_ids, dtype=torch.long)
            new_loss = compute_loss(shapes, shape_groups)
            # Restore
            group.shape_ids = original_shape_ids
        
        delta = new_loss - baseline_loss
        
        verdict = "KEEP (Essential)" if delta > 0 else "REMOVE (Harmful)"
        if abs(delta) < 1e-6: verdict = "NEUTRAL (Useless)"
        
        print(f"{i:<10} | {g_idx:<10} | {color_str:<20} | {delta:<15.6f} | {verdict}")
        
        scores.append({
            "id": i,
            "group_id": g_idx,
            "delta": delta,
            "verdict": verdict
        })

    # Save Diagnostic Map
    map_path = os.path.join(args.output_dir, "diagnostic_map_shapes.png")
    img_pil.save(map_path)
    print(f"\nShape Diagnostic Map saved to: {map_path}")

    # 7. Recommendation
    print(f"\n{'='*80}")
    print("Recommendation (Sorted by Harmfulness - Most harmful first):")
    
    scores.sort(key=lambda x: x["delta"])
    
    for item in scores:
        if item["delta"] < 0:
            print(f"Shape {item['id']} (Group {item['group_id']}): Improves loss by {-item['delta']:.6f} if removed. -> SUGGEST REMOVE")
        elif item["delta"] == 0:
            print(f"Shape {item['id']} (Group {item['group_id']}): No effect. -> OPTIONAL REMOVE")
            
    print(f"{'='*80}")
    
    # 8. Sampling Analysis
    print(f"\n[Sampling Analysis for Shapes]")
    for i in range(len(shapes)):
        path = shapes[i]
        if path.points.shape[0] == 0: continue
        
        if i not in shape_to_group_obj: continue
        group = shape_to_group_obj[i]
        
        shape_color = group.fill_color[:3]
        
        points = path.points
        num_points = points.shape[0]
        
        if num_points > 20:
            indices = torch.linspace(0, num_points-1, 20).long()
            samples = points[indices]
        else:
            samples = points
            
        match_count = 0
        total_samples = len(samples)
        
        print(f"Shape {i}:")
        for j, pt in enumerate(samples):
            cx = torch.clamp(pt[0].long(), 0, args.image_size - 1)
            cy = torch.clamp(pt[1].long(), 0, args.image_size - 1)
            target_color = target_composed[0, :, cy, cx]
            dist = torch.norm(target_color - shape_color).item()
            is_match = dist < 0.6
            if is_match: match_count += 1
            # print(f"  Pt {j}: Dist={dist:.4f} {'[MATCH]' if is_match else ''}")
            
        ratio = match_count / total_samples
        
        # Calculate Compactness
        compactness = calculate_compactness(path)
        
        print(f"  -> Match Ratio: {ratio:.2f} ({match_count}/{total_samples}) | Compactness: {compactness:.2f}")

if __name__ == "__main__":
    main()
