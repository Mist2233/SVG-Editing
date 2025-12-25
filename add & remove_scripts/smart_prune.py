import argparse
import os
import torch
import pydiffvg
import numpy as np
from PIL import Image
from svgpathtools import svg2paths, wsvg
import sys
import math

def calculate_compactness(path):
    try:
        if not path.isclosed():
            return 0.0, 0.0
        length = path.length()
        if length == 0: return 0.0, 0.0
        area = abs(path.area())
        # Isoperimetric Quotient: 4 * pi * Area / Perimeter^2
        # Circle = 1.0, Square = 0.78, Thin line -> 0.0
        compactness = (4 * math.pi * area) / (length ** 2)
        return compactness, area
    except:
        return 0.0, 0.0

def main():
    parser = argparse.ArgumentParser(description="Smart Pruning with Color & Compactness Awareness")
    parser.add_argument("input_svg", help="Input SVG path")
    parser.add_argument("target_path", help="Target Image path")
    parser.add_argument("output_svg", help="Output SVG path")
    parser.add_argument("--image_size", type=int, default=256, help="Render resolution")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)

    print(f"Loading Target: {args.target_path}")
    target = Image.open(args.target_path).convert("RGBA")
    target = target.resize((args.image_size, args.image_size), Image.BICUBIC)
    target_np = np.array(target).astype(np.float32) / 255.0
    target_tensor = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Pre-calculate target composed (on white)
    bg = torch.ones(1, 3, args.image_size, args.image_size).to(device)
    target_alpha = target_tensor[:, 3:4, :, :]
    target_rgb = target_tensor[:, :3, :, :]
    target_composed = target_alpha * target_rgb + (1 - target_alpha) * bg

    # 1. Explode Paths
    print(f"Exploding SVG: {args.input_svg}")
    paths, attributes = svg2paths(args.input_svg)
    exploded_paths = []
    exploded_attrs = []
    
    for path, attr in zip(paths, attributes):
        subpaths = path.continuous_subpaths()
        if len(subpaths) > 1:
            for sub in subpaths:
                exploded_paths.append(sub)
                exploded_attrs.append(attr.copy())
        else:
            exploded_paths.append(path)
            exploded_attrs.append(attr)
            
    print(f"Exploded from {len(paths)} to {len(exploded_paths)} paths.")
    
    # Save Temp Exploded
    temp_exploded = "temp_exploded.svg"
    wsvg(exploded_paths, attributes=exploded_attrs, filename=temp_exploded)

    # Load SVG paths for geometric analysis
    print(f"Loading SVG geometry from: {temp_exploded}")
    svg_paths, _ = svg2paths(temp_exploded)
    shape_metrics = [calculate_compactness(p) for p in svg_paths]
    print(f"Metrics (Compactness, Area): {['({:.2f}, {:.1f})'.format(c, a) for c, a in shape_metrics]}")
    
    # 2. Load into PyDiffVG
    print("Loading into Differentiable Renderer...")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(temp_exploded)
    
    # Check consistency
    if len(shapes) != len(exploded_paths):
        print(f"Warning: Renderer found {len(shapes)} shapes, but exploded has {len(exploded_paths)}.")
        print("Alignment might be lost. Proceeding with Renderer's indices.")
    
    # Scale shapes
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

    def compute_loss(curr_shapes, curr_groups):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            args.image_size, args.image_size, curr_shapes, curr_groups
        )
        img = pydiffvg.RenderFunction.apply(
            args.image_size, args.image_size, 2, 2, 0, None, *scene_args
        )
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * torch.ones(args.image_size, args.image_size, 3, device=device)
        img_permuted = img_rgb.permute(2, 0, 1).unsqueeze(0)
        return (img_permuted - target_composed).pow(2).mean().item()

    # 3. Iterative Pruning
    # Active indices (indices in the `shapes` list)
    active_indices = list(range(len(shapes)))
    
    # Map shape_index to group object
    shape_to_group = {}
    for g in shape_groups:
        for sid in g.shape_ids:
            shape_to_group[int(sid)] = g

    iteration = 0
    while True:
        iteration += 1
        baseline_loss = compute_loss(shapes, shape_groups)
        print(f"Iter {iteration}: Baseline Loss = {baseline_loss:.6f} | Active Shapes: {len(active_indices)}")
        
        best_delta = 0
        worst_shape_idx = -1
        
        # Check all active shapes
        for i in active_indices:
            # 1. Color Consistency Check (Local Protection)
            # If the shape color matches the target image color at ANY of its key points, PROTECT IT.
            path = shapes[i]
            if path.points.shape[0] == 0: continue
            
            group = shape_to_group[i]
            shape_color = group.fill_color[:3] # [3]
            
            # Robust Sampling: Check multiple points
            points = path.points
            num_points = points.shape[0]
            
            # Sample up to 20 points for better statistical significance
            if num_points > 20:
                indices = torch.linspace(0, num_points-1, 20).long()
                samples = points[indices]
            else:
                samples = points
                
            match_count = 0
            total_samples = len(samples)
            
            for pt in samples:
                cx = torch.clamp(pt[0].long(), 0, args.image_size - 1)
                cy = torch.clamp(pt[1].long(), 0, args.image_size - 1)
                
                target_color = target_composed[0, :, cy, cx]
                dist = torch.norm(target_color - shape_color).item()
                
                # Threshold: 0.5 (Strict match)
                if dist < 0.6: # Relaxed slightly from 0.5 to 0.6
                    match_count += 1
            
            match_ratio = match_count / total_samples
            
            # CONSENSUS VOTING:
            # We set the threshold strictly at 0.15.
            # Rationale: Any shape with Match Ratio <= 0.15 is considered "Foreign Object" (Noise).
            # Any shape with Match Ratio > 0.15 is considered "Part of Image" (Essential).
            if match_ratio > 0.15:
                # print(f"  -> Shape {i} Protected (Match Ratio: {match_ratio:.2f})")
                continue
            
            # COMPACTNESS & AREA CHECK:
            # Strategy:
            # 1. Protect BIG structural parts (Area > 500).
            #    Example: Pizza Crust (4900), Fish Fin (2200), Fish Body (25000).
            # 2. Protect NATURAL shapes (0.35 < Compactness < 0.65).
            #    Example: Fish Fin (0.51).
            # 3. Remove ARTIFACTS (Low Compactness) and GEOMETRIC DECORATIONS (High Compactness).
            #    Example: Fish Mouth (0.20, Area 383) -> REMOVE.
            #    Example: Pizza Pepperoni (0.97, Area 171) -> REMOVE.
            #    Example: Pizza Smile (0.69, Area 87) -> REMOVE.
            
            compactness, area = shape_metrics[i] if i < len(shape_metrics) else (0.0, 0.0)
            
            if area > 500:
                print(f"  -> Shape {i} Protected (Area: {area:.1f} > 500)")
                continue
                
            if 0.35 < compactness < 0.65:
                print(f"  -> Shape {i} Protected (Compactness: {compactness:.2f} in [0.35, 0.65])")
                continue

            # 2. MSE Impact Check
            # We only prune if the Delta is SIGNIFICANTLY negative.
            # Shape 8 (Mouth) Delta: -0.001954
            # Shape 2 (Noise) Delta: -0.005784
            # Shape 0 (Noise) Delta: -0.000703 (Very small impact)
            # Shape 9 (Noise) Delta: -0.000501 (Very small impact)
            
            # To be conservative and avoid over-pruning small details (like Shape 0/9),
            # we increase the prune threshold.
            # Only remove if it improves loss by at least 0.001.
            PRUNE_THRESHOLD = -0.001
            original_ids = group.shape_ids.clone()
            
            # Temporarily remove
            new_ids = [sid for sid in original_ids if int(sid) != i]
            
            if len(new_ids) == 0:
                 # Group becomes empty
                 original_color = group.fill_color.clone()
                 group.fill_color[3] = 0.0
                 loss = compute_loss(shapes, shape_groups)
                 group.fill_color = original_color # Restore
            else:
                group.shape_ids = torch.tensor(new_ids, dtype=torch.long)
                loss = compute_loss(shapes, shape_groups)
                group.shape_ids = original_ids # Restore
                
            delta = loss - baseline_loss
            
            # We want the most negative delta (biggest improvement)
            if delta < best_delta:
                best_delta = delta
                worst_shape_idx = i
        
        # Apply Prune Threshold
        if worst_shape_idx != -1 and best_delta < PRUNE_THRESHOLD:
            print(f"  -> Found Harmful Shape {worst_shape_idx} (Delta: {best_delta:.6f}, Match Ratio < 0.15, Unprotected Geometry). Removing.")
            
            # Permanently remove from pydiffvg scene
            active_indices.remove(worst_shape_idx)
            group = shape_to_group[worst_shape_idx]
            new_ids = [sid for sid in group.shape_ids if int(sid) != worst_shape_idx]
            
            if len(new_ids) == 0:
                # Group is empty, remove it from shape_groups list to prevent renderer issues
                if group in shape_groups:
                    shape_groups.remove(group)
            else:
                group.shape_ids = torch.tensor(new_ids, dtype=torch.long)
        else:
            print("  -> No more harmful shapes found.")
            break
            
    # 4. Save Final
    # Filter exploded_paths based on active_indices
    final_paths = []
    final_attrs = []
    
    if len(shapes) == len(exploded_paths):
        # We need to preserve the original order of indices
        # active_indices is a list of integers, not necessarily sorted
        # But we want to iterate 0..N and check if i is in active_indices
        active_set = set(active_indices)
        
        for i in range(len(exploded_paths)):
            if i in active_set:
                final_paths.append(exploded_paths[i])
                final_attrs.append(exploded_attrs[i])
        
        wsvg(final_paths, attributes=final_attrs, filename=args.output_svg)
        print(f"Saved optimized SVG to: {args.output_svg}")
        
        # Save Final PNG
        png_path = args.output_svg.replace(".svg", ".png")
        print(f"Rendering final PNG to: {png_path}")
        
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            args.image_size, args.image_size, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            args.image_size, args.image_size, 2, 2, 0, None, *scene_args
        )
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * torch.ones(args.image_size, args.image_size, 3, device=device)
        img_pil = Image.fromarray((img_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
        img_pil.save(png_path)
        
    else:
        print("Error: Index mismatch preventing safe save. Check temp_exploded.svg manually.")

    # Cleanup
    if os.path.exists(temp_exploded):
        os.remove(temp_exploded)

if __name__ == "__main__":
    main()
