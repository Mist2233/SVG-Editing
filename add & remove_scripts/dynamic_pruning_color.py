import argparse
import torch
import pydiffvg
import os
import ttools.modules
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="SVG Vectorization with Color-Guided Pruning")
    parser.add_argument("target_path", help="Path to the target image")
    parser.add_argument("--num_paths", type=int, default=256, help="Initial number of paths")
    parser.add_argument("--num_iter", type=int, default=200, help="Number of optimization iterations")
    parser.add_argument("--output_name", default=None, help="Output filename (without ext)")
    parser.add_argument("--prune_threshold", type=float, default=0.01, help="Opacity threshold for pruning")
    parser.add_argument("--color_threshold", type=float, default=0.4, help="Color difference threshold for pruning")
    parser.add_argument("--prune_interval", type=int, default=50, help="Iterations between pruning checks")
    parser.add_argument("--init_svg", default=None, help="Path to initial SVG to load")
    parser.add_argument("--use_mse", action="store_true", help="Use MSE loss instead of LPIPS for speed")
    parser.add_argument("--max_paths", type=int, default=0, help="Max paths to keep from init_svg (0 = keep all)")
    
    args = parser.parse_args()

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
    
    # 2. Load Target Image
    target = Image.open(args.target_path).convert("RGBA")
    # Resize to 256x256 to ensure speed
    target_size = (256, 256)
    if target.size != target_size:
        print(f"Resizing target image from {target.size} to {target_size} for speed...")
        target = target.resize(target_size, Image.BICUBIC)
        
    canvas_width, canvas_height = target.size
    target_tensor = torch.from_numpy(np.array(target)).float() / 255.0
    target_tensor = target_tensor.to(device)
    # Ensure NCHW for LPIPS and Color Sampling
    target_permuted = target_tensor.permute(2, 0, 1).unsqueeze(0) # [1, 4, H, W]

    # 3. Initialize Shapes
    shapes = []
    shape_groups = []
    
    if args.init_svg:
        print(f"Loading initial SVG from {args.init_svg}...")
        svg_w, svg_h, shapes, shape_groups = pydiffvg.svg_to_scene(args.init_svg)
        
        # Scale shapes to match 256x256 canvas if necessary
        if svg_w != canvas_width or svg_h != canvas_height:
            print(f"Scaling SVG from {svg_w}x{svg_h} to {canvas_width}x{canvas_height}...")
            scale_x = canvas_width / svg_w
            scale_y = canvas_height / svg_h
            for path in shapes:
                path.points[:, 0] *= scale_x
                path.points[:, 1] *= scale_y
                path.stroke_width *= (scale_x + scale_y) / 2
        
        # Ensure device and gradients
        for shape in shapes:
            shape.points = shape.points.to(device)
            shape.stroke_width = shape.stroke_width.to(device)
            
        for group in shape_groups:
            group.fill_color = group.fill_color.to(device)
            
        args.num_paths = len(shapes) # Update count
        
    else:
        # Random Initialization (Fallback)
        for i in range(args.num_paths):
            num_segments = 3
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            
            p0 = (torch.rand(2) * 0.8 + 0.1) * torch.tensor([float(canvas_width), float(canvas_height)])
            p0 = p0.to(device)
            points.append(p0)
            
            radius = 5.0
            for j in range(num_segments):
                points.append(p0 + (torch.rand(2).to(device) - 0.5) * radius) 
                points.append(p0 + (torch.rand(2).to(device) - 0.5) * radius)
                points.append(p0 + (torch.rand(2).to(device) - 0.5) * radius)
                
            path = pydiffvg.Path(
                num_control_points=num_control_points,
                points=torch.stack(points),
                stroke_width=torch.tensor(1.0).to(device),
                is_closed=True
            )
            shapes.append(path)
            
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=torch.rand(4).to(device), # Random RGBA
            )
            shape_groups.append(path_group)

    # 4. Optimizer Setup
    points_vars = [path.points.requires_grad_(True) for path in shapes]
    color_vars = [group.fill_color.requires_grad_(True) for group in shape_groups]
    
    optimizer = torch.optim.Adam([
        {"params": points_vars, "lr": 1.0},
        {"params": color_vars, "lr": 0.01}
    ])
    
    # LPIPS Loss
    perception_loss = None
    if not args.use_mse:
        perception_loss = ttools.modules.LPIPS().to(device)

    # Output naming
    if args.output_name is None:
        base_name = os.path.splitext(os.path.basename(args.target_path))[0]
        args.output_name = f"{base_name}_{args.num_paths}path_{args.num_iter}iter_pruning"
    
    os.makedirs("output", exist_ok=True)
    
    # 5. Optimization Loop
    print(f"Start optimizing with {len(shapes)} paths...")
    
    for t in range(args.num_iter):
        optimizer.zero_grad()
        
        # Render
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        
        # Compose with white background
        bg = torch.ones(canvas_height, canvas_width, 3, device=device)
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * bg
        
        img_permuted = img_rgb.permute(2, 0, 1).unsqueeze(0)
        
        # Loss
        # Target has 4 channels now (RGBA), but loss usually expects RGB or RGBA matching.
        # For simplicity and compatibility with existing loss logic, let's compare against target RGB composed on white too.
        
        target_alpha = target_permuted[:, 3:4, :, :]
        target_rgb = target_permuted[:, :3, :, :]
        # bg needs to be permuted to match target shape [1, 3, H, W] for the multiplication
        bg_permuted = bg.permute(2, 0, 1).unsqueeze(0)
        
        target_composed = target_alpha * target_rgb + (1 - target_alpha) * bg_permuted
        
        if args.use_mse:
            loss = (img_permuted - target_composed).pow(2).mean()
        else:
            lpips_loss = perception_loss(img_permuted, target_composed)
            color_loss = (img_permuted - target_composed).pow(2).mean()
            loss = lpips_loss + color_loss
        
        loss.backward()
        
        # Gradient clipping/updating
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
            
        if t % 10 == 0:
            print(f"Iter {t}: Loss = {loss.item():.4f}, Paths = {len(shapes)}")
            
        # --- Color-Guided Pruning Logic ---
        if t % args.prune_interval == 0 and t > 0:
            with torch.no_grad():
                indices_to_keep = []
                pruned_count_opacity = 0
                pruned_count_color = 0
                
                for idx, group in enumerate(shape_groups):
                    should_keep = True
                    
                    # 1. Opacity Check
                    # Alpha channel is index 3
                    if group.fill_color[3] <= args.prune_threshold:
                        should_keep = False
                        pruned_count_opacity += 1
                    
                    # 2. Color & Alpha Consistency Check (if not already pruned)
                    if should_keep:
                        path = shapes[idx]
                        path_color = group.fill_color[:3]
                        path_alpha = group.fill_color[3]
                        
                        # Multi-point sampling strategy
                        # We sample all on-curve points
                        num_points = path.points.shape[0]
                        sample_indices = range(0, num_points, 3)
                        
                        has_match = False
                        valid_samples = 0
                        
                        for pt_idx in sample_indices:
                            p = path.points[pt_idx]
                            cx, cy = int(p[0].item()), int(p[1].item())
                            
                            # Boundary check
                            if 0 <= cx < canvas_width and 0 <= cy < canvas_height:
                                valid_samples += 1
                                
                                # Get Target Pixel (RGBA)
                                target_pixel = target_permuted[0, :, cy, cx]
                                target_color = target_pixel[:3]
                                target_alpha = target_pixel[3]
                                
                                # Logic:
                                # We only consider it a "match" if:
                                # 1. The target is NOT transparent (it's part of the object)
                                # 2. The color matches
                                
                                if target_alpha > 0.1: # Target is solid
                                    # Calculate Color Difference
                                    color_diff = torch.norm(target_color - path_color).item()
                                    
                                    # Any-match policy: if ANY point matches, we keep the path.
                                    if color_diff <= args.color_threshold:
                                        has_match = True
                                        break # Found a match, this path is valid!
                                # If target_alpha is low (background), this point cannot save the path.
                        
                        # If we had valid samples but NO match found among them -> Prune
                        # This handles:
                        # 1. Path is in background (target_alpha always low -> has_match False)
                        # 2. Path is on object but wrong color (color_diff high -> has_match False)
                        if valid_samples > 0 and not has_match:
                            should_keep = False
                            pruned_count_color += 1

                    if should_keep:
                        indices_to_keep.append(idx)
                
                if len(indices_to_keep) < len(shapes):
                    total_pruned = len(shapes) - len(indices_to_keep)
                    print(f"Pruning: Removing {total_pruned} paths (Opacity: {pruned_count_opacity}, Color: {pruned_count_color})")
                    
                    new_shapes = [shapes[i] for i in indices_to_keep]
                    new_shape_groups = [shape_groups[i] for i in indices_to_keep]
                    
                    # Re-index shape_ids in groups
                    for new_idx, group in enumerate(new_shape_groups):
                        group.shape_ids = torch.tensor([new_idx])
                        
                    shapes = new_shapes
                    shape_groups = new_shape_groups
                    
                    # Re-create optimizer
                    if len(shapes) > 0:
                        points_vars = [path.points.requires_grad_(True) for path in shapes]
                        color_vars = [group.fill_color.requires_grad_(True) for group in shape_groups]
                        optimizer = torch.optim.Adam([
                            {"params": points_vars, "lr": 1.0},
                            {"params": color_vars, "lr": 0.01}
                        ])
                    else:
                        print("Warning: All paths pruned! Stopping optimization.")
                        break
                    
    # Final Save
    pydiffvg.save_svg(
        f"output/{args.output_name}.svg",
        canvas_width, canvas_height, shapes, shape_groups
    )
    
    # Save preview PNG
    img_pil = Image.fromarray((img_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    img_pil.save(f"output/{args.output_name}.png")
    
    print(f"Done. Final path count: {len(shapes)}")

if __name__ == "__main__":
    main()
