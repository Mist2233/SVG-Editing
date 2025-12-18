import argparse
import torch
import pydiffvg
import os
import ttools.modules
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="SVG Vectorization with Dynamic Path Pruning")
    parser.add_argument("target_path", help="Path to the target image")
    parser.add_argument("--num_paths", type=int, default=256, help="Initial number of paths")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--output_name", default=None, help="Output filename (without ext)")
    parser.add_argument("--prune_threshold", type=float, default=0.01, help="Opacity threshold for pruning")
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
    target = Image.open(args.target_path).convert("RGB")
    # Resize to 256x256 to ensure speed
    if target.size != (256, 256):
        print(f"Resizing target image from {target.size} to (256, 256) for speed...")
        target = target.resize((256, 256), Image.BICUBIC)
        
    canvas_width, canvas_height = target.size
    target_tensor = torch.from_numpy(np.array(target)).float() / 255.0
    target_tensor = target_tensor.to(device)
    # Ensure NCHW for LPIPS
    target_permuted = target_tensor.permute(2, 0, 1).unsqueeze(0)

    # 3. Initialize Shapes
    shapes = []
    shape_groups = []
    
    if args.init_svg:
        print(f"Loading initial SVG from {args.init_svg}...")
        svg_w, svg_h, shapes, shape_groups = pydiffvg.svg_to_scene(args.init_svg)
        
        # NOTE: Removed automatic scaling logic as requested. 
        # Assuming init_svg matches 256x256 target.
        
        # Ensure device and gradients
        for shape in shapes:
            shape.points = shape.points.to(device)
            shape.stroke_width = shape.stroke_width.to(device)
            
        for group in shape_groups:
            group.fill_color = group.fill_color.to(device)
            
        # NOTE: Removed mandatory subsampling to preserve high quality from init_svg
        # if args.max_paths > 0 and len(shapes) > args.max_paths:
        #     print(f"Subsampling paths from {len(shapes)} to {args.max_paths}...")
        #     # Use stride slicing to preserve overall structure
        #     stride = len(shapes) // args.max_paths
        #     indices = list(range(0, len(shapes), stride))[:args.max_paths]
        #     
        #     shapes = [shapes[i] for i in indices]
        #     shape_groups = [shape_groups[i] for i in indices]
        #     
        #     # Re-index
        #     for new_idx, group in enumerate(shape_groups):
        #         group.shape_ids = torch.tensor([new_idx])
                
        args.num_paths = len(shapes) # Update count
        
    else:
        # Random Initialization
        for i in range(args.num_paths):
            # Use 3 segments (Cubic Bezier) -> needs 1 + 3*3 = 10 points
            num_segments = 3
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            
            # Keep points within canvas
            p0 = (torch.rand(2) * 0.8 + 0.1) * torch.tensor([float(canvas_width), float(canvas_height)])
            p0 = p0.to(device)
            points.append(p0)
            
            # Reduced perturbation to 5.0 to avoid wild paths
            radius = 5.0
            for j in range(num_segments):
                # Cubic Bezier needs 3 points per segment (2 control + 1 end)
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
        
        # LPIPS Input format: NCHW, range [-1, 1] usually preferred but [0, 1] ok for some impls
        # ttools LPIPS expects NCHW
        img_permuted = img_rgb.permute(2, 0, 1).unsqueeze(0)
        
        # Loss
        if args.use_mse:
            # Simple MSE
            loss = (img_permuted - target_permuted).pow(2).mean()
        else:
            lpips_loss = perception_loss(img_permuted, target_permuted)
            # Add color loss for stability
            color_loss = (img_permuted - target_permuted).pow(2).mean()
            loss = lpips_loss + color_loss
        
        loss.backward()
        
        # Gradient clipping/updating
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
            
        if t % 10 == 0:
            print(f"Iter {t}: Loss = {loss.item():.4f}, Paths = {len(shapes)}")
            
        # --- Dynamic Pruning Logic ---
        if t % args.prune_interval == 0 and t > 0:
            with torch.no_grad():
                indices_to_keep = []
                for idx, group in enumerate(shape_groups):
                    # Check opacity (alpha channel is index 3)
                    if group.fill_color[3] > args.prune_threshold:
                        indices_to_keep.append(idx)
                
                if len(indices_to_keep) < len(shapes):
                    print(f"Pruning: Removing {len(shapes) - len(indices_to_keep)} paths with alpha < {args.prune_threshold}")
                    
                    new_shapes = [shapes[i] for i in indices_to_keep]
                    new_shape_groups = [shape_groups[i] for i in indices_to_keep]
                    
                    # Re-index shape_ids in groups
                    for new_idx, group in enumerate(new_shape_groups):
                        group.shape_ids = torch.tensor([new_idx])
                        
                    shapes = new_shapes
                    shape_groups = new_shape_groups
                    
                    # Re-create optimizer (tricky part: need to preserve momentum? usually easier to restart or filter param groups)
                    # Simple approach: Re-init optimizer. Momentum loss is acceptable for pruning step.
                    points_vars = [path.points.requires_grad_(True) for path in shapes]
                    color_vars = [group.fill_color.requires_grad_(True) for group in shape_groups]
                    optimizer = torch.optim.Adam([
                        {"params": points_vars, "lr": 1.0},
                        {"params": color_vars, "lr": 0.01}
                    ])
                    # Note: We lose optimizer state here. For advanced impl, we should modify param_groups in place.
                    
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
