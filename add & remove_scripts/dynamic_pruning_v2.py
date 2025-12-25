import argparse
import torch
import pydiffvg
import os
import ttools.modules
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="SVG Vectorization with Dynamic Path Pruning V2 (Stroke Support)")
    parser.add_argument("target_path", help="Path to the target image")
    parser.add_argument("--num_paths", type=int, default=256, help="Initial number of paths")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--output_name", default=None, help="Output filename (without ext)")
    parser.add_argument("--prune_threshold", type=float, default=0.01, help="Opacity threshold for pruning")
    parser.add_argument("--prune_interval", type=int, default=50, help="Iterations between pruning checks")
    parser.add_argument("--init_svg", default=None, help="Path to initial SVG to load")
    parser.add_argument("--use_mse", action="store_true", help="Use MSE loss instead of LPIPS for speed")
    parser.add_argument("--point_lr", type=float, default=0.0, help="Learning rate for points (0.0 to freeze shapes)")
    parser.add_argument("--color_lr", type=float, default=0.01, help="Learning rate for colors")
    
    args = parser.parse_args()

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
    
    # 2. Load Target Image
    target = Image.open(args.target_path).convert("RGB")
    if target.size != (256, 256):
        print(f"Resizing target image from {target.size} to (256, 256) for speed...")
        target = target.resize((256, 256), Image.BICUBIC)
        
    canvas_width, canvas_height = target.size
    target_tensor = torch.from_numpy(np.array(target)).float() / 255.0
    target_tensor = target_tensor.to(device)
    target_permuted = target_tensor.permute(2, 0, 1).unsqueeze(0)

    # 3. Initialize Shapes
    shapes = []
    shape_groups = []
    
    if args.init_svg:
        print(f"Loading initial SVG from {args.init_svg}...")
        svg_w, svg_h, shapes, shape_groups = pydiffvg.svg_to_scene(args.init_svg)
        
        # Ensure device and gradients
        for shape in shapes:
            shape.points = shape.points.to(device)
            shape.stroke_width = shape.stroke_width.to(device)
            
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color = group.fill_color.to(device)
            if group.stroke_color is not None:
                group.stroke_color = group.stroke_color.to(device)
                
        args.num_paths = len(shapes)
    else:
        print("Error: init_svg is required for pruning mode.")
        return

    # 4. Optimizer Setup
    points_vars = []
    color_vars = []
    
    for path in shapes:
        path.points.requires_grad = (args.point_lr > 0)
        if args.point_lr > 0:
            points_vars.append(path.points)
            
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
        if group.stroke_color is not None:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
    
    # 构建优化器参数列表
    optim_params = []
    if len(color_vars) > 0:
        optim_params.append({"params": color_vars, "lr": args.color_lr})
    if len(points_vars) > 0:
        optim_params.append({"params": points_vars, "lr": args.point_lr})
        
    optimizer = torch.optim.Adam(optim_params)
    
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
    print(f"Start optimizing with {len(shapes)} paths (Point LR: {args.point_lr})...")
    
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
        if args.use_mse:
            loss = (img_permuted - target_permuted).pow(2).mean()
        else:
            lpips_loss = perception_loss(img_permuted, target_permuted)
            color_loss = (img_permuted - target_permuted).pow(2).mean()
            loss = lpips_loss + color_loss
        
        loss.backward()
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)
            
        if t % 20 == 0:
            print(f"Iter {t}: Loss = {loss.item():.4f}, Paths = {len(shapes)}")
            
        # --- Dynamic Pruning Logic V2 ---
        if t % args.prune_interval == 0 and t > 0:
            with torch.no_grad():
                indices_to_keep = []
                for idx, group in enumerate(shape_groups):
                    # Check visibility
                    is_visible = False
                    
                    # Check Fill Alpha (Index 3)
                    if group.fill_color is not None and group.fill_color[3] > args.prune_threshold:
                        is_visible = True
                        
                    # Check Stroke Alpha (Index 3)
                    if group.stroke_color is not None and group.stroke_color[3] > args.prune_threshold:
                        is_visible = True
                        
                    if is_visible:
                        indices_to_keep.append(idx)
                
                if len(indices_to_keep) < len(shapes):
                    print(f"Pruning: Removing {len(shapes) - len(indices_to_keep)} paths (Alpha < {args.prune_threshold})")
                    
                    new_shapes = [shapes[i] for i in indices_to_keep]
                    new_shape_groups = [shape_groups[i] for i in indices_to_keep]
                    
                    # Re-index shape_ids
                    for new_idx, group in enumerate(new_shape_groups):
                        group.shape_ids = torch.tensor([new_idx])
                        
                    shapes = new_shapes
                    shape_groups = new_shape_groups
                    
                    # Re-create optimizer
                    points_vars = []
                    color_vars = []
                    for path in shapes:
                        if args.point_lr > 0:
                            path.points.requires_grad = True
                            points_vars.append(path.points)
                    for group in shape_groups:
                        if group.fill_color is not None:
                            group.fill_color.requires_grad = True
                            color_vars.append(group.fill_color)
                        if group.stroke_color is not None:
                            group.stroke_color.requires_grad = True
                            color_vars.append(group.stroke_color)
                            
                    optim_params = []
                    if len(color_vars) > 0:
                        optim_params.append({"params": color_vars, "lr": args.color_lr})
                    if len(points_vars) > 0:
                        optim_params.append({"params": points_vars, "lr": args.point_lr})
                    
                    if optim_params:
                        optimizer = torch.optim.Adam(optim_params)
                    
    # Final Save
    pydiffvg.save_svg(
        f"output/{args.output_name}.svg",
        canvas_width, canvas_height, shapes, shape_groups
    )
    
    img_pil = Image.fromarray((img_rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    img_pil.save(f"output/{args.output_name}.png")
    
    print(f"Done. Final path count: {len(shapes)}")

if __name__ == "__main__":
    main()
