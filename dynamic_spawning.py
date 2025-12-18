import argparse
import torch
import pydiffvg
import os
import ttools.modules
from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="SVG Vectorization with Dynamic Path Spawning")
    parser.add_argument("target_path", help="Path to the target image")
    parser.add_argument("--num_paths", type=int, default=128, help="Initial number of paths (start small)")
    parser.add_argument("--max_paths", type=int, default=512, help="Maximum allowed paths")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--output_name", default=None, help="Output filename (without ext)")
    parser.add_argument("--spawn_interval", type=int, default=100, help="Iterations between spawning checks")
    parser.add_argument("--init_svg", default=None, help="Path to initial SVG to load")
    parser.add_argument("--use_mse", action="store_true", help="Use MSE loss instead of LPIPS for speed")
    
    args = parser.parse_args()

    # 1. Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    if torch.cuda.is_available():
        pydiffvg.set_device(device)
    
    # 2. Load Target Image
    target = Image.open(args.target_path).convert("RGB")
    
    # Resize to 256x256 to ensure speed (consistent with pruning script)
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
        
        # Ensure device and gradients
        for shape in shapes:
            shape.points = shape.points.to(device)
            shape.stroke_width = shape.stroke_width.to(device)
            
        for group in shape_groups:
            group.fill_color = group.fill_color.to(device)
            
        args.num_paths = len(shapes) # Update count based on loaded SVG
    
    # Helper for random path creation
    def create_random_path(center_x=None, center_y=None, scale=5.0):
        # Use 3 segments (Cubic Bezier) -> needs 1 + 3*3 = 10 points
        num_segments = 3
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        if center_x is None:
            center_x = torch.rand(1) * float(canvas_width)
            center_y = torch.rand(1) * float(canvas_height)
        
        p0 = torch.tensor([float(center_x), float(center_y)]).to(device)
        points.append(p0)
        for j in range(num_segments):
            points.append(p0 + (torch.rand(2).to(device) - 0.5) * scale)
            points.append(p0 + (torch.rand(2).to(device) - 0.5) * scale)
            points.append(p0 + (torch.rand(2).to(device) - 0.5) * scale)
            
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=torch.stack(points),
            stroke_width=torch.tensor(1.0).to(device),
            is_closed=True
        )
        return path

    if not args.init_svg:
        # Only initialize random paths if no SVG loaded
        for i in range(args.num_paths):
            path = create_random_path()
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=torch.rand(4).to(device),
            )
            shape_groups.append(path_group)

    # 4. Optimizer Setup
    # Helper to rebuild optimizer
    def get_optimizer(shapes, shape_groups):
        points_vars = [path.points.requires_grad_(True) for path in shapes]
        color_vars = [group.fill_color.requires_grad_(True) for group in shape_groups]
        return torch.optim.Adam([
            {"params": points_vars, "lr": 1.0},
            {"params": color_vars, "lr": 0.01}
        ])

    optimizer = get_optimizer(shapes, shape_groups)
    
    # LPIPS Loss
    perception_loss = None
    if not args.use_mse:
        perception_loss = ttools.modules.LPIPS().to(device)

    # Output naming
    if args.output_name is None:
        base_name = os.path.splitext(os.path.basename(args.target_path))[0]
        args.output_name = f"{base_name}_{args.num_iter}iter_spawning"
    
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
        if args.use_mse:
            loss = (img_permuted - target_permuted).pow(2).mean()
        else:
            lpips_loss = perception_loss(img_permuted, target_permuted)
            color_loss = (img_permuted - target_permuted).pow(2).mean()
            loss = lpips_loss + color_loss
        
        # Calculate pixel-wise error for spawning map
        with torch.no_grad():
            diff = (img_rgb - target_tensor).abs().mean(dim=2) # [H, W] error map
        
        loss.backward()
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)
            
        if t % 10 == 0:
            print(f"Iter {t}: Loss = {loss.item():.4f}, Paths = {len(shapes)}")
            
        # --- Dynamic Spawning Logic ---
        if t % args.spawn_interval == 0 and t > 0 and len(shapes) < args.max_paths:
            # Find region with max error
            # Smooth error map slightly to find robust peaks? Optional.
            # Simple approach: Find pixel with max error
            # To avoid single pixel noise, maybe pool?
            # Let's pick Top-K worst pixels or random sampling weighted by error.
            
            # Convert error map to probability distribution
            error_flat = diff.view(-1)
            error_prob = error_flat / error_flat.sum()
            
            # Sample 5 new locations based on error
            # multinomial expects 2D? No, 1D is fine for 1 sample batch
            num_new = min(5, args.max_paths - len(shapes))
            if num_new > 0:
                indices = torch.multinomial(error_flat, num_new)
                
                print(f"Spawning {num_new} new paths at high error regions...")
                
                for idx in indices:
                    y = (idx // canvas_width).item()
                    x = (idx % canvas_width).item()
                    
                    # Create new small path at this location
                    new_path = create_random_path(center_x=x, center_y=y, scale=20) # Start smaller
                    new_path.points = new_path.points.to(device)
                    new_path.stroke_width = new_path.stroke_width.to(device)
                    shapes.append(new_path)
                    
                    new_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([len(shapes) - 1]),
                        fill_color=target_tensor[int(y), int(x)].detach().clone().to(device), # Initialize with target color at that pixel + random alpha?
                    )
                    # Add alpha
                    new_group.fill_color = torch.cat([new_group.fill_color, torch.tensor([0.5]).to(device)]) 
                    shape_groups.append(new_group)

                # Re-index shape_ids (actually append is fine, indices are consistent if sequential)
                # But let's be safe and re-init optimizer
                optimizer = get_optimizer(shapes, shape_groups)
                
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
