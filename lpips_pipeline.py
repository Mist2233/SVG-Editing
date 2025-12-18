import pydiffvg
import torch
import skimage.io
import argparse
import os
import time
import ttools.modules

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_path", help="Path to the target image (e.g., data/apple.jpg)"
    )
    parser.add_argument("--num_paths", type=int, default=128)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--output_name", default=None, help="Output file base name. If None, auto-generated.")
    args = parser.parse_args()

    # 自动生成 output_name
    if args.output_name is None:
        base_name = os.path.splitext(os.path.basename(args.target_path))[0]
        args.output_name = f"{base_name}_{args.num_paths}path_{args.num_iter}iter_lpips"
        print(f"--> Auto-generated output name: {args.output_name}")

    # 1. 强制使用 GPU
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. LPIPS optimization requires GPU.")
        return
        
    pydiffvg.set_use_gpu(True)
    device = torch.device("cuda:0")
    pydiffvg.set_device(device)

    # 2. 初始化 LPIPS
    print("--> Initializing LPIPS loss...")
    # LPIPS expects inputs in [-1, 1] usually, or [0, 1]? 
    # ttools LPIPS usually takes [0, 1] images if not specified otherwise, but let's check standard usage.
    # In painterly_rendering.py it just passes the image (0-1 float).
    perception_loss = ttools.modules.LPIPS().to(device)

    # 3. 读取图片
    print(f"--> Loading {args.target_path}...")
    target = skimage.io.imread(args.target_path)
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.to(device)
    
    # 确保是 RGBA
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )

    # 准备 Target for LPIPS (Composite on white -> NCHW)
    # Target is HWC
    bg = torch.ones(target.shape[0], target.shape[1], 3, device=device)
    target_rgb = target[:, :, 3:4] * target[:, :, :3] + (1 - target[:, :, 3:4]) * bg
    # NCHW
    target_permuted = target_rgb.unsqueeze(0).permute(0, 3, 1, 2)

    # 4. 初始化形状 (Efficient Grid Init)
    canvas_width, canvas_height = target.shape[1], target.shape[0]
    shapes = []
    shape_groups = []

    for i in range(args.num_paths):
        num_segments = 1
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        # 随机分布
        p0 = (torch.rand(2) * torch.tensor([canvas_width, canvas_height])).to(device)
        points.append(p0)
        for j in range(num_segments):
            radius = 5.0
            p1 = (torch.rand(2) * radius).to(device) + p0
            p2 = (torch.rand(2) * radius).to(device) + p0
            p3 = (torch.rand(2) * radius).to(device) + p0
            points.append(p1)
            points.append(p2)
            points.append(p3)
            points = [p.contiguous() for p in points]

        path = pydiffvg.Path(num_control_points = num_control_points,
                            points = torch.stack(points),
                            stroke_width = torch.tensor(2.0).to(device),
                            is_closed = True)
        shapes.append(path)
        group = pydiffvg.ShapeGroup(
            torch.tensor([len(shapes) - 1]), torch.rand(4).to(device)
        )
        shape_groups.append(group)

    # 5. 优化器
    points_vars = [p.points for p in shapes]
    color_vars = [g.fill_color for g in shape_groups]
    for v in points_vars + color_vars:
        v.requires_grad = True

    optimizer = torch.optim.Adam(
        [{"params": points_vars, "lr": 1.0}, {"params": color_vars, "lr": 0.01}]
    )

    # 6. 训练循环
    print(f"--> Start training {args.num_iter} iterations with LPIPS...")
    start_time = time.time()

    for t in range(args.num_iter):
        optimizer.zero_grad()
        
        # Forward Render
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )

        # Prepare Image for LPIPS
        # Composite on white
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * bg
        # NCHW
        img_permuted = img_rgb.unsqueeze(0).permute(0, 3, 1, 2)

        # Calculate Loss
        # LPIPS + Mean Color Regularization
        lpips_loss = perception_loss(img_permuted, target_permuted)
        color_loss = (img_permuted.mean() - target_permuted.mean()).pow(2)
        
        loss = lpips_loss + color_loss

        if t % 20 == 0:
            print(f"  Iter {t}: Loss = {loss.item():.5f} (LPIPS: {lpips_loss.item():.5f})")

        loss.backward()
        optimizer.step()
        
        # Clamp colors
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    total_time = time.time() - start_time
    print(
        f"--> Finished in {total_time:.2f}s (Avg: {total_time/args.num_iter*1000:.1f}ms/it)"
    )

    # 7. 保存结果
    os.makedirs("output", exist_ok=True)
    save_path = f"output/{args.output_name}.svg"
    pydiffvg.save_svg(save_path, canvas_width, canvas_height, shapes, shape_groups)
    print(f"--> Saved SVG to {save_path}")
    
    # Optional: Save PNG for preview
    pydiffvg.imwrite(img.cpu(), f"output/{args.output_name}.png", gamma=1.0)
    print(f"--> Saved PNG to output/{args.output_name}.png")

if __name__ == "__main__":
    main()
