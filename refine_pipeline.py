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
        "init_svg", help="Path to the source SVG (e.g., output/apple_red_lpips.svg)"
    )
    parser.add_argument(
        "target_image", help="Path to the target image (e.g., data/apple_green.jpg)"
    )
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--output_name", default=None, help="Output filename (without ext). Auto-generated if None.")
    args = parser.parse_args()

    # 1. 自动生成输出文件名
    if args.output_name is None:
        base_svg = os.path.splitext(os.path.basename(args.init_svg))[0]
        target_name = os.path.splitext(os.path.basename(args.target_image))[0]
        # 命名格式: SourceSVG_to_TargetImg_Iter
        args.output_name = f"{base_svg}_to_{target_name}_{args.num_iter}iter_refine"
        print(f"--> Auto-generated output name: {args.output_name}")

    # 2. 准备 GPU
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Optimization requires GPU.")
        return

    pydiffvg.set_use_gpu(True)
    device = torch.device("cuda:0")
    pydiffvg.set_device(device)

    # 3. 初始化 Loss (LPIPS)
    print("--> Initializing LPIPS loss...")
    perception_loss = ttools.modules.LPIPS().to(device)

    # 4. 读取目标图片 (Target Image)
    print(f"--> Loading Target Image: {args.target_image}")
    target = skimage.io.imread(args.target_image)
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.to(device)
    
    # 确保 Target 是 RGBA
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )

    # 预处理 Target for LPIPS (合成白底 + NCHW)
    bg = torch.ones(target.shape[0], target.shape[1], 3, device=device)
    target_rgb = target[:, :, 3:4] * target[:, :, :3] + (1 - target[:, :, 3:4]) * bg
    target_permuted = target_rgb.unsqueeze(0).permute(0, 3, 1, 2)

    # 5. 加载初始 SVG (Source SVG)
    print(f"--> Loading Source SVG: {args.init_svg}")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        args.init_svg
    )
    
    # 强制覆盖画布尺寸以匹配 Target (假设是一对一转换)
    # 如果 SVG 和 Target 尺寸不一致，可能需要缩放，这里暂时假设一致或以 Target 为准
    canvas_width, canvas_height = target.shape[1], target.shape[0]

    # 6. 数据迁移到 GPU
    for path in shapes:
        path.points = path.points.to(device)
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width = path.stroke_width.to(device)
        else:
            path.stroke_width = torch.tensor(path.stroke_width).to(device)

    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)

    # 7. 设置优化器
    # 仅优化 points (形状) 和 color (颜色)，不改变拓扑结构
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

    # Fine-tuning 策略：
    # 形状学习率可以适中 (lr=1.0)，因为需要从红苹果变成绿苹果，形状可能有微调
    # 颜色学习率需要保持 (lr=0.01)，因为颜色变化可能很大
    optimizer = torch.optim.Adam(
        [{"params": points_vars, "lr": 1.0}, {"params": color_vars, "lr": 0.01}]
    )

    # 8. 训练循环
    print(f"--> Start Refinement ({args.num_iter} iterations)...")
    start_time = time.time()

    for t in range(args.num_iter):
        optimizer.zero_grad()

        # Render
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )

        # Prepare Image for Loss (合成白底 + NCHW)
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + (1 - img[:, :, 3:4]) * bg
        img_permuted = img_rgb.unsqueeze(0).permute(0, 3, 1, 2)

        # Calculate Loss (LPIPS + Color MSE)
        lpips_loss = perception_loss(img_permuted, target_permuted)
        color_loss = (img_permuted.mean() - target_permuted.mean()).pow(2)
        
        # 组合 Loss
        loss = lpips_loss + color_loss

        if t % 20 == 0:
            print(f"  Iter {t}: Loss = {loss.item():.5f} (LPIPS: {lpips_loss.item():.5f})")

        loss.backward()
        optimizer.step()

        # 颜色约束
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)

    total_time = time.time() - start_time
    print(
        f"--> Finished in {total_time:.2f}s (Avg: {total_time/args.num_iter*1000:.1f}ms/it)"
    )

    # 9. 保存结果
    os.makedirs("output", exist_ok=True)
    
    # 保存 SVG
    svg_path = f"output/{args.output_name}.svg"
    pydiffvg.save_svg(svg_path, canvas_width, canvas_height, shapes, shape_groups)
    print(f"--> Saved Refined SVG to {svg_path}")
    
    # 保存 PNG (白底，所见即所得)
    png_path = f"output/{args.output_name}.png"
    pydiffvg.imwrite(img_rgb.cpu(), png_path, gamma=1.0)
    print(f"--> Saved Refined PNG to {png_path}")

if __name__ == "__main__":
    main()
