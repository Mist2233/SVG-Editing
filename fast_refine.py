import pydiffvg
import torch
import skimage.io
import argparse
import os
import time

# 【关键】强制关闭阻塞，防止死锁
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "init_svg", help="Path to the initial SVG (e.g., output/svg_a.svg)"
    )
    parser.add_argument("target_image", help="Path to the target image")
    parser.add_argument("--num_iter", type=int, default=300)
    parser.add_argument("--output_name", default="svg_b")
    args = parser.parse_args()

    # 1. 准备 GPU
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = torch.device("cuda:0")
    pydiffvg.set_device(device)

    # 2. 读取目标图片
    print(f"--> Loading Target: {args.target_image}")
    target = skimage.io.imread(args.target_image)
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.to(device)
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )
    canvas_width, canvas_height = target.shape[1], target.shape[0]

    # 3. 加载 SVG
    print(f"--> Loading Init SVG: {args.init_svg}")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        args.init_svg
    )

    # 4. 数据迁移到 GPU (健壮版)
    for path in shapes:
        path.points = path.points.to(device)
        # 处理 stroke_width 可能是 float 的情况
        if isinstance(path.stroke_width, torch.Tensor):
            path.stroke_width = path.stroke_width.to(device)
        else:
            path.stroke_width = torch.tensor(path.stroke_width).to(device)

    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color = group.fill_color.to(device)
        if group.stroke_color is not None:
            group.stroke_color = group.stroke_color.to(device)

    # 5. 设置优化器
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

    # Fine-tuning 阶段，学习率可以稍微小一点
    optimizer = torch.optim.Adam(
        [{"params": points_vars, "lr": 1.0}, {"params": color_vars, "lr": 0.01}]
    )

    # 6. 训练循环
    print(f"--> Start Refinement ({args.num_iter} iterations)...")
    start_time = time.time()

    for t in range(args.num_iter):
        optimizer.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )

        # MSE Loss
        loss = (img - target).pow(2).mean()

        # if t % 50 == 0:
        print(f"  Iter {t}: Loss = {loss.item():.5f}")

        loss.backward()
        optimizer.step()

        # 颜色约束
        for group in shape_groups:
            if group.fill_color is not None:
                group.fill_color.data.clamp_(0.0, 1.0)
            if group.stroke_color is not None:
                group.stroke_color.data.clamp_(0.0, 1.0)

    total_time = time.time() - start_time
    print(f"--> Finished in {total_time:.2f}s")

    # 7. 保存结果
    os.makedirs("output", exist_ok=True)
    save_path = f"output/{args.output_name}.svg"
    pydiffvg.save_svg(save_path, canvas_width, canvas_height, shapes, shape_groups)
    print(f"--> Saved SVG B to {save_path}")


if __name__ == "__main__":
    main()
