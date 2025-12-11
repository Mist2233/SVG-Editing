import pydiffvg
import torch
import skimage.io
import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target_path", help="Path to the target image (e.g., data/apple.jpg)"
    )
    parser.add_argument("--num_paths", type=int, default=128)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--output_name", default="result")
    args = parser.parse_args()

    # 1. 强制使用 GPU
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = torch.device("cuda:0")
    pydiffvg.set_device(device)

    # 2. 读取图片
    print(f"--> Loading {args.target_path}...")
    target = skimage.io.imread(args.target_path)
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.to(device)
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )

    # 3. 初始化形状 (官方的高效初始化)
    canvas_width, canvas_height = target.shape[1], target.shape[0]
    shapes = []
    shape_groups = []

    # 使用 Grid 初始化，比随机圆收敛更快
    for i in range(args.num_paths):
        num_segments = 1
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        # 随机分布在画布上
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

    # 4. 优化器
    points_vars = [p.points for p in shapes]
    color_vars = [g.fill_color for g in shape_groups]
    for v in points_vars + color_vars:
        v.requires_grad = True

    optimizer = torch.optim.Adam(
        [{"params": points_vars, "lr": 1.0}, {"params": color_vars, "lr": 0.01}]
    )

    # 5. 极速训练循环
    print(f"--> Start training {args.num_iter} iterations...")
    start_time = time.time()

    for t in range(args.num_iter):
        optimizer.zero_grad()
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )

        # 使用 MSE Loss (极速)
        loss = (img - target).pow(2).mean()

        # if t % 50 == 0:
        print(f"  Iter {t}: Loss = {loss.item():.5f}")

        loss.backward()
        optimizer.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    total_time = time.time() - start_time
    print(
        f"--> Finished in {total_time:.2f}s (Avg: {total_time/args.num_iter*1000:.1f}ms/it)"
    )

    # 6. 保存直接到 Windows 目录
    os.makedirs("output", exist_ok=True)
    save_path = f"output/{args.output_name}.svg"
    pydiffvg.save_svg(save_path, canvas_width, canvas_height, shapes, shape_groups)
    print(f"--> Saved to {save_path}")


if __name__ == "__main__":
    main()
