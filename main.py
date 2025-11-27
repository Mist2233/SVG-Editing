import pydiffvg
import torch
import skimage.io
import argparse
import os


def main():
    # 1. 准备工作
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    # 2. 读取目标图片 (Target Image)
    # 请确保你在 data 目录下放了一张 cat.jpg
    target_path = "data/cat.jpg"
    target = skimage.io.imread(target_path)
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.to(device)

    # 确保图片是 RGBA 或 RGB
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )

    # 3. 初始化 SVG 形状 (随机初始化一些圆和路径)
    canvas_width, canvas_height = target.shape[1], target.shape[0]
    num_paths = 256  # 路径数量，越多越精细但越慢

    shapes = []
    shape_groups = []

    # 随机生成一些圆作为初始形状
    for i in range(num_paths):
        num_segments = 1
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        points = []
        p0 = (torch.rand(2) * canvas_width).to(device)
        points.append(p0)
        for j in range(num_segments):
            radius = 10
            p1 = (torch.rand(2) * radius).to(device) + p0
            p2 = (torch.rand(2) * radius).to(device) + p0
            p3 = (torch.rand(2) * radius).to(device) + p0
            points.append(p1)
            points.append(p2)
            points.append(p3)
            points = [p.contiguous() for p in points]  # 内存连续化，防止报错

        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=torch.stack(points),
            stroke_width=torch.tensor(1.0).to(device),
            is_closed=True,
        )
        shapes.append(path)

        # 随机颜色
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]),
            fill_color=torch.rand(4).to(device),
        )
        shape_groups.append(path_group)

    # 4. 设置优化器 (Adam)
    # 我们要优化两个东西：形状的位置(points) 和 颜色(fill_color)
    points_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    for group in shape_groups:
        group.fill_color.requires_grad = True
        color_vars.append(group.fill_color)

    optimizer = torch.optim.Adam(
        [{"params": points_vars, "lr": 1.0}, {"params": color_vars, "lr": 0.01}]
    )

    # 5. 开始迭代优化
    print("Start optimizing...")
    # 创建输出目录
    os.makedirs("output", exist_ok=True)

    for t in range(200):  # 跑200轮试试
        optimizer.zero_grad()

        # 渲染当前的 SVG
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )

        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )

        # 计算 Loss (生成图和原图的像素差)
        loss = (img - target).pow(2).mean()

        print(f"Iteration {t}, Loss: {loss.item()}")

        loss.backward()

        # 梯度修剪，防止形状飞出画面
        optimizer.step()

        # 约束颜色在 0-1 之间
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

    # 6. 保存最终结果 SVG
    pydiffvg.save_svg(
        "output/result.svg", canvas_width, canvas_height, shapes, shape_groups
    )
    print("Done! Check output/result.svg")


if __name__ == "__main__":
    main()
