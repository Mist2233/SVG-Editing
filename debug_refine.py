import pydiffvg
import torch
import skimage.io
import argparse
import os
import time

# 【调试核心】取消阻塞，防止玄学死锁
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("init_svg", help="Path to the initial SVG")
    parser.add_argument("target_image", help="Path to the target image")
    parser.add_argument("--output_name", default="svg_b")
    args = parser.parse_args()

    print("[Debug] 1. Initializing GPU...")
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = torch.device("cuda:0")
    pydiffvg.set_device(device)
    torch.cuda.synchronize()

    print(f"[Debug] 2. Loading Target Image: {args.target_image}")
    target = skimage.io.imread(args.target_image)
    target = torch.from_numpy(target).to(torch.float32) / 255.0
    target = target.to(device)
    if target.shape[2] == 3:
        target = torch.cat(
            [target, torch.ones(target.shape[0], target.shape[1], 1).to(device)], dim=2
        )
    canvas_width, canvas_height = target.shape[1], target.shape[0]

    print(f"[Debug] 3. Loading SVG: {args.init_svg}")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        args.init_svg
    )

    # -------------------------------------------------
    # 【调试重点】检查数据是否都在 GPU 上
    # -------------------------------------------------
    print("[Debug] 4. Moving shapes to GPU...")
    try:
        for i, path in enumerate(shapes):
            path.points = path.points.to(device)
            # 检查 stroke_width
            if isinstance(path.stroke_width, torch.Tensor):
                path.stroke_width = path.stroke_width.to(device)
            else:
                # 如果是数字，转成 tensor
                path.stroke_width = torch.tensor(path.stroke_width).to(device)

        for i, group in enumerate(shape_groups):
            if group.fill_color is not None:
                group.fill_color = group.fill_color.to(device)
            if group.stroke_color is not None:
                group.stroke_color = group.stroke_color.to(device)
        torch.cuda.synchronize()
        print("   -> Shapes moved successfully.")
    except Exception as e:
        print(f"   -> Error moving shapes: {e}")
        return

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

    optimizer = torch.optim.Adam(points_vars + color_vars, lr=0.01)

    print("[Debug] 6. Starting Loop...")
    # 只跑 5 轮，看看哪里卡住
    for t in range(5):
        print(f"--- Iter {t} Start ---")

        optimizer.zero_grad()

        print("   a. Serialize Scene...")
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups
        )

        print("   b. Rendering (Forward)...")
        # 这一步最容易卡
        img = pydiffvg.RenderFunction.apply(
            canvas_width, canvas_height, 2, 2, 0, None, *scene_args
        )
        torch.cuda.synchronize()  # 强制等待 GPU
        print("      -> Render finished.")

        print("   c. Calculating Loss...")
        loss = (img - target).pow(2).mean()
        print(f"      -> Loss: {loss.item()}")

        print("   d. Backward Pass...")
        loss.backward()
        torch.cuda.synchronize()  # 强制等待 GPU
        print("      -> Backward finished.")

        print("   e. Optimizer Step...")
        optimizer.step()
        torch.cuda.synchronize()

        print(f"--- Iter {t} Done ---")

    print("[Debug] Test Finished Successfully.")


if __name__ == "__main__":
    main()
