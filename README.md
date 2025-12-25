## 🛠️ Development Workflow

本项目使用 Docker 容器化环境进行开发，以确保 `diffvg` 及其依赖（CUDA, PyTorch, NumPy）的一致性与稳定性。

### 1. 前置准备 (Prerequisites)

*   **Windows 10/11**
*   **Docker Desktop** (需开启 WSL 2 后端支持)
*   **NVIDIA Driver** (宿主机需安装显卡驱动)
*   **DiffVG 镜像**: 确保已通过 `docker build` 或 `docker commit` 构建了名为 `diffvg-env` 的镜像。

### 2. 项目目录结构
请保持以下目录结构，以便 Docker 挂载路径正确：

```text
E:\Development\SVG-Editing\   <-- 项目根目录
├── data\                     # 存放输入图片 (e.g., input.jpg)
├── output\                   # 存放生成的 SVG/PNG 结果
├── color_scripts\            # [New] 核心颜色优化脚本
│   ├── batch_color_refine_smart.py   # 用于有边框的图像
│   └── batch_lineart_thicken.py      # 用于无边框/清洗后的线稿
├── _archive\                 # [New] 归档的旧版本脚本
├── ref_docs\                 # [New] AI生成的开发文档和问题日志
├── main.py                   # 你的主程序代码
├── Dockerfile                # 环境构建文件
└── README.md                 # 说明文档
```

### 3. 启动开发环境 (Start Environment)

每次开始开发前，请在 Windows PowerShell 中运行以下命令启动容器：

```powershell
# 进入项目根目录
cd E:\Development\SVG-Editing

# 启动容器 (挂载当前目录到容器内的 /workspace)
# --rm: 退出时自动删除容器 (防止残留)
# --gpus all: 开启显卡支持
docker run --gpus all -it --rm -v ${PWD}:/workspace diffvg-env
```

成功进入后，终端提示符会变为 `# root@...`，且当前目录为 `/workspace`。

### 4. 运行代码 (Run Code)

在容器内部，你的代码位于 `/workspace` 目录下。

#### 4.1 基础示例
*   **运行官方 Demo (测试环境):**
    ```bash
    # 官方示例脚本位于容器的临时目录中
    cd /tmp/diffvg/apps
    python painterly_rendering.py /workspace/data/cat.jpg --num_paths 512 --max_width 4.0 --use_lpips_loss
    ```

#### 4.2 LPIPS 矢量化 (Step 1)
使用 `lpips_pipeline.py` 将位图转换为矢量图，使用 LPIPS 感知损失以获得更好的结构和纹理。

```bash
# 用法: python lpips_pipeline.py <目标图片路径> [参数]
# 示例: 将红苹果矢量化 (2048个路径, 500轮迭代)
python lpips_pipeline.py data/apple_red.jpg --num_paths 2048 --num_iter 500
```
*   **输入**: `data/apple_red.jpg`
*   **输出**: `output/apple_red_2048path_500iter_lpips.svg` (自动命名)

#### 4.3 SVG 拓扑编辑与微调 (Step 2)
使用 `refine_pipeline.py` 加载现有的 SVG，保持其拓扑结构（Path 数量和顺序）不变，仅微调参数使其拟合另一张目标图片。

```bash
# 用法: python refine_pipeline.py <源SVG路径> <目标图片路径> [参数]
# 示例: 将红苹果 SVG 变成 绿苹果 (500轮迭代)
python refine_pipeline.py output/apple_red_2048path_500iter_lpips.svg data/apple_green.jpg --num_iter 500
```
*   **输入**: 源 SVG (`apple_red...svg`) + 目标图片 (`apple_green.jpg`)
*   **输出**: `output/apple_red..._to_apple_green_500iter_refine.svg` (自动命名)
*   **用途**: 制作成对的 SVG 数据集 (SVG_A -> SVG_B)，用于训练 SVG 编辑模型。

#### 4.4 动态路径优化 (Advanced)
通过动态剪枝（Pruning）和生长（Spawning）来优化路径数量和分布，适合处理复杂的拓扑变化。

**A. 动态剪枝 (Dynamic Pruning)**
移除低透明度或对画面贡献小的路径，精简 SVG。支持加载现有 SVG 进行热启动。

```bash
# 用法: python dynamic_pruning.py <目标图片> --init_svg <初始SVG> [参数]
# 示例: 基于红苹果 SVG，生成被咬一口的红苹果 (自动移除被咬掉部分的路径)
python dynamic_pruning.py data/apple_red_bite.jpg --init_svg output/apple_red_2048path_500iter_lpips.svg --output_name apple_bite_pruning --num_iter 50 --prune_threshold 0.005 --use_mse
```
*   `--prune_threshold`: 透明度阈值，低于此值的路径将被删除。
*   `--use_mse`: 使用 MSE 损失代替 LPIPS，大幅提升速度（适合微调）。

**B. 动态生长 (Dynamic Spawning)**
在高误差区域自动生成新路径，补充细节（如增加叶子）。

```bash
# 用法: python dynamic_spawning.py <目标图片> --init_svg <初始SVG> [参数]
# 示例: 基于红苹果 SVG，生成带叶子的苹果 (自动在叶子区域生长新路径)
python dynamic_spawning.py data/apple_red_with_leaves.jpg --init_svg output/apple_red_2048path_500iter_lpips.svg --output_name apple_with_leaves --num_iter 50 --spawn_interval 50 --use_mse --max_paths 2200
```
*   `--spawn_interval`: 每隔多少轮尝试生长一次。
*   `--max_paths`: 允许的最大路径数（建议设置得比初始路径数大，以便有空间生长）。

### 5. 颜色与线稿优化 (New Workflow)

针对成对数据（Lineart -> Color）的专用优化脚本，现已整理至 `color_scripts/` 目录。

#### 5.1 目录说明
*   **`color_scripts/`**: 存放当前正在使用的、验证有效的生产脚本。
*   **`_archive/`**: 存放所有历史版本、实验性或已废弃的脚本（如 v2, v3, solid 等）。
*   **`ref_docs/`**: 存放开发过程中的技术文档、问题排查日志和原理说明。

#### 5.2 核心脚本使用
1.  **`batch_color_refine_smart.py`**
    *   **适用场景**: **有边框**的原始 SVG 图像。
    *   **特点**: 智能处理透明度，能够很好地保持原有边框的结构，同时优化填充颜色。
    *   **用法**:
        ```bash
        cd color_scripts
        python batch_color_refine_smart.py --only 1 2
        ```

2.  **`batch_lineart_thicken.py`**
    *   **适用场景**: **无边框** 或 **经过 `clean_svg.py` 清洗后** 的 SVG 图像。
    *   **特点**: 包含“战术性加粗”策略，防止细线在优化过程中消失；强制不透明度为 1.0，确保颜色鲜艳且覆盖完整。
    *   **用法**:
        ```bash
        cd color_scripts
        python batch_lineart_thicken.py --only 3 4
        ```

### 6. 编写代码 (Coding)
