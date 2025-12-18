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

### 5. 编写代码 (Coding)

*   **编辑器:** 直接在 Windows 上使用 **VS Code** 打开 `E:\Development\SVG-Editing` 文件夹。
*   **编辑:** 在 VS Code 中编写/修改代码，保存文件 (`Ctrl+S`)。
*   **生效:** 由于使用了挂载 (`-v`)，你修改的代码会**实时同步**到 Docker 容器中，无需重启容器，直接在容器终端再次运行 `python main.py` 即可生效。

### 6. 查看结果 (Check Results)

程序生成的输出文件（如 `output/result.svg`）会保存在 Windows 的 `output` 文件夹中。
你可以直接在 Windows 上双击打开 SVG 文件查看效果。

### ⚡ 常见问题 (FAQ)

*   **Q: 报错 `RuntimeError: Numpy is not available`?**
    *   **A:** 容器内的 NumPy 版本过高。在容器内运行 `pip install "numpy<2.0"` 即可解决。
*   **Q: 运行速度慢？**
    *   **A:** 确保 Windows 上没有运行其他占用显存的大型程序（如游戏）。DiffVG 需要独占 GPU 进行渲染。
*   **Q: 找不到 `pydiffvg` 模块？**
    *   **A:** 请确保你在 Docker 容器内运行代码。Windows 本地的 Python 环境并没有安装这个库。

其他配置问题，可以参考issue1: https://github.com/Mist2233/SVG-Editing/issues/1
