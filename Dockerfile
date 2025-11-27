# 基础镜像 (你刚才辛苦拉下来的那个)
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 1. 安装系统依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git cmake ffmpeg libsm6 libxext6 build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install svgwrite svgpathtools cssutils numba torch-tools visdom scikit-image lpips tensorboardX

# 3. 安装 DiffVG
WORKDIR /tmp
# TUN 模式下，直接 clone 即可，无需镜像源魔法
RUN git clone --recursive https://github.com/BachiLi/diffvg.git
WORKDIR /tmp/diffvg

# 4. 配置与编译 (适配 RTX 4060)
RUN sed -i 's/CMAKE_CXX_STANDARD 14/CMAKE_CXX_STANDARD 17/g' CMakeLists.txt
RUN sed -i 's/add_subdirectory(pydiffvg_tensorflow)/# add_subdirectory(pydiffvg_tensorflow)/g' CMakeLists.txt
# 4060 属于 Ada 架构 (8.9)，但 CUDA 11.6 不认识，所以用 8.6+PTX 兼容
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
RUN python setup.py install

# 5. 设置工作区
WORKDIR /workspace