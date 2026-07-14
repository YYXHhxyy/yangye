# 平台要求 ARM64 架构，Ubuntu 为基础系统
FROM --platform=linux/arm64 ubuntu:22.04

# 安装 Python 和系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 安装 ARM64 版 PyTorch CPU
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir ultralytics tifffile && \
    pip3 install --no-cache-dir opencv-python-headless --force-reinstall

COPY . /workspace
WORKDIR /workspace

CMD ["python3", "run.py", "/input_path", "/output_path"]
