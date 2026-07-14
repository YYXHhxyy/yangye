FROM --platform=linux/arm64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q https://yangye-java-ai.oss-cn-beijing.aliyuncs.com/best.pt -O /workspace/best.pt

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics tifffile && \
    pip install --no-cache-dir opencv-python-headless --force-reinstall

COPY run.py /workspace/

WORKDIR /workspace

CMD ["python", "run.py", "/input_path", "/output_path"]
