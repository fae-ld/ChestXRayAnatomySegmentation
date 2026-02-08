FROM python:3.9-slim-bookworm

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxext6 libgl1 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Install Python packages (DOWNGRADE TORCH ke 2.5.1)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir cxas==0.0.15 streamlit opencv-python-headless pillow numpy

# --- FIX 1: PATCHING CXAS (Colorcet Error) ---
RUN sed -i 's/colors = \[cc.cm.glasbey_bw_minc_20(i) for i in range(len(id2label_dict.keys()))\]/colors = []/g' /usr/local/lib/python3.9/site-packages/cxas/label_mapper.py

# --- FIX 2: WEIGHTS PRE-POSITIONING ---
COPY ./weights/UNet_ResNet50_default.pth /root/.cxas/weights/UNet_ResNet50_default.pth

# 3. Copy files
COPY . .
RUN mkdir -p /app/tmp/output && chmod -R 777 /app/tmp

EXPOSE 8501

# --- FIX 3: ENVIRONMENT VARIABLE ---
# Force PyTorch to allow loading weights
ENV TORCH_LOAD_WEIGHTS_ONLY=0

CMD ["streamlit", "run", "interactive_cxas_app.py", "--server.port=8501", "--server.address=0.0.0.0"]