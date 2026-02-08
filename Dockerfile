# Use a stable, specific Debian-based Python image
FROM python:3.9-slim-bookworm

WORKDIR /app

# Copy weights and app
COPY ./weights/ /app/.cxas/weights
COPY interactive_cxas_app.py /app/

ENV CXAS_PATH=/app/

# Clean apt-get list and install only what is strictly necessary
# Removed libgl1-mesa-glx as opencv-python-headless doesn't need it
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir cxas==0.0.15 streamlit opencv-python-headless pillow numpy

EXPOSE 8501

CMD ["streamlit", "run", "interactive_cxas_app.py", "--server.port=8501", "--server.address=0.0.0.0"]