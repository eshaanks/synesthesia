FROM python:3.10-slim

# system deps for librosa / soundfile / ffmpeg audio decode
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first — needs custom index URL, keep as its own layer for cache
RUN pip install --no-cache-dir \
    torch==2.12.0 torchvision==0.27.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# remaining deps from pinned requirements
COPY server/requirements.txt requirements.txt
RUN grep -vE "^torch|^torchvision|^torchaudio" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# server code + vox_src model wrapper
COPY server/ ./server/

# model weights baked in — build.sh copies them from host HF cache before docker build
COPY model_cache/ /root/.cache/huggingface/hub/

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 5001
CMD ["python3", "server/server.py"]
