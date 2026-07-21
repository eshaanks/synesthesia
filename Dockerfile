FROM python:3.10-slim

# system deps for librosa / soundfile / ffmpeg audio decode
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch first — separate layer for cache (large, rarely changes)
# use PyPI as primary index so typing-extensions resolves correctly;
# PyTorch CPU index as extra for the actual torch wheels
RUN pip install --no-cache-dir \
    torch==2.12.0 torchaudio==2.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# remaining deps
COPY server/requirements.txt requirements.txt
RUN grep -vE "^torch|^torchvision|^torchaudio" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# server code, client, presets
COPY server/   ./server/
COPY client/   ./client/
COPY presets/  ./presets/

# model weights baked in — build.sh copies from host HF cache before docker build
COPY model_cache/ /root/.cache/huggingface/hub/

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

EXPOSE 5001
CMD ["python3", "server/server.py"]
