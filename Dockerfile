FROM python:3.10-slim

# system deps for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps first (layer cache)
COPY server/requirements.txt requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir flask flask-cors numpy librosa safetensors transformers

# copy server code and vox_src
COPY server/ ./server/

# bake model weights in — copied from host cache at build time
# (build script copies them into build-context/model_cache before docker build)
COPY model_cache/ /root/.cache/huggingface/hub/

ENV PYTHONUNBUFFERED=1
EXPOSE 5001

CMD ["python3", "server/server.py"]
