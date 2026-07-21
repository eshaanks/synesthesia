#!/usr/bin/env bash
# Dev launcher — runs Flask directly (no Docker)
set -e
cd "$(dirname "$0")"

# create venv and install deps on first run
if [ ! -f ".venv/bin/python3" ]; then
  echo "First run — setting up Python environment (this takes a few minutes)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  # torch needs PyTorch index for CPU wheels; PyPI as primary avoids typing-extensions name mismatch
  .venv/bin/pip install torch==2.12.0 torchaudio==2.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
  grep -vE "^torch" server/requirements.txt | .venv/bin/pip install -r /dev/stdin
  echo "Setup complete."
fi
PYTHON=".venv/bin/python3"

# on first run the server will download model weights (~2.4 GB) from HuggingFace
if [ ! -d "$HOME/.cache/huggingface/hub/models--tiantiaf--wavlm-large-msp-podcast-emotion-dim" ]; then
  echo "Note: first server start will download model weights (~2.4 GB) — needs internet."
fi

pkill -f "server/server.py" 2>/dev/null || true
sleep 0.5

echo "Starting Flask server on :5001 (serves client + API)..."
cd server
"$PYTHON" server.py &
SERVER_PID=$!
cd ..

echo "Waiting for server..."
for i in $(seq 1 20); do
  curl -sf http://localhost:5001/health > /dev/null 2>&1 && break
  sleep 0.5
done

echo ""
echo "✓ Synesthesia running at http://localhost:5001"
open http://localhost:5001

trap "kill $SERVER_PID 2>/dev/null; echo stopped." INT TERM
wait
