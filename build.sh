#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════"
echo "  Synesthesia — build Docker image"
echo "═══════════════════════════════════════════════"

HF_CACHE="$HOME/.cache/huggingface/hub"
MODEL_CACHE="./model_cache"

# ── copy model weights into build context ─────────────────────────────────────
echo "[1/3] copying model weights into build context..."
rm -rf "$MODEL_CACHE"
mkdir -p "$MODEL_CACHE"

for MODEL in \
  "models--tiantiaf--wavlm-large-msp-podcast-emotion-dim" \
  "models--microsoft--wavlm-large"
do
  if [ -d "$HF_CACHE/$MODEL" ]; then
    echo "  → $MODEL"
    cp -r "$HF_CACHE/$MODEL" "$MODEL_CACHE/"
  else
    echo "  ✗ $MODEL not found in $HF_CACHE — run the server once first to download it"
    exit 1
  fi
done

# ── build Docker image ────────────────────────────────────────────────────────
echo "[2/3] building Docker image (this takes a few minutes first time)..."
docker build -t synesthesia:latest .

# ── clean up build context ─────────────────────────────────────────────────────
echo "[3/3] cleaning up..."
rm -rf "$MODEL_CACHE"

echo ""
echo "✓ Docker image built: synesthesia:latest"
echo ""
echo "To test: docker run --rm -p 5001:5001 synesthesia:latest"
echo "Then open client/index.html in your browser."
echo ""
echo "To build the Electron app:"
echo "  cd electron && npm install && npm run build:mac"
