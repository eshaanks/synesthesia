#!/usr/bin/env bash
# Build the Synesthesia Docker image and export it as a portable tarball.
# Run this on your development machine before copying to a USB drive.
set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════"
echo "  Synesthesia — build"
echo "═══════════════════════════════════════════════"

HF_CACHE="$HOME/.cache/huggingface/hub"
MODEL_CACHE="./model_cache"

# ── 1. copy model weights into build context ──────────────────────────────────
echo "[1/4] copying model weights..."
rm -rf "$MODEL_CACHE"
mkdir -p "$MODEL_CACHE"

for MODEL in \
  "models--tiantiaf--wavlm-large-msp-podcast-emotion-dim" \
  "models--microsoft--wavlm-large"
do
  if [ -d "$HF_CACHE/$MODEL" ]; then
    echo "  → $MODEL"
    cp -rP "$HF_CACHE/$MODEL" "$MODEL_CACHE/"
  else
    echo "  ✗ $MODEL not found in $HF_CACHE"
    echo "    Start the server once to download it, then re-run build.sh"
    exit 1
  fi
done

# drop pytorch_model.bin snapshot for wavlm-large — safetensors is preferred
# and keeping both doubles the blob storage inside the image
WAVLM_SNAPS="$MODEL_CACHE/models--microsoft--wavlm-large/snapshots"
if [ -d "$WAVLM_SNAPS" ]; then
  for SNAP in "$WAVLM_SNAPS"/*/; do
    if [ -e "$SNAP/pytorch_model.bin" ] && ! [ -e "$SNAP/model.safetensors" ]; then
      echo "  removing pytorch-only snapshot: $(basename $SNAP)"
      rm -rf "$SNAP"
    fi
  done
fi

# ── 2. build Docker image ─────────────────────────────────────────────────────
echo "[2/4] building Docker image..."
docker build -t synesthesia:latest .

# ── 3. clean up build context ─────────────────────────────────────────────────
echo "[3/4] cleaning up build context..."
rm -rf "$MODEL_CACHE"

# ── 4. export tarball for USB distribution ────────────────────────────────────
echo "[4/4] exporting synesthesia.tar.gz (this takes a minute)..."
docker save synesthesia:latest | gzip > synesthesia.tar.gz

echo ""
echo "✓ Done."
echo ""
echo "USB contents needed:"
echo "  synesthesia.tar.gz  ($(du -sh synesthesia.tar.gz | cut -f1))"
echo "  .env                (with GROQ_API_KEY=...)"
echo "  install.command     (Mac)"
echo "  install.bat         (Windows)"
echo ""
echo "To test locally:"
echo "  docker run --rm -p 5001:5001 --env-file .env synesthesia:latest"
echo "  open http://localhost:5001"
