#!/usr/bin/env bash
# Dev launcher — runs Flask directly (no Docker)
set -e
cd "$(dirname "$0")"

VENV="/Users/aiswaryashybu/ART+CODE/Proj/ImageBind/imagebind-env"
PYTHON="$VENV/bin/python3"

if [ ! -f "$PYTHON" ]; then
  echo "ERROR: Python env not found at $VENV"
  echo "Update the VENV path in start.sh to match your environment."
  exit 1
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
