#!/usr/bin/env bash
# Dev launcher — runs Flask directly (no Docker)
set -e
cd "$(dirname "$0")"

# use local venv if present, otherwise fall back to system python3
if [ -f ".venv/bin/python3" ]; then
  PYTHON=".venv/bin/python3"
elif [ -f "venv/bin/python3" ]; then
  PYTHON="venv/bin/python3"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
else
  echo "ERROR: python3 not found. Create a venv with 'python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt'"
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
