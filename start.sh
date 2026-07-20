#!/usr/bin/env bash
# Start the Synesthesia installation: Flask server + static file server + browser
set -e
cd "$(dirname "$0")"

VENV="/Users/aiswaryashybu/ART+CODE/Proj/ImageBind/imagebind-env"
PYTHON="$VENV/bin/python3"

if [ ! -f "$PYTHON" ]; then
  echo "ERROR: Python env not found at $VENV"
  echo "Update the VENV path in start.sh to match your environment."
  exit 1
fi

# kill any stale instances
pkill -f "server/server.py" 2>/dev/null || true
pkill -f "http.server 8000" 2>/dev/null || true
sleep 0.5

echo "[1/2] starting Flask server on :5001..."
cd server
"$PYTHON" server.py &
SERVER_PID=$!
cd ..

echo "[2/2] starting client on :8000..."
"$PYTHON" -m http.server 8000 --directory client &
CLIENT_PID=$!

# wait for server to be ready
echo "waiting for server..."
for i in $(seq 1 20); do
  curl -sf http://localhost:5001/health > /dev/null 2>&1 && break
  sleep 0.5
done

echo ""
echo "✓ Synesthesia running"
echo "  Flask:  http://localhost:5001"
echo "  Client: http://localhost:8000"
echo ""
echo "Opening browser..."
open http://localhost:8000

echo "Press Ctrl+C to stop."
trap "kill $SERVER_PID $CLIENT_PID 2>/dev/null; echo 'stopped.'" INT TERM
wait
