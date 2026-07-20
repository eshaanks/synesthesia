#!/usr/bin/env bash
# Synesthesia — Mac launcher
# Double-click this file to start the installation.
cd "$(dirname "$0")"

echo "Starting Synesthesia..."

# check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo ""
  echo "Docker is not running."
  echo "Please open Docker Desktop and wait for the whale icon in the menu bar, then double-click this file again."
  read -p "Press Enter to close..."
  exit 1
fi

# load image if not already loaded
if ! docker image inspect synesthesia:latest > /dev/null 2>&1; then
  echo "Loading Synesthesia image (first run, takes a minute)..."
  docker load < synesthesia.tar.gz
fi

# stop any previous instance
docker rm -f synesthesia 2>/dev/null || true

# check .env exists
if [ ! -f ".env" ]; then
  echo ""
  echo "Missing .env file with GROQ_API_KEY."
  echo "Create a .env file next to this script containing:"
  echo "  GROQ_API_KEY=your_key_here"
  read -p "Press Enter to close..."
  exit 1
fi

# run
echo "Launching..."
docker run -d --name synesthesia -p 5001:5001 --env-file .env synesthesia:latest

# wait for ready
echo "Waiting for server..."
for i in $(seq 1 30); do
  curl -sf http://localhost:5001/health > /dev/null 2>&1 && break
  sleep 1
done

echo "Opening browser..."
open http://localhost:5001

echo ""
echo "Synesthesia is running at http://localhost:5001"
echo "Close this window to keep it running, or press Ctrl+C to stop."
echo ""
trap "docker rm -f synesthesia 2>/dev/null; echo stopped." INT TERM
wait
