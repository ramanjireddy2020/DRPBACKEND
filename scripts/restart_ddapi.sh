#!/bin/bash

# Kill processes on port 8080
pids=$(lsof -t -i:8080)
if [ -n "$pids" ]; then
  echo "Killing PIDs: $pids"
  kill -9 $pids
else
  echo "No process using port 8080"
fi

# Restart uvicorn in conda env
source ~/miniconda3/bin/activate
conda deactivate
conda activate pymol_env
# Run from src/ so the DRP_Main package is importable (imports use DRP_Main.app.*)
cd "$(dirname "$0")/../src"
nohup uvicorn DRP_Main.app.main:app --host 0.0.0.0 --port 8080 --reload &
conda deactivate

echo "✅ Uvicorn restarted on port 8080 in pymol_env"

