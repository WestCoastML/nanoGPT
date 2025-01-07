#!/bin/bash

set -euo pipefail

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Setup directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# First, try graceful shutdown
log "Attempting graceful shutdown of processes..."

# Stop wandb agents
if pgrep -f "wandb agent" > /dev/null; then
    log "Stopping wandb agents..."
    pkill -TERM -f "wandb agent"
    sleep 5  # Give processes time to cleanup
fi

# Stop training processes
if pgrep -f "python.*train.py" > /dev/null; then
    log "Stopping training processes..."
    pkill -TERM -f "python.*train.py"
    sleep 5
fi

# Force kill if processes still exist
log "Checking for remaining processes..."

for pattern in "wandb agent" "python.*train.py" "torchrun"; do
    if pgrep -f "$pattern" > /dev/null; then
        log "Force killing $pattern processes..."
        pkill -9 -f "$pattern"
    fi
done

# Clean up PID files
log "Cleaning up PID files..."
rm -f "$PROJECT_ROOT/runs/sweep_pids"/*.pid

# Update sweep status files if they exist
find "$PROJECT_ROOT/runs/sweeps" -name "status" -type f -exec sh -c 'echo "stopped" > "$1"' sh {} \;

log "Sweep shutdown completed"