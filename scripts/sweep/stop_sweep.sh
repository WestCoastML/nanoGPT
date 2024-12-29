#!/bin/bash

# ./scripts/sweep/stop_sweep.sh

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Kill DDP agent if it exists
if [ -f "$PROJECT_ROOT/runs/sweep_pids/agent_ddp.pid" ]; then
    pid=$(cat "$PROJECT_ROOT/runs/sweep_pids/agent_ddp.pid")
    echo "Stopping DDP agent with PID $pid"
    kill $pid 2>/dev/null
    rm "$PROJECT_ROOT/runs/sweep_pids/agent_ddp.pid"
fi

# Kill single-GPU agents if they exist
for pid_file in "$PROJECT_ROOT/runs/sweep_pids"/agent_gpu*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        echo "Stopping agent with PID $pid"
        kill $pid 2>/dev/null
        rm "$pid_file"
    fi
done

echo "All agents stopped"