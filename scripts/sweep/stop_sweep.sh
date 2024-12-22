#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory (two levels up from script location)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Kill agents running on each GPU
for pid_file in "$PROJECT_ROOT/runs/sweep_pids"/agent_gpu*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        echo "Stopping agent with PID $pid"
        kill $pid 2>/dev/null
        rm "$pid_file"
    fi
done

echo "All agents stopped"