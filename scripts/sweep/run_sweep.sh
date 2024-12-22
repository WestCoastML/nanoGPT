#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Get the project root directory (two levels up from script location)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Initialize the sweep first
cd "$PROJECT_ROOT"  # Change to project root directory

# Create a temporary file to store wandb output
TEMP_OUTPUT=$(mktemp)

# Run wandb sweep, capturing BOTH stdout and stderr, then tee to temp file
wandb sweep config/sweep/wandb_sweep_config.yaml 2>&1 | tee "$TEMP_OUTPUT"

# Extract the sweep ID from the temporary file
SWEEP_ID=$(grep "wandb: Run sweep agent with:" "$TEMP_OUTPUT" \
  | awk '{print $NF}' \
  | awk -F'/' '{print $NF}')

# Optional: If you'd like to keep the full output, comment out or remove the next line
# rm "$TEMP_OUTPUT"

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to get sweep ID."
    exit 1
fi

echo "Successfully captured sweep ID: $SWEEP_ID"

# Create a directory for PIDs if it doesn't exist
mkdir -p "$PROJECT_ROOT/runs/sweep_pids"

# Function to run an agent on a specific GPU
run_agent() {
    local gpu=$1
    local sweep_id=$2
    
    echo "Starting agent on GPU $gpu with sweep ID: $sweep_id"
    CUDA_VISIBLE_DEVICES=$gpu wandb agent wcml/VSLM/$sweep_id &

    # Store the PID of the agent
    echo $! > "$PROJECT_ROOT/runs/sweep_pids/agent_gpu${gpu}.pid"
}

# Example: Start agents on 2 GPUs (adjust indices or add more as needed)
run_agent 0 "$SWEEP_ID"
run_agent 1 "$SWEEP_ID"

echo "Started agents on both GPUs. To stop them, run: ./scripts/sweep/stop_sweep.sh"

# Keep the script running until interrupted, so the agents remain active
wait
