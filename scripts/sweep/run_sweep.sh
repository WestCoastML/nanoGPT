#!/bin/bash

# How to run:-
# #Run in single-GPU mode with 5 agents, one on each GPU
# ./scripts/sweep/run_sweep.sh --num-gpus 5

# # Run in DDP mode with 3 GPUs
# ./scripts/sweep/run_sweep.sh --ddp --num-gpus 3

# # To stop the agents, run the following command
# ./scripts/sweep/stop_sweep.sh

# Add debug output
set -x

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Parse command line arguments
USE_DDP=0
NUM_GPUS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --ddp)
            USE_DDP=1
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create runs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/runs/sweep_pids"

if [ $USE_DDP -eq 1 ]; then
    echo "Creating temporary sweep config with $NUM_GPUS GPUs..."
    # Create a temporary config file with the correct number of GPUs
    TMP_CONFIG=$(mktemp)
    sed "s/--nproc_per_node=[0-9]*/--nproc_per_node=$NUM_GPUS/" \
        config/sweep/wandb_sweep_config_ddp.yaml > "$TMP_CONFIG"
    
    echo "Initializing DDP sweep with modified config..."
    SWEEP_OUTPUT=$(wandb sweep "$TMP_CONFIG" 2>&1)
    rm "$TMP_CONFIG"
else
    echo "Initializing single-GPU sweep..."
    SWEEP_OUTPUT=$(wandb sweep config/sweep/wandb_sweep_config_single.yaml 2>&1)
fi

# Extract sweep ID
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep "wandb: Run sweep agent with:" | \
           awk '{print $NF}' | awk -F'/' '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to get sweep ID."
    echo "Full sweep output:"
    echo "$SWEEP_OUTPUT"
    exit 1
fi

echo "Successfully captured sweep ID: $SWEEP_ID"

if [ $USE_DDP -eq 1 ]; then
    echo "Running DDP sweep across $NUM_GPUS GPUs..."
    # Run wandb agent directly - torchrun is configured in the sweep config
    WANDB_AGENT_DISABLE_FLAPPING=true wandb agent wcml/VSLM/$SWEEP_ID &
    echo $! > "$PROJECT_ROOT/runs/sweep_pids/agent_ddp.pid"
    echo "Started DDP sweep agent. To stop it, run: ./scripts/sweep/stop_sweep.sh"
else
    echo "Running individual agents on $NUM_GPUS GPUs..."
    # Start separate agent on each GPU
    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        # Add unique run_dir for each agent to prevent conflicts
        CUDA_VISIBLE_DEVICES=$gpu WANDB_RUN_DIR="./runs/agent_${gpu}" wandb agent wcml/VSLM/$SWEEP_ID &
        echo $! > "$PROJECT_ROOT/runs/sweep_pids/agent_gpu${gpu}.pid"
    done
    echo "Started agents on $NUM_GPUS GPUs. To stop them, run: ./scripts/sweep/stop_sweep.sh"
fi

# Keep the script running until interrupted
wait