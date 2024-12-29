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

# Initialize the sweep and capture the sweep ID
TEMP_OUTPUT=$(mktemp)

if [ $USE_DDP -eq 1 ]; then
    echo "Initializing DDP sweep with command:"
    echo "wandb sweep config/sweep/wandb_sweep_config_ddp.yaml"
    wandb sweep config/sweep/wandb_sweep_config_ddp.yaml 2>&1 | tee "$TEMP_OUTPUT"
else
    echo "Initializing single-GPU sweep with command:"
    echo "wandb sweep config/sweep/wandb_sweep_config_single.yaml"
    wandb sweep config/sweep/wandb_sweep_config_single.yaml 2>&1 | tee "$TEMP_OUTPUT"
fi

SWEEP_ID=$(grep "wandb: Run sweep agent with:" "$TEMP_OUTPUT" \
    | awk '{print $NF}' \
    | awk -F'/' '{print $NF}')

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to get sweep ID."
    echo "Full temp output:"
    cat "$TEMP_OUTPUT"
    exit 1
fi

echo "Successfully captured sweep ID: $SWEEP_ID"

if [ $USE_DDP -eq 1 ]; then
    echo "Running DDP sweep across $NUM_GPUS GPUs..."
    # Create comma-separated list of GPU indices
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    
    # Print full command
    echo "Launching command:"
    echo "WANDB_AGENT_DISABLE_FLAPPING=true CUDA_VISIBLE_DEVICES=$GPU_LIST wandb agent wcml/VSLM/$SWEEP_ID"
    
    # Launch single DDP process that uses multiple GPUs
    WANDB_AGENT_DISABLE_FLAPPING=true CUDA_VISIBLE_DEVICES=$GPU_LIST \
    wandb agent --count 1 wcml/VSLM/$SWEEP_ID &
    
    # Store the PID
    echo $! > "$PROJECT_ROOT/runs/sweep_pids/agent_ddp.pid"
    echo "Started DDP sweep agent. To stop it, run: ./scripts/sweep/stop_sweep.sh"
else
    echo "Running individual agents on $NUM_GPUS GPUs..."
    # Start separate agent on each GPU
    for gpu in $(seq 0 $((NUM_GPUS-1))); do
        # Print full command for each GPU
        echo "Starting agent on GPU $gpu with command:"
        echo "CUDA_VISIBLE_DEVICES=$gpu wandb agent wcml/VSLM/$SWEEP_ID"
        CUDA_VISIBLE_DEVICES=$gpu wandb agent wcml/VSLM/$SWEEP_ID &
        echo $! > "$PROJECT_ROOT/runs/sweep_pids/agent_gpu${gpu}.pid"
    done
    echo "Started agents on $NUM_GPUS GPUs. To stop them, run: ./scripts/sweep/stop_sweep.sh"
fi

# Keep the script running until interrupted
wait
