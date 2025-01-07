#!/bin/bash

# How to run:-
# #Run in single-GPU mode with 5 agents, one on each GPU
# ./scripts/sweep/run_sweep.sh --num-gpus 5

# # Run in DDP mode with 3 GPUs
# ./scripts/sweep/run_sweep.sh --ddp --num-gpus 3

# # To stop the agents, run the following command
# ./scripts/sweep/stop_sweep.sh

# Strict error handling and debugging
set -euo pipefail
set -x

# Determine script and project root directories early
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Setup logging folder and LOG_FILE before using log()
mkdir -p "$PROJECT_ROOT/runs/logs"
LOG_FILE="$PROJECT_ROOT/runs/logs/sweep_$(date +%Y%m%d_%H%M%S).log"

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $*" >&2
}

echo "Starting run_sweep.sh. Logging to $LOG_FILE"

# Parse arguments
USE_DDP=0
NUM_GPUS=1
RESUME_SWEEP=""

# Grab a timestamp for a local ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

while [[ $# -gt 0 ]]; do
    case $1 in
        --ddp) USE_DDP=1; shift ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME_SWEEP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Current issue: Doesn't properly handle sweep resumption
if [ -n "$RESUME_SWEEP" ]; then
    SWEEP_ID="$RESUME_SWEEP"
else
    config_file="config/sweep/wandb_sweep_config_$([ $USE_DDP -eq 1 ] && echo ddp || echo single).yaml"
    log "Creating new sweep from $config_file"
    SWEEP_OUTPUT="$(wandb sweep "$config_file" 2>&1 || true)"
    log "Sweep output: $SWEEP_OUTPUT"

    # Parse real W&B sweep ID (if any)
    REAL_SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'ID: [a-zA-Z0-9]*' | cut -d' ' -f2)
    if [ -z "$REAL_SWEEP_ID" ]; then
      log "Failed to get sweep ID from W&B. Using local ID only."
      REAL_SWEEP_ID="NoWandbID"
    fi

    # This is the actual SWEEP_ID used for wandb agent:
    SWEEP_ID="$REAL_SWEEP_ID"
fi

log "Using sweep ID: $SWEEP_ID"

# Construct a local ID combining the timestamp + W&B ID
LOCAL_SWEEP_ID="${TIMESTAMP}_${REAL_SWEEP_ID}"

log "LOCAL_SWEEP_ID: $LOCAL_SWEEP_ID"

# Process Management
cleanup() {
    log "Cleanup initiated..."
    
    # Kill all child processes in the process group
    pkill -P $$
    
    # Kill any wandb agents that might be running
    pkill -f "wandb agent"
    
    # Kill any related Python processes
    pkill -f "python.*train.py"
    
    # Clean up PID files
    rm -f "$PROJECT_ROOT/runs/sweep_pids"/*.pid
    
    # Optionally write sweep status to a file
    if [ -n "${SWEEP_ID:-}" ]; then
        echo "stopped" > "$PROJECT_ROOT/runs/sweeps/$SWEEP_ID/status"
    fi
    
    log "Cleanup completed"
    exit 0
}

# Trap signals as early as possible
trap cleanup SIGINT SIGTERM EXIT

monitor_resources() {
    local sweep_id=$1
    local monitor_log="$PROJECT_ROOT/runs/logs/resources_${sweep_id}.csv"
    
    echo "timestamp,gpu_util,gpu_mem,cpu_util,ram_util" > "$monitor_log"
    
    while true
    do
        timestamp=$(date +%s)
        gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0,0")
        cpu_util=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
        ram_util=$(free | grep Mem | awk '{print $3/$2 * 100}')
        
        echo "$timestamp,$gpu_stats,$cpu_util,$ram_util" >> "$monitor_log"
        sleep 5
    done
}

setup_directories() {
    mkdir -p "$PROJECT_ROOT/runs/sweeps"
    mkdir -p "$PROJECT_ROOT/runs/sweep_pids"
    mkdir -p "$PROJECT_ROOT/runs/logs"
    mkdir -p "$PROJECT_ROOT/runs/checkpoints"

    # Optionally also create directory for the local sweep:
    mkdir -p "$PROJECT_ROOT/runs/sweeps/$LOCAL_SWEEP_ID"
}

launch_ddp_sweep() {
    local num_gpus=$1
    local wandb_sweep_id=$2

    # We’ll pass the local sweep ID + the W&B ID (if any) to the environment
    export LOCAL_SWEEP_ID
    export WANDB_SWEEP_ID="$wandb_sweep_id"

    log "Launching DDP sweep with $num_gpus GPUs"
    
    # Start resource monitoring
    monitor_resources "$sweep_id" &
    monitor_pid=$!
    echo $monitor_pid > "$PROJECT_ROOT/runs/sweep_pids/monitor_${sweep_id}.pid"
    
    # Set up DDP environment
    export NUM_GPUS=$num_gpus
    export USE_DDP=1
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpus-1)))
    export NCCL_DEBUG=INFO
    export NCCL_IB_TIMEOUT=23
    export NCCL_SOCKET_TIMEOUT=120
    
    # Launch single DDP agent
    WANDB_AGENT_DISABLE_FLAPPING=true \
    WANDB_SWEEP_ID="$wandb_sweep_id" \
    WANDB_RUN_DIR="$PROJECT_ROOT/runs/sweeps/$wandb_sweep_id/ddp_agent" \
    bash "$PROJECT_ROOT/runs/sweep_pids/ddp_agent_wrapper.sh" $wandb_sweep_id $num_gpus &
    
    agent_pid=$!
    echo $agent_pid > "$PROJECT_ROOT/runs/sweep_pids/agent_ddp_${sweep_id}.pid"
    
    wait $agent_pid || {
        log "DDP agent failed with exit code $?"
        cleanup
        exit 1
    }
}

launch_single_gpu_sweep() {
    local num_gpus=$1
    local sweep_id=$2

    log "Launching single-GPU sweep across $num_gpus GPUs"

    export LOCAL_SWEEP_ID
    export WANDB_SWEEP_ID="$sweep_id"
    
    # Use expandable_segments to reduce fragmentation
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    # Start resource monitoring in the background
    monitor_resources "$sweep_id" &
    monitor_pid=$!
    
    # We'll collect the agent PIDs in an array, then wait on those specifically
    agent_pids=()

    for gpu in $(seq 0 $((num_gpus-1)))
    do
        export CUDA_VISIBLE_DEVICES=$gpu
        export WANDB_RUN_DIR="$PROJECT_ROOT/runs/sweeps/$sweep_id/agent_${gpu}"
        export WANDB_AGENT_DISABLE_FLAPPING=true
        export WANDB_SWEEP_ID=$sweep_id
        
        wandb agent "wcml/VSLM/$sweep_id" \
            2>&1 | tee "$PROJECT_ROOT/runs/logs/sweep_${sweep_id}_gpu${gpu}.log" &
        
        agent_pid=$!
        # Store each agent PID in a file (unchanged) and also in our array:
        echo "$agent_pid" > "$PROJECT_ROOT/runs/sweep_pids/agent_gpu${gpu}_${sweep_id}.pid"
        agent_pids+=("$agent_pid")
    done
    
    # Wait for all wandb agent PIDs to complete in a loop (or we can do a simple for wait).
    for pid in "${agent_pids[@]}"; do
        wait "$pid" || {
            log "One or more wandb agents failed (PID=$pid)"
            cleanup
            exit 1
        }
    done

    log "All wandb agents are done, stopping resource monitor."

    if [[ -n "${monitor_pid:-}" ]]; then
        log "Killing resource monitor (PID=$monitor_pid)"
        kill $monitor_pid >/dev/null 2>&1 || true
        wait $monitor_pid 2>/dev/null || true
    fi

    # Now function can exit and the script will return to the prompt
}

# Main script

# Setup trap for cleanup
trap cleanup EXIT INT TERM

# Setup directories
setup_directories

# Setup logging
mkdir -p "$PROJECT_ROOT/runs/logs"
LOG_FILE="$PROJECT_ROOT/runs/logs/sweep_$(date +%Y%m%d_%H%M%S).log"
log "Starting sweep script. Logging to $LOG_FILE"

# Check available GPUs
if [ "$NUM_GPUS" -gt "$(nvidia-smi -L | wc -l)" ]; then
    error "Requested $NUM_GPUS GPUs but only $(nvidia-smi -L | wc -l) available"
    exit 1
fi

# Get or create sweep ID
if [ -n "$RESUME_SWEEP" ]; then
    SWEEP_ID="$RESUME_SWEEP"
    log "Resuming sweep: $SWEEP_ID"
else
    config_file="config/sweep/wandb_sweep_config_$([ $USE_DDP -eq 1 ] && echo ddp || echo single).yaml"
    
    log "Creating new sweep from $config_file"
    SWEEP_OUTPUT="$(wandb sweep "$config_file" 2>&1)"
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent.*' | cut -d'/' -f3)
    
    if [ -z "$SWEEP_ID" ]; then
        log "Failed to get sweep ID"
        echo "Full sweep output:"
        echo "$SWEEP_OUTPUT"
        exit 1
    fi
fi

log "Using sweep ID: $SWEEP_ID"

# Create sweep directory
mkdir -p "$PROJECT_ROOT/runs/sweeps/$SWEEP_ID"

# Write sweep status
echo "running" > "$PROJECT_ROOT/runs/sweeps/$SWEEP_ID/status"

# Launch sweep based on mode
if [ $USE_DDP -eq 1 ]; then
    launch_ddp_sweep "$NUM_GPUS" "$SWEEP_ID"
else
    launch_single_gpu_sweep "$NUM_GPUS" "$SWEEP_ID"
fi

log "Sweep $SWEEP_ID completed successfully"
log "Logs available in $PROJECT_ROOT/runs/logs/"
log "To stop the sweep, run: ./scripts/sweep/stop_sweep.sh"