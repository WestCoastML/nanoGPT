#!/bin/bash

# How to run:-
# #Run in single-GPU mode with 5 agents, one on each GPU
# ./scripts/sweep/run_sweep.sh --num-gpus 5

# # Run in DDP mode with 3 GPUs
# ./scripts/sweep/run_sweep.sh --ddp --num-gpus 3

# # To stop the agents, run the following command
# ./scripts/sweep/stop_sweep.sh

# Strict error handling
set -euo pipefail

# Add debug output
set -x

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Parse arguments
USE_DDP=0
NUM_GPUS=1
RESUME_SWEEP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ddp) USE_DDP=1; shift ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --resume) RESUME_SWEEP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Current issue: Doesn't properly handle sweep resumption
if [ -n "$RESUME_SWEEP" ]; then
    SWEEP_ID="$RESUME_SWEEP"
    log "Resuming sweep: $SWEEP_ID"
else
    config_file="config/sweep/wandb_sweep_config_$([ $USE_DDP -eq 1 ] && echo 'ddp' || echo 'single').yaml"
    SWEEP_OUTPUT=$(wandb sweep "$config_file")
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent.*' | cut -d'/' -f3)
fi

# Also add better process management:
cleanup() {
    kill_process_tree() {
        local parent=$1
        for child in $(ps -o pid --no-headers --ppid ${parent}); do
            kill_process_tree ${child}
        done
        kill ${parent} 2>/dev/null
    }
    
    for pid_file in "$PROJECT_ROOT/runs/sweep_pids"/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            kill_process_tree $pid
            rm "$pid_file"
        fi
    done
}
trap cleanup EXIT

# cleanup() {
#     local exit_code=$?
#     log "Cleaning up processes..."
    
#     if [ -d "$PROJECT_ROOT/runs/sweep_pids" ]; then
#         # Kill process group instead of individual processes
#         for pid_file in "$PROJECT_ROOT/runs/sweep_pids"/*.pid; do
#             if [ -f "$pid_file" ]; then
#                 pid=$(cat "$pid_file")
#                 pgid=$(ps -o pgid= $pid | grep -o '[0-9]*')
#                 if [ ! -z "$pgid" ]; then
#                     log "Stopping process group $pgid"
#                     kill -TERM -$pgid 2>/dev/null || true
#                 fi
#                 rm "$pid_file"
#             fi
#         done
#     fi
    
#     # Wait for processes to terminate
#     wait 2>/dev/null || true
#     exit $exit_code
# }

monitor_resources() {
    local sweep_id=$1
    local monitor_log="$PROJECT_ROOT/runs/logs/resources_${sweep_id}.csv"
    
    echo "timestamp,gpu_util,gpu_mem,cpu_util,ram_util" > "$monitor_log"
    
    while true; do
        timestamp=$(date +%s)
        gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0,0")
        cpu_util=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
        ram_util=$(free | grep Mem | awk '{print $3/$2 * 100}')
        
        echo "$timestamp,$gpu_stats,$cpu_util,$ram_util" >> "$monitor_log"
        sleep 60
    done
}

setup_directories() {
    mkdir -p "$PROJECT_ROOT/runs/sweeps"
    mkdir -p "$PROJECT_ROOT/runs/sweep_pids"
    mkdir -p "$PROJECT_ROOT/runs/logs"
    mkdir -p "$PROJECT_ROOT/runs/checkpoints"
}

launch_ddp_sweep() {
    local num_gpus=$1
    local sweep_id=$2
    
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
    WANDB_SWEEP_ID=$sweep_id \
    WANDB_RUN_DIR="$PROJECT_ROOT/runs/sweeps/$sweep_id/ddp_agent" \
    bash "$PROJECT_ROOT/runs/sweep_pids/ddp_agent_wrapper.sh" $sweep_id $num_gpus &
    
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
    
    # Start resource monitoring
    monitor_resources "$sweep_id" &
    monitor_pid=$!
    echo $monitor_pid > "$PROJECT_ROOT/runs/sweep_pids/monitor_${sweep_id}.pid"
    
    for gpu in $(seq 0 $((num_gpus-1))); do
        export CUDA_VISIBLE_DEVICES=$gpu
        export WANDB_RUN_DIR="$PROJECT_ROOT/runs/sweeps/$sweep_id/agent_${gpu}"
        export WANDB_AGENT_DISABLE_FLAPPING=true
        export WANDB_SWEEP_ID=$sweep_id
        
        wandb agent "wcml/VSLM/$sweep_id" \
            2>&1 | tee "$PROJECT_ROOT/runs/logs/sweep_${sweep_id}_gpu${gpu}.log" &
        
        agent_pid=$!
        echo $agent_pid > "$PROJECT_ROOT/runs/sweep_pids/agent_gpu${gpu}_${sweep_id}.pid"
    done
    
    # Wait for all agents
    wait || {
        log "One or more agents failed"
        cleanup
        exit 1
    }
}

# Main script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Setup trap for cleanup
trap cleanup EXIT INT TERM

# Setup directories
setup_directories

# Get or create sweep ID
if [ -n "$RESUME_SWEEP" ]; then
    SWEEP_ID="$RESUME_SWEEP"
    log "Resuming sweep: $SWEEP_ID"
else
    config_file="config/sweep/wandb_sweep_config_$([ $USE_DDP -eq 1 ] && echo 'ddp' || echo 'single').yaml"
    
    log "Creating new sweep from $config_file"
    SWEEP_OUTPUT=$(wandb sweep "$config_file" 2>&1)
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent.*' | cut -d'/' -f3)
    
    if [ -z "$SWEEP_ID" ]; then
        log "Failed to get sweep ID"
        echo "Full sweep output:"
        echo "$SWEEP_OUTPUT"
        exit 1
    fi
fi

# Create sweep directory
mkdir -p "$PROJECT_ROOT/runs/sweeps/$SWEEP_ID"

# Launch sweep based on mode
if [ $USE_DDP -eq 1 ]; then
    launch_ddp_sweep "$NUM_GPUS" "$SWEEP_ID"
else
    launch_single_gpu_sweep "$NUM_GPUS" "$SWEEP_ID"
fi

log "Sweep $SWEEP_ID completed successfully"
log "Logs available in $PROJECT_ROOT/runs/logs/"