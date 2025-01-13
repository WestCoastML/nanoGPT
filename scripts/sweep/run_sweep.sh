#!/bin/bash

# How to run:-
# 1) Single-GPU mode with multiple agents: 
#      ./scripts/sweep/run_sweep.sh --num-gpus 5
#
# 2) DDP mode with N GPUs:
#      ./scripts/sweep/run_sweep.sh --ddp --num-gpus 3
#
# 3) Resume an existing sweep:
#      ./scripts/sweep/run_sweep.sh --resume <SWEEP_ID>
#
# 4) To stop the agents:
#      ./scripts/sweep/stop_sweep.sh

# Strict error handling and debugging
set -euo pipefail
set -x

DEBUG_MODE=0

# Determine script and project root directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Setup logging folder and LOG_FILE
mkdir -p "$PROJECT_ROOT/runs/logs"
LOG_FILE="$PROJECT_ROOT/runs/logs/sweep_$(date +%Y%m%d_%H%M%S).log"

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

debug_log() {
    if [ "$DEBUG_MODE" -eq 1 ]; then
        echo "[DEBUG $(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
    fi
}

error() {
    log "ERROR: $*" >&2
}

echo "Starting run_sweep.sh. Logging to $LOG_FILE"

# Parse arguments
USE_DDP=0
NUM_GPUS=1
RESUME_SWEEP=""
DEBUG_MODE=0

# Grab timestamp for e.g. 20250109_130516_<SWEEP_ID>
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

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
        --resume)
            RESUME_SWEEP="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_MODE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

[[ "$DEBUG_MODE" -eq 1 ]] && set -x

# Possibly pass --debug to Hydra or set +debug=True
if [[ "$DEBUG_MODE" -eq 1 ]]; then
    export debug=True
fi

# Re-define log functions after we parse debug
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}
debug_log() {
    if [ "$DEBUG_MODE" -eq 1 ]; then
        echo "[DEBUG $(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
    fi
}

# Create or resume a sweep
if [ -n "$RESUME_SWEEP" ]; then
    echo "[DEBUG] Checking environment in run_sweep.sh before resuming sweep..."
    echo "[DEBUG] which python => $(which python)"
    echo "[DEBUG] python executable => $(python -c "import sys; print(sys.executable)")"
    pip show wandb >/dev/null 2>&1 || echo "[DEBUG] wandb not found in this environment"
    python -c "import sweeps; print(f'[DEBUG] sweeps version is {sweeps.__version__}')" 2>/dev/null \
        || echo "[DEBUG] sweeps not found"
    pip show wandb >/dev/null 2>&1 && \
        echo "[DEBUG] wandb version is $(pip show wandb | grep Version | awk '{print $2}')" || \
        echo "[DEBUG] wandb not found"
    log "Resuming sweep with ID: $RESUME_SWEEP"
    SWEEP_ID="$RESUME_SWEEP"
else
    # Create a new sweep
    config_file="config/sweep/wandb_sweep_config_$([ $USE_DDP -eq 1 ] && echo ddp || echo single).yaml"
    log "Creating new sweep from $config_file"
    echo "[DEBUG] Checking environment in run_sweep.sh before creating new sweep..."
    echo "[DEBUG] which python => $(which python)"
    echo "[DEBUG] python executable => $(python -c "import sys; print(sys.executable)")"
    pip show wandb >/dev/null 2>&1 || echo "[DEBUG] wandb not found in this environment"
    python -c "import sweeps; print(f'[DEBUG] sweeps version is {sweeps.__version__}')" 2>/dev/null \
        || echo "[DEBUG] sweeps not found"
    pip show wandb >/dev/null 2>&1 && \
        echo "[DEBUG] wandb version is $(pip show wandb | grep Version | awk '{print $2}')" || \
        echo "[DEBUG] wandb not found"
    SWEEP_OUTPUT="$(wandb sweep "$config_file" 2>&1)"
    log "Sweep output: $SWEEP_OUTPUT"

    # Extract entire line containing 'wandb agent ...'
    FULL_SWEEP="$(echo "$SWEEP_OUTPUT" | grep -o 'wandb agent.*')"
    # Get the third space-delimited token, e.g. "wcml/VSLM/123abc"
    FULL_SWEEP="$(echo "$FULL_SWEEP" | cut -d' ' -f3)"
    # From that, get the third slash-delimited token, e.g. "123abc"
    SWEEP_ID="$(echo "$FULL_SWEEP" | cut -d'/' -f3)"
    if [ -z "$SWEEP_ID" ]; then
        log "Failed to parse sweep ID from W&B output."
        SWEEP_ID="NoWandbID"
    fi
    log "Using newly created sweep ID: $SWEEP_ID"
fi

# Export WANDB_SWEEP_ID before using it
export WANDB_SWEEP_ID="$SWEEP_ID"
debug_log "Exported WANDB_SWEEP_ID=$WANDB_SWEEP_ID"

log "Using sweep ID: $SWEEP_ID"

# Construct local ID: TIMESTAMP_SWEEPID
LOCAL_SWEEP_ID="${TIMESTAMP}_${SWEEP_ID}"
log "LOCAL_SWEEP_ID: $LOCAL_SWEEP_ID"

FULL_SWEEP="wcml/VSLM/$SWEEP_ID"

# Cleanup function
cleanup() {
    log "Cleanup initiated..."
    # Kill child processes in this group
    pkill -P $$
    # Kill any wandb agents
    pkill -f "wandb agent"
    # Kill any related Python processes
    pkill -f "python.*train.py"
    # Cleanup PID files
    rm -f "$PROJECT_ROOT/runs/sweep_pids"/*.pid
    # Optionally write status
    if [ -n "${SWEEP_ID:-}" ]; then
        echo "stopped" > "$PROJECT_ROOT/runs/sweeps/$SWEEP_ID/status"
    fi
    log "Cleanup completed"
    exit 0
}
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
    mkdir -p "$PROJECT_ROOT/runs/sweeps/$LOCAL_SWEEP_ID"
}

launch_ddp_sweep() {
    local num_gpus=$1
    local wandb_sweep_id=$2
    export LOCAL_SWEEP_ID
    export WANDB_SWEEP_ID="$wandb_sweep_id"
    log "Launching DDP sweep with $num_gpus GPUs"

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
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    monitor_resources "$sweep_id" &
    monitor_pid=$!

    agent_pids=()

    for gpu in $(seq 0 $((num_gpus-1)))
    do
        export CUDA_VISIBLE_DEVICES=$gpu
        debug_log "Launching wandb agent on GPU $gpu with WANDB_SWEEP_ID=$WANDB_SWEEP_ID"

        # Additional debug:
        echo "[DEBUG] GPU index = ${gpu}"
        echo "[DEBUG] Checking environment before calling wandb agent..."
        echo "[DEBUG] which python => $(which python)"
        echo "[DEBUG] python executable => $(python -c 'import sys; print(sys.executable)')"
        pip show wandb >/dev/null 2>&1 || echo "[DEBUG] wandb not found in this environment"
        python -c "import sweeps; print(f'[DEBUG] sweeps version is {sweeps.__version__}')" 2>/dev/null \
            || echo "[DEBUG] sweeps not found"
        pip show wandb >/dev/null 2>&1 && \
            echo "[DEBUG] wandb version is $(pip show wandb | grep Version | awk '{print $2}')" || \
            echo "[DEBUG] wandb not found"
        echo "[DEBUG] Launching wandb agent with FULL_SWEEP=$FULL_SWEEP"
        echo "[DEBUG] PATH=$PATH"
        echo "[DEBUG] PYTHONPATH=${PYTHONPATH:-}"

        [ "$DEBUG_MODE" -eq 1 ] && DEBUG_ARG="--debug" || DEBUG_ARG=""

        export WANDB_RUN_DIR="$PROJECT_ROOT/runs/sweeps/$sweep_id/agent_${gpu}"
        export WANDB_AGENT_DISABLE_FLAPPING=true
        export WANDB_SWEEP_ID=$sweep_id

        export sweep_id=$sweep_id  # Added: Export sweep_id for Hydra interpolation
        export wandb_sweep_id=$sweep_id  # Add explicit lowercase version

        wandb agent "$FULL_SWEEP" 2>&1 | tee "$PROJECT_ROOT/runs/logs/sweep_${sweep_id}_gpu${gpu}.log" &
        agent_pid=$!
        echo "$agent_pid" > "$PROJECT_ROOT/runs/sweep_pids/agent_gpu${gpu}_${sweep_id}.pid"
        agent_pids+=("$agent_pid")
    done

    # Wait for all wandb agents
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
}

# Main script
trap cleanup EXIT INT TERM

setup_directories

mkdir -p "$PROJECT_ROOT/runs/logs"
LOG_FILE="$PROJECT_ROOT/runs/logs/sweep_$(date +%Y%m%d_%H%M%S).log"
log "Starting sweep script. Logging to $LOG_FILE"

# Check GPU availability
if [ "$NUM_GPUS" -gt "$(nvidia-smi -L | wc -l)" ]; then
    error "Requested $NUM_GPUS GPUs but only $(nvidia-smi -L | wc -l) available"
    exit 1
fi

# Re-affirm we have a single SWEEP_ID at this point:
log "Using sweep ID: $SWEEP_ID"
export WANDB_SWEEP_ID="$SWEEP_ID"
debug_log "Exported WANDB_SWEEP_ID=$WANDB_SWEEP_ID"

FULL_SWEEP="wcml/VSLM/$SWEEP_ID"

# Make a directory for the sweep if it doesn't exist
mkdir -p "$PROJECT_ROOT/runs/sweeps/$SWEEP_ID"
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
