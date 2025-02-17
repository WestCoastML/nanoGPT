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
debug=""
EXTRA_ARGS=()

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
RESUME_RUN=""
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
        --resume-run)
            RESUME_RUN="$2"
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
        --)
            shift
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

[[ "$DEBUG_MODE" -eq 1 ]] && set -x

# Possibly pass --debug to Hydra or set debug=True
if [[ "$DEBUG_MODE" -eq 1 ]]; then
    debug=True
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

construct_wandb_name() {
    local sweep_id=$1
    local run_id=$2
    local prefix=$3
    
    if [ -n "$prefix" ]; then
        echo "${prefix}_${sweep_id}_${run_id}"
    else
        echo "${sweep_id}_${run_id}"
    fi
}

# Resume run or sweep setup
if [ -n "$RESUME_RUN" ]; then
    # Remove any extraneous quotes
    RESUME_RUN=$(echo "$RESUME_RUN" | tr -d '"')
    log "Requested --resume-run for a finished run: $RESUME_RUN"
    CHECKPOINT_PATH="runs/sweeps/${RESUME_RUN}/checkpoints/latest.pt"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        error "Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi

    # Extract the bare run id and the sweep id from RESUME_RUN (format: SWEEP_ID/RUN_ID)
    RUN_ID_BASENAME=$(basename "$RESUME_RUN")
    SWEEP_ID=$(echo "$RESUME_RUN" | cut -d'/' -f1)

    # Check if the run exists in WandB using the bare run id.
    RUN_EXISTS=$(python -c "import wandb; api=wandb.Api(); print(api.run('wcml/VSLM/${RUN_ID_BASENAME}') is not None)" 2>/dev/null || echo "False")

    if [ "$RUN_EXISTS" != "True" ]; then
        log "Run ${RUN_ID_BASENAME} not found in WandB. The run has finished. Forking a new run..."

        # Extract configuration parameters from Hydra (config/hydra/train.yaml)
        MODEL_ARCHITECTURE=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.model_architecture)")
        LEARNING_RATE=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.learning_rate)")
        BATCH_SIZE=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.batch_size)")
        BASE_DIM=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.base_dim)")

        FORK_RUN_ID="fork_run_${SWEEP_ID}_$(date +%Y%m%d_%H%M%S)"
        unset WANDB_SWEEP_ID
        export WANDB_RUN_ID="$FORK_RUN_ID"
        NEW_OUT_DIR="runs/sweeps/${SWEEP_ID}/${FORK_RUN_ID}"
        log "Fork run output directory set to $NEW_OUT_DIR"

        if [ -n "$debug" ]; then
            DEBUG_ARG="+debug=$debug"
        else
            DEBUG_ARG=""
        fi

        # Convert EXTRA_ARGS from '--key=value' to Hydra override format (key=value)
        converted_args=()
        for arg in "${EXTRA_ARGS[@]}"; do
          if [[ $arg == --* ]]; then
             converted_args+=( "${arg:2}" )
          else
             converted_args+=( "$arg" )
          fi
        done

        python "$PROJECT_ROOT/train.py" \
            init_from=resume \
            wandb_run_name="'sweepid=${SWEEP_ID}_runid=${RUN_ID_BASENAME}_model=${MODEL_ARCHITECTURE}_lr=${LEARNING_RATE}_bs=${BATCH_SIZE}_dim=${BASE_DIM}'" \
            wandb_log=true \
            out_dir="$NEW_OUT_DIR" \
            +resume_checkpoint="$CHECKPOINT_PATH" \
            "${converted_args[@]}" || {
                error "Failed to fork new run from $RESUME_RUN"
                exit 1
            }
        exit 0
    else
        export WANDB_RUN_ID="$RUN_ID_BASENAME"
        if [ -n "$debug" ]; then
            DEBUG_ARG="+debug=$debug"
        else
            DEBUG_ARG=""
        fi
        python -m utils.continue_run --run_id "$RUN_ID_BASENAME" --checkpoint "$CHECKPOINT_PATH" ${DEBUG_ARG} "${EXTRA_ARGS[@]}" || {
            error "Failed to resume run $RESUME_RUN"
            exit 1
        }
        log "Resumed run launched. Exiting script now..."
        exit 0
    fi
elif [ -n "$RESUME_SWEEP" ]; then
    log "Requested --resume for a previously completed sweep: $RESUME_SWEEP"
    log "In Weights & Biases, once a sweep is completed, you cannot attach new runs to that exact ID."
    log "We will fork a new run that loads from the old sweeps checkpoint instead."

    # Extract configuration parameters from Hydra
    MODEL_ARCHITECTURE=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.model_architecture)")
    LEARNING_RATE=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.learning_rate)")
    BATCH_SIZE=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.batch_size)")
    BASE_DIM=$(python -c "from omegaconf import OmegaConf; cfg=OmegaConf.load('config/hydra/train.yaml'); print(cfg.base_dim)")

    FORK_SWEEP_ID="fork_of_${RESUME_SWEEP}_$(date +%Y%m%d_%H%M%S)"
    log "Fork sweep ID => $FORK_SWEEP_ID"

    # Unset old sweep ID so we do not call wandb agent on the finished sweep
    unset WANDB_SWEEP_ID
    SWEEP_ID="$FORK_SWEEP_ID"

    # Also define a new run ID, so Hydra's fallback "upl53903" won't appear.
    # We'll incorporate the old sweep ID in the new run ID for clarity.
    FORK_RUN_ID="fork_run_${RESUME_SWEEP}_$(date +%Y%m%d_%H%M%S)"
    unset WANDB_SWEEP_ID
    export WANDB_RUN_ID="$FORK_RUN_ID"
    NEW_OUT_DIR="runs/sweeps/${RESUME_SWEEP}/${FORK_RUN_ID}"
    log "Fork run output directory set to $NEW_OUT_DIR"
    
    # Validate checkpoint path exists
    CHECKPOINT_PATH="runs/sweeps/${RESUME_SWEEP}/checkpoints/latest.pt"
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        error "Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi
    
    unset WANDB_SWEEP_ID      # remove the environment var so W&B doesn't try to attach to an old sweep
    export WANDB_RUN_ID="$FORK_RUN_ID"

    # Example: directly run your training script in resume mode
    # Note: Using ++ for wandb_run_name to override existing config
    python "$PROJECT_ROOT/train.py" \
        init_from=resume \
        wandb_run_name="'sweepid=${RESUME_SWEEP}_runid=${FORK_RUN_ID}_model=${MODEL_ARCHITECTURE}_lr=${LEARNING_RATE}_bs=${BATCH_SIZE}_dim=${BASE_DIM}'" \
        wandb_log=true \
        out_dir="$NEW_OUT_DIR" \
        +debug=$debug \
        +resume_checkpoint="runs/sweeps/${RESUME_SWEEP}/checkpoints/latest.pt" \
        "${EXTRA_ARGS[@]}" \
        || {
            local exit_code=$?
            error "Failed to fork new run from $RESUME_SWEEP"
            exit 1
        }

    log "Fork run launched. Exiting script now..."
    exit 0
else
    # Create a new sweep
    config_file="config/sweep/wandb_sweep_config_$([ $USE_DDP -eq 1 ] && echo ddp || echo single).yaml"
    log "Creating new sweep from $config_file"
    echo "[DEBUG] Checking environment in run_sweep.sh before creating new sweep..."
    echo "[DEBUG] which python => $(which python)"
    echo "[DEBUG] python executable => $(python -c "import sys; print(sys.executable)")"
    pip show wandb >/dev/null 2>&1 || echo "[DEBUG] wandb not found in this environment"
    python -c "import sweeps; print(f'[DEBUG] sweeps version is {sweeps.__version__}')" 2>/dev/null || echo "[DEBUG] sweeps not found"
    
    if pip show wandb >/dev/null 2>&1; then
        echo "[DEBUG] wandb version is $(pip show wandb | grep Version | awk '{print $2}')"
    else
        echo "[DEBUG] wandb not found"
    fi

    SWEEP_OUTPUT="$(wandb sweep "$config_file")" || {
        error "Failed to create sweep"
        exit 1
    }
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
