#!/bin/bash
# scripts/quick_test.sh

# Exit on error
set -e

echo "Running quick functionality test..."

# Create test directory
TEST_DIR=$(mktemp -d)
trap 'rm -rf "$TEST_DIR"' EXIT

# 1. Run a small sweep
echo "1. Testing sweep initialization..."
./scripts/sweep/run_sweep.sh --num-gpus 1 --max-iters 10 &
SWEEP_PID=$!
sleep 10
./scripts/sweep/stop_sweep.sh

# 2. Get latest run ID
LATEST_RUN=$(ls sweep_outputs/run_* | tail -n 1)
RUN_ID=$(basename $LATEST_RUN | cut -d'_' -f2-)

# 3. Test run continuation
echo "2. Testing run continuation..."
python -m utils.continue_run $RUN_ID --max-iters 20 --gpu 0

# 4. Run all tests
echo "3. Running pytest suite..."
pytest tests/

echo "All quick tests completed!"