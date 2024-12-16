#!/bin/bash

# Set PYTHONPATH to current directory
export PYTHONPATH=$(pwd)

# Run all tests
pytest tests/
