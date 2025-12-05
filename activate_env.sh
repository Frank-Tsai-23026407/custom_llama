#!/bin/bash
# Activate llama-env conda environment for this project
# Usage: source activate_env.sh

# Initialize conda for bash shell if needed
if ! command -v conda &> /dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

# Activate the environment
conda activate llama-env

echo "✓ Activated llama-env"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"