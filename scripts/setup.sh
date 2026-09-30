#!/bin/bash
# One-time project bootstrap: create the working directories and write .env.
# Nothing in the repository composes before this, since configs/paths/default.yaml
# reads ${oc.env:PROJECT_ROOT}. To keep the heavy directories on another
# filesystem, edit RES_DIR in .env afterwards and run scripts/symbolink.sh.
#
#   bash scripts/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# There is no checkpoints/: checkpoints live under each run's own output dir.
for folder in data logs outputs results; do
    if [ -d "$folder" ]; then
        echo "Folder already exists: $folder"
    else
        mkdir -p "$folder"
        echo "Created folder: $folder"
    fi
done

# Never replace an existing .env: it may already point RES_DIR at a scratch
# filesystem, and the next run would fill the home directory with checkpoints.
env_file=".env"
if [ -e "$env_file" ]; then
    echo
    echo "$PWD/$env_file already exists - left untouched."
    echo "Delete it first if you want a fresh one."
    exit 0
fi

# MLFLOW_ALLOW_FILE_STORE: mlflow 3 refuses the file: tracking store that
# configs/logger/mlflow.yaml points at without it. HF_HUB_CACHE holds both record
# snapshots, the jets one 5.7 GB and MNIST 30 MB (huggingface_hub reads it at
# import; the jets loader's normalised per-nconstituents caches are separate and
# land in data/hf/nconst_<N>/, the decoded MNIST arrays in data/mnist/). On a
# machine with a hub cache already, point it there rather than downloading twice.
cat > "$env_file" << EOL
PROJECT_ROOT="."
RES_DIR="." # set to the desired location, then run scripts/symbolink.sh
DATA_DIR="\${RES_DIR}/data"
LOG_DIR="\${RES_DIR}/logs"
OUTPUT_DIR="\${RES_DIR}/outputs"
RESULTS_DIR="\${RES_DIR}/results"
HF_HUB_CACHE="\${RES_DIR}/data/hf_hub"
MLFLOW_ALLOW_FILE_STORE="true"
EOL

echo
echo ".env file created successfully at $PWD/$env_file"
