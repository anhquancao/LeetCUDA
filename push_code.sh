#!/bin/bash

# --- Configuration ---
LOCAL_PROJECT_DIR="/home/acao/code/LeetCUDA/"
REMOTE_HOST_ALIAS="jean-zay-backup"
REMOTE_PROJECT_DIR="/lustre/fswork/projects/rech/kvd/uyl37fq/code/LeetCUDA/"
# --- End Configuration ---

while true; do
    # Ensure trailing slashes for rsync to copy contents correctly
    LOCAL_PROJECT_DIR_SYNC="${LOCAL_PROJECT_DIR%/}/"
    REMOTE_PROJECT_DIR_SYNC="${REMOTE_PROJECT_DIR%/}/"

    echo ">>> Pushing code changes (excluding slurm log directories) from WSL2 to ${REMOTE_HOST_ALIAS}..."
    echo "    Local source: ${LOCAL_PROJECT_DIR_SYNC}"
    echo "    Remote destination: ${REMOTE_HOST_ALIAS}:${REMOTE_PROJECT_DIR_SYNC}"

    # Exclude all specific slurm log directories
    # Using --exclude for each one.
    # Note: 'slurm/' by itself would exclude the entire slurm directory,
    # which might be too broad if you have other non-log things in 'slurm/'.
    # If 'slurm/' ONLY contains these log outputs, then --exclude 'slurm/' is simpler.
    # Assuming 'slurm/' might have other things, we'll be specific.
    rsync -avzP --delete \
        --exclude '__pycache__/' \
        --exclude '*/build/' \
        --exclude '*.egg-info/' \
        --exclude '*.egg-info' \
        --exclude '.git/' \
        "${LOCAL_PROJECT_DIR_SYNC}" \
        "${REMOTE_HOST_ALIAS}:${REMOTE_PROJECT_DIR_SYNC}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ">>> Code push complete."
    else
        echo ">>> ERROR: Code push failed with exit code ${EXIT_CODE}."
    fi

    echo "Waiting for 3 seconds before next push..."
    sleep 3
done