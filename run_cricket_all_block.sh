#!/bin/bash
#SBATCH --job-name=jointvae_all_blocks
#SBATCH --output=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_all_blocks_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_all_blocks_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-3
#SBATCH --time=16:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# One array task = one GPU that runs PROCS windows at a time (a window is small, so one process
# leaves most of the GPU idle). The request is one core and about 3 GB per process: change
# --cpus-per-task and --mem together with PROCS.

set -euo pipefail

# Configuration
PROCS=4
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_WORKERS=$((N_TASKS * PROCS))

N_CATEGORIES=3
GAMMA=10
CAPACITY_Z=40
LAMBDA_MARGINAL=0
DROPOUT=0.1

MAX_EPOCHS=400
BATCH_SIZE=32
SEED=1024

DATA_DIR="/mnt/shared/scratch/zzhang/hawaii_300"                                # read only
CODE_DIR="/mnt/shared/scratch/weiyeli/crop_vae/repos/JOINT_VAE_CRICKET"
RESULT_DIR="/mnt/shared/scratch/weiyeli/crop_vae/results_cricket/mirror/jointvae"
PYTHON_SCRIPT="${CODE_DIR}/train.py"

# Environment
source /mnt/apps/users/weiyeli/conda/bin/activate jointvae
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${CODE_DIR}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_NAME="extended_K${N_CATEGORIES}_gamma${GAMMA}_cz${CAPACITY_Z}_lambda${LAMBDA_MARGINAL}"
RUN_NAME="${RUN_NAME}_dropout${DROPOUT}_epochs${MAX_EPOCHS}_batch${BATCH_SIZE}_seed${SEED}_allblocks"

echo "Started: $(date); task=${TASK_ID}; workers=${N_WORKERS}; run=${RUN_NAME}"

run_worker() {
    local WORKER_ID=$1

    # Visit scaffolds in order.
    for SCAFFOLD in {1..14}; do

        local SCAFFOLD_DIR="${DATA_DIR}/scaffold_${SCAFFOLD}"
        local INPUT_DIR="${SCAFFOLD_DIR}/out_scaffold${SCAFFOLD}_morph_geography"
        local LD_BLOCK_PKL="${SCAFFOLD_DIR}/scaffold_${SCAFFOLD}_merged_ld_blocks.pkl"

        local N_BLOCKS
        N_BLOCKS=$(python -c '
import pickle
import sys

with open(sys.argv[1], "rb") as handle:
    print(len(pickle.load(handle)["merged_blocks"]))
' "${LD_BLOCK_PKL}")

        # Worker 0 handles blocks 0, N_WORKERS, 2 N_WORKERS, ...
        for ((BLOCK_ID=WORKER_ID; BLOCK_ID<N_BLOCKS; BLOCK_ID+=N_WORKERS)); do

            local SAVE_DIR="${RESULT_DIR}/scaffold_${SCAFFOLD}/outputs_jointvae_scaffold${SCAFFOLD}_${RUN_NAME}/block_${BLOCK_ID}"

            # Skip completed blocks.
            if [[ -s "${SAVE_DIR}/downstream_summary.csv" ]]; then
                continue
            fi

            mkdir -p "${SAVE_DIR}"

            # Train all five outer folds for this block.
            python -u "${PYTHON_SCRIPT}" \
                --out_dir "${INPUT_DIR}" \
                --ld_block_pkl "${LD_BLOCK_PKL}" \
                --save_dir "${SAVE_DIR}" \
                --block_id "${BLOCK_ID}" \
                --use_ext_window \
                --n_splits 5 \
                --inner_val_fraction 0.125 \
                --n_categories "${N_CATEGORIES}" \
                --gamma "${GAMMA}" \
                --capacity_z "${CAPACITY_Z}" \
                --lambda_marginal "${LAMBDA_MARGINAL}" \
                --dropout "${DROPOUT}" \
                --max_epochs "${MAX_EPOCHS}" \
                --batch_size "${BATCH_SIZE}" \
                --seed "${SEED}" \
                --device cuda > "${SAVE_DIR}/train.log" 2>&1 \
                || echo "[Failed] scaffold=${SCAFFOLD}; block=${BLOCK_ID}; see ${SAVE_DIR}/train.log"

            echo "Finished block: $(date); scaffold=${SCAFFOLD}; block=${BLOCK_ID}"

        done
    done
}

for ((P=0; P<PROCS; P++)); do
    run_worker $((TASK_ID * PROCS + P)) &
done

wait

echo "Task ${TASK_ID} completed: $(date)"
