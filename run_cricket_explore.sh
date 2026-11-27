#!/bin/bash
#SBATCH --job-name=jointvae_cricket_explore
#SBATCH --output=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_cricket_explore_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_cricket_explore_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-7
#SBATCH --time=06:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# Exploration of model and pre-processing options. Every line of SETTINGS is "name<TAB>extra
# arguments of train.py"; every line of WINDOWS is "scaffold block group". Each combination is one
# five-fold run with K 3, gamma 10, C_z 40, lambda 0 and dropout 0.1 unless the extra arguments say otherwise. Finished runs
# are skipped. Settings are compared on the validation crickets (explore/summary.py).

set -euo pipefail

CODE_DIR="/mnt/shared/scratch/weiyeli/crop_vae/repos/JOINT_VAE_CRICKET"
SETTINGS="${SETTINGS:-${CODE_DIR}/explore/settings_final.txt}"
WINDOWS="${WINDOWS:-${CODE_DIR}/explore/windows_final.txt}"
RESULT_DIR="${RESULT_DIR:-/mnt/shared/scratch/weiyeli/crop_vae/results_cricket/mirror/jointvae/explore}"
DATA_DIR="/mnt/shared/scratch/zzhang/hawaii_300"                                # read only

PROCS=${PROCS:-4}
N_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
N_WORKERS=$((N_TASKS * PROCS))

source /mnt/apps/users/weiyeli/conda/bin/activate jointvae
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${CODE_DIR}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# All combinations, windows in the outer loop so that the workers get a mix of settings.
COMBOS=()
while read -r SCAFFOLD BLOCK_ID GROUP; do
    [[ -z "${SCAFFOLD}" ]] && continue
    while IFS=$'\t' read -r NAME EXTRA; do
        [[ -z "${NAME}" ]] && continue
        COMBOS+=("${SCAFFOLD}|${BLOCK_ID}|${NAME}|${EXTRA}")
    done < "${SETTINGS}"
done < "${WINDOWS}"

echo "Started: $(date); task=${TASK_ID}; workers=${N_WORKERS}; runs=${#COMBOS[@]}; node=$(hostname)"

# A card that cannot be initialised would fail every run of this task: stop at once instead.
if ! python -c "import torch; torch.zeros(1, device='cuda')" 2> /dev/null; then
    echo "[No usable GPU] node=$(hostname); GPU=${SLURM_JOB_GPUS:-?}; task ${TASK_ID} stops."
    exit 3
fi

run_worker() {
    local WORKER_ID=$1

    for ((I=WORKER_ID; I<${#COMBOS[@]}; I+=N_WORKERS)); do
        IFS='|' read -r SCAFFOLD BLOCK_ID NAME EXTRA <<< "${COMBOS[$I]}"

        local SCAFFOLD_DIR="${DATA_DIR}/scaffold_${SCAFFOLD}"
        local SAVE_DIR="${RESULT_DIR}/${NAME}/scaffold_${SCAFFOLD}/block_${BLOCK_ID}"

        if [[ -s "${SAVE_DIR}/downstream_summary.csv" ]]; then
            continue
        fi

        mkdir -p "${SAVE_DIR}"

        # Two attempts: a card occasionally fails to initialise when several runs start together.
        local ATTEMPT
        for ATTEMPT in 1 2; do
            # shellcheck disable=SC2086
            python -u "${CODE_DIR}/train.py" \
                --out_dir "${SCAFFOLD_DIR}/out_scaffold${SCAFFOLD}_morph_geography" \
                --ld_block_pkl "${SCAFFOLD_DIR}/scaffold_${SCAFFOLD}_merged_ld_blocks.pkl" \
                --save_dir "${SAVE_DIR}" \
                --block_id "${BLOCK_ID}" \
                --use_ext_window \
                --n_categories 3 --gamma 10 --capacity_z 40 --lambda_marginal 0 --dropout 0.1 \
                --device cuda ${EXTRA} > "${SAVE_DIR}/train.log" 2>&1 && break

            echo "[Failed, attempt ${ATTEMPT}] ${SCAFFOLD}:${BLOCK_ID} ${NAME}; see ${SAVE_DIR}/train.log"
            sleep 30
        done

        echo "Finished: $(date); ${SCAFFOLD}:${BLOCK_ID} ${NAME}"
    done
}

for ((P=0; P<PROCS; P++)); do
    run_worker $((TASK_ID * PROCS + P)) &
done

wait

echo "Task ${TASK_ID} completed: $(date)"
