#!/bin/bash
#SBATCH --job-name=plot_jointvae_ablation
#SBATCH --output=/mnt/shared/scratch/weiyeli/crop_vae/logs/plot_jointvae_ablation_%j.out
#SBATCH --error=/mnt/shared/scratch/weiyeli/crop_vae/logs/plot_jointvae_ablation_%j.err
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

set -euo pipefail

CODE_DIR="/mnt/shared/scratch/weiyeli/crop_vae/repos/JOINT_VAE_CRICKET"
RESULTS_ROOT="/mnt/shared/scratch/weiyeli/crop_vae/results_cricket/mirror/jointvae/ablation"

source /mnt/apps/users/weiyeli/conda/bin/activate jointvae
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg

cd "${CODE_DIR}"

# One figure and one table per varied setting; the other settings are the line labels.
for VARY in capacity_z n_categories lambda_marginal; do
    python -u plot_cricket_ablation.py \
        --root "${RESULTS_ROOT}" \
        --output_dir "${RESULTS_ROOT}/plots" \
        --vary "${VARY}"
done
