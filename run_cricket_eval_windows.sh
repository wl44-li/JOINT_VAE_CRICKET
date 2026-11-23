#!/bin/bash
#SBATCH --job-name=jointvae_cricket_eval
#SBATCH --output=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_cricket_eval_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_cricket_eval_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --array=0-1
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# Frozen settings on the 32 evaluation windows. The settings were chosen on the 12 tuning windows
# (validation crickets only); no evaluation window was used for the choice.

set -euo pipefail

WINDOWS=("1 133" "1 136" "1 138" "2 202" "5 96" "8 124" "2 330" "2 374" "2 466" "2 470" "10 176" "14 0" "2 239" "11 71" "1 49" "2 303" "2 337" "3 200" "4 11" "4 176" "5 105" "6 49" "6 58" "7 11" "8 98" "8 141" "10 43" "10 161" "12 35" "1 180" "2 325" "14 39")

K_VALUES=(3)
GAMMA_VALUES=(10)
CAPACITY_VALUES=(40)
LAMBDA_VALUES=(0)
DROPOUT_VALUES=(0.1)

RESULT_DIR="${RESULT_DIR:-/mnt/shared/scratch/weiyeli/crop_vae/results_cricket/mirror/jointvae/evaluation}"

source /mnt/shared/scratch/weiyeli/crop_vae/repos/JOINT_VAE_CRICKET/ablation_common.sh
