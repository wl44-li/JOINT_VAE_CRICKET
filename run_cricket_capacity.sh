#!/bin/bash
#SBATCH --job-name=jointvae_cricket_capacity
#SBATCH --output=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_cricket_capacity_%A_%a.out
#SBATCH --error=/mnt/shared/scratch/weiyeli/crop_vae/logs/jointvae_cricket_capacity_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --array=0-3
#SBATCH --time=08:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4

# Weight and final capacity of the Gaussian latent.
# Runs on the 12 tuning windows; see ablation_common.sh.

set -euo pipefail

K_VALUES=(3)
GAMMA_VALUES=(300 1000 3000)
CAPACITY_VALUES=(0.25 1 2.3 5 10 20 40)
LAMBDA_VALUES=(0)
DROPOUT_VALUES=(0.1 0.5)

source /mnt/shared/scratch/weiyeli/crop_vae/repos/JOINT_VAE_CRICKET/ablation_common.sh
