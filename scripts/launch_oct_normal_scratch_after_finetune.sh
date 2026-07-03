#!/usr/bin/env bash
set -euo pipefail

FINETUNE_PID="${1:?usage: launch_oct_normal_scratch_after_finetune.sh FINETUNE_PID}"
ROOT_DIR="/mnt/nas/media/ubuntu/Thesis/maisi-v2/code/NV-Generate-CTMR"
LOG_DIR="${ROOT_DIR}/outputs/oct_diffusion_256_normal_scratch"
MODEL_DIR="${ROOT_DIR}/models/oct_diffusion_256_normal_scratch"
LOG_PATH="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

while kill -0 "${FINETUNE_PID}" 2>/dev/null; do
  sleep 60
done

cd "${ROOT_DIR}"
source "/home/user/.venvs/nv-generate-ctmr/bin/activate"

python -m scripts.train_oct_diffusion \
  --latents-dir "outputs/oct_latents_256_adv_overnight" \
  --vae-checkpoint "models/oct_vae_256_adv_overnight/autoencoder_oct_256_best.pt" \
  --network-config "configs/config_network_oct_rflow.json" \
  --train-config "configs/config_maisi_diffusion_oct_256_normal_scratch.json" \
  --label-filter 4 \
  --model-dir "models/oct_diffusion_256_normal_scratch" \
  --output-dir "outputs/oct_diffusion_256_normal_scratch" \
  --wandb \
  --wandb-project "oct-maisi" \
  --wandb-name "DIFF-256-normal-scratch-bs96-300ep-pat12" \
  > "${LOG_PATH}" 2>&1
