#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

workspace_root="${PYLATRO_WORKSPACE:-/workspace}"
repo_dir="${workspace_root}/pylatro"
run_name="${PYLATRO_RUN_NAME:-ppo_strategy_v13_tarot_seal_bestv12u280}"
checkpoint_dir="${workspace_root}/checkpoints/${run_name}"
log_dir="${workspace_root}/runs/${run_name}"
source_checkpoint="${PYLATRO_SOURCE_CHECKPOINT:-${workspace_root}/checkpoints/ppo_strategy_v12_structural_fixes_bestv10/ppo_best_eval.pt}"

cd "${repo_dir}"
mkdir -p "${checkpoint_dir}" "${log_dir}"
exec > >(tee -a "${workspace_root}/training-strategy-v13-tarot-seal.log") 2>&1

init_args=(
  --pretrained "${source_checkpoint}"
  --updates 2000
  --reinit-value-head
  --critic-warmup-updates 15
  --critic-warmup-min-ev 0.4
)
if [[ -f "${checkpoint_dir}/ppo_latest.pt" ]]; then
  init_args=(--resume "${checkpoint_dir}/ppo_latest.pt" --updates 2000)
fi

exec /venv/main/bin/python train.py ppo \
  "${init_args[@]}" \
  --envs 16 \
  --rollout-length 256 \
  --batch 352 \
  --ppo-epochs 4 \
  --device cuda \
  --win-ante 4 \
  --hl-gauss \
  --gamma 0.997 \
  --lr 1e-5 \
  --clip-eps 0.1 \
  --target-kl 0.03 \
  --target-kl-p95 0.10 \
  --target-kl-max 0.15 \
  --min-minibatch-fraction 0.5 \
  --dense-reward-scale 0.25 \
  --strategic-event-reward-scale 1.0 \
  --score-build-potential \
  --planet-match-shaping \
  --planet-unmatched-use-penalty-coeff 0.25 \
  --planet-unmatched-claim-penalty-coeff 0.10 \
  --entropy-coeff 0.01 \
  --action-type-entropy-scale 0.25 \
  --danger-rollout-temperature 0.85 \
  --danger-death-probability-threshold 0.35 \
  --danger-shop-leave-logit-penalty 2.0 \
  --max-idle-steps 32 \
  --eval-games 100 \
  --eval-interval 10 \
  --eval-regression-tolerance 0.10 \
  --eval-regression-patience 2 \
  --checkpoint-interval 100 \
  --log-interval 5 \
  --checkpoint-dir "${checkpoint_dir}" \
  --log-dir "${log_dir}"
