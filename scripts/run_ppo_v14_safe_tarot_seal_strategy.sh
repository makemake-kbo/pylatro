#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

workspace_root="${PYLATRO_WORKSPACE:-/workspace}"
repo_dir="${PYLATRO_REPO_DIR:-${workspace_root}/pylatro}"
python_bin="${PYLATRO_PYTHON:-/venv/main/bin/python}"
run_name="${PYLATRO_RUN_NAME:-ppo_strategy_v14_safe_tarot_seal_bestv12u280_criticlr1e5_batch320}"
checkpoint_dir="${workspace_root}/checkpoints/${run_name}"
log_dir="${workspace_root}/runs/${run_name}"
source_checkpoint="${PYLATRO_SOURCE_CHECKPOINT:-${workspace_root}/checkpoints/ppo_strategy_v12_structural_fixes_bestv10/ppo_best_eval.pt}"
source_sha256="${PYLATRO_SOURCE_SHA256:-5a81807c38213d0aadd5d984158652daafd259df0e416d70097698dd75cc0660}"
source_sha256="${source_sha256,,}"
recipe_id="pylatro-v14-safe-v3"
run_name_lower="${run_name,,}"
source_checkpoint_lower="${source_checkpoint,,}"

# v14 may resume only its own directory. Never seed or resume from the
# regressed v13 policy, even through an accidental environment override.
if [[ "${run_name_lower}" == *v13* || "${source_checkpoint_lower}" == *v13* ]]; then
  echo "Refusing to start v14 from a v13 run/checkpoint." >&2
  exit 2
fi

cd "${repo_dir}"
mkdir -p "${checkpoint_dir}" "${log_dir}"
exec > >(tee -a "${workspace_root}/training-strategy-v14-safe-tarot-seal.log") 2>&1

run_marker="${checkpoint_dir}/.pylatro-v14-safe-run"
if [[ ! -f "${run_marker}" ]] && compgen -G "${checkpoint_dir}/*.pt" >/dev/null; then
  echo "Refusing to use an unmarked non-empty checkpoint directory; v14 requires a fresh or owned run." >&2
  exit 2
fi

init_args=(
  --pretrained "${source_checkpoint}"
  --updates 2000
  --reinit-value-head
)
latest_checkpoint="${checkpoint_dir}/ppo_latest.pt"
if [[ ! -f "${latest_checkpoint}" ]]; then
  "${python_bin}" scripts/validate_ppo_run_checkpoint.py source \
    "${source_checkpoint}" --expected-sha256 "${source_sha256}"
fi

if [[ -f "${run_marker}" ]]; then
  run_uuid="$("${python_bin}" scripts/validate_ppo_run_checkpoint.py marker-validate \
    "${run_marker}" --run-name "${run_name}" --source-sha256 "${source_sha256}" \
    --recipe-id "${recipe_id}")"
else
  # This is the sole UUID generation point and is reachable only after the
  # pinned source hash + metadata validation above.
  run_uuid="$("${python_bin}" scripts/validate_ppo_run_checkpoint.py marker-create \
    "${run_marker}" --run-name "${run_name}" --source-sha256 "${source_sha256}" \
    --recipe-id "${recipe_id}")"
fi

if [[ -f "${latest_checkpoint}" ]]; then
  "${python_bin}" scripts/validate_ppo_run_checkpoint.py resume "${latest_checkpoint}" \
    --run-uuid "${run_uuid}" --source-sha256 "${source_sha256}" --recipe-id "${recipe_id}"
  init_args=(--resume "${latest_checkpoint}" --updates 2000)
fi

if [[ "${PYLATRO_VALIDATE_ONLY:-0}" == "1" ]]; then
  echo "v14 checkpoint provenance and ownership validation passed."
  exit 0
fi

exec "${python_bin}" train.py ppo \
  "${init_args[@]}" \
  --envs 16 \
  --rollout-length 256 \
  --batch 320 \
  --ppo-epochs 4 \
  --device cuda \
  --win-ante 4 \
  --hl-gauss \
  --gamma 0.997 \
  --lr 3e-6 \
  --clip-eps 0.1 \
  --critic-warmup-updates 20 \
  --critic-warmup-lr 1e-5 \
  --critic-warmup-min-ev 0.4 \
  --critic-warmup-ev-window 5 \
  --critic-warmup-max-updates 80 \
  --actor-ramp-updates 25 \
  --actor-ramp-start-clip-fraction 0.5 \
  --ppo-run-uuid "${run_uuid}" \
  --ppo-source-sha256 "${source_sha256}" \
  --ppo-recipe-id "${recipe_id}" \
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
  --eval-regression-tolerance 0.05 \
  --eval-regression-patience 2 \
  --checkpoint-interval 100 \
  --log-interval 5 \
  --checkpoint-dir "${checkpoint_dir}" \
  --log-dir "${log_dir}"
