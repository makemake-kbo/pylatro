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
run_name="${PYLATRO_RUN_NAME:-ppo_v8_conditional_survival}"
checkpoint_dir="${workspace_root}/checkpoints/${run_name}"
log_dir="${workspace_root}/runs/${run_name}"
source_checkpoint="${PYLATRO_SOURCE_CHECKPOINT:-${workspace_root}/checkpoints/supervised/supervised_epoch10.pt}"
source_sha256="${PYLATRO_SOURCE_SHA256:?set PYLATRO_SOURCE_SHA256 to the v8 supervised checkpoint SHA256}"
source_sha256="${source_sha256,,}"
win_ante="${PYLATRO_WIN_ANTE:-4}"
recipe_id="pylatro-v8-conditional-survival-v1"
if [[ ! "${win_ante}" =~ ^[1-8]$ ]]; then
  echo "PYLATRO_WIN_ANTE must be an integer from 1 through 8 (got ${win_ante@Q})." >&2
  exit 2
fi

cd "${repo_dir}"
mkdir -p "${checkpoint_dir}" "${log_dir}"
exec > >(tee -a "${workspace_root}/training-v8-conditional-survival.log") 2>&1

run_marker="${checkpoint_dir}/.pylatro-v8-run"
if [[ ! -f "${run_marker}" ]] && compgen -G "${checkpoint_dir}/*.pt" >/dev/null; then
  echo "Refusing to use an unmarked non-empty checkpoint directory; v8 requires a fresh or owned run." >&2
  exit 2
fi

init_args=(
  --pretrained "${source_checkpoint}"
  --updates 2000
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
  echo "v8 checkpoint provenance and ownership validation passed."
  exit 0
fi

exec "${python_bin}" train.py ppo \
  "${init_args[@]}" \
  --envs 16 \
  --rollout-length 256 \
  --batch 320 \
  --micro-batch-size 160 \
  --ppo-epochs 4 \
  --device cuda \
  --win-ante "${win_ante}" \
  --gamma 0.997 \
  --lr 3e-6 \
  --clip-eps 0.1 \
  --outcome-loss-coeff 0.10 \
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
