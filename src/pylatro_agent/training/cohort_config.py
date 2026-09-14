"""Validation and strict resume contract for the complete-cohort experiment."""

from __future__ import annotations

from dataclasses import asdict

COHORT_VERSION = 1
COHORT_FIELDS = (
    "collection_mode",
    "cohort_games",
    "cohort_training_rows",
    "cohort_max_episode_steps",
    "cohort_workers",
    "cohort_seeds",
    "outcome_label_mode",
)
# These controls affect execution/reporting, not the next policy update.
RESUME_OVERRIDES = {
    "total_updates",
    "total_timesteps",
    "save_dir",
    "log_dir",
    "stop_request_path",
    "checkpoint_interval",
    "log_interval",
    "eval_interval",
    "eval_before_training",
}


def validate_cohort_config(config):
    if config.collection_mode not in ("fixed_steps", "complete_cohort"):
        raise ValueError("Unknown collection_mode")
    if config.outcome_label_mode not in ("completion_window", "complete_episode"):
        raise ValueError("Unknown outcome_label_mode")
    if config.collection_mode == "fixed_steps":
        if config.outcome_label_mode != "completion_window":
            raise ValueError("complete_episode labels require complete_cohort collection")
        return
    for key in ("num_envs", "cohort_games", "cohort_training_rows", "cohort_max_episode_steps", "total_updates"):
        value = getattr(config, key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"Complete-cohort {key} must be an explicit positive integer")
    if config.cohort_games % config.num_envs:
        raise ValueError("cohort_games must be divisible by num_envs")
    if type(config.cohort_workers) is not int or config.cohort_workers < 0:
        raise ValueError("cohort_workers must be nonnegative")
    seeds = config.cohort_seeds
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Complete-cohort collection requires distinct explicit cohort_seeds")
    from .ppo_evaluation import training_seed_reservations

    if set(seeds) & set(training_seed_reservations(config)):
        raise ValueError("Cohort training/evaluation seed overlap")
    if config.terminal_replay_updates_per_ppo_update or config.sil_coeff or config.return_path_coeff:
        raise ValueError("Complete-cohort experiments require replay, SIL and return-path imitation disabled")
    if config.archive_config is not None or config.counterfactual_diagnostic_interval:
        raise ValueError("Complete-cohort experiments do not support archives or scoring interventions")
    reward = config.reward_config
    if reward is None or reward.objective != "milestone" or reward.milestone_reward_budget != 0 or config.gamma != 1:
        raise ValueError("Complete-cohort v1 requires terminal-only milestone rewards and gamma=1")
    if config.milestone_final_scale != 0 or config.adaptive_entropy:
        raise ValueError("Complete-cohort v1 requires fixed reward and entropy settings")
    if config.precision not in (None, "fp32") or config.device not in ("cpu", "mps"):
        raise ValueError("Complete-cohort v1 supports FP32 on CPU or MPS")
    if config.eval_regression_tolerance is not None:
        raise ValueError("Use the matched cohort study endpoints instead of an in-training regression stop")


def validate_cohort_resume(config, payload):
    saved_mode = payload.get("ppo_config_fields", {}).get("collection_mode", "fixed_steps")
    if saved_mode != config.collection_mode:
        raise ValueError("Collection mode changed; create an explicit provenance-recorded cohort branch")
    if saved_mode == "fixed_steps":
        return
    saved = payload.get("cohort_ppo_config")
    state = payload.get("cohort_state")
    if not isinstance(saved, dict) or not isinstance(state, dict) or state.get("version") != COHORT_VERSION:
        raise ValueError("Missing or incompatible complete-cohort resume metadata")
    active = asdict(config)
    if set(saved) != set(active):
        raise ValueError("Incompatible complete-cohort configuration schema")
    changed = [key for key in active if key not in RESUME_OVERRIDES and active[key] != saved[key]]
    if changed:
        raise ValueError(f"Complete-cohort recipe changed on resume: {changed}")
