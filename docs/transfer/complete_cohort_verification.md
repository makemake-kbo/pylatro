# Complete-cohort implementation verification

Verified on 2026-09-14: **103 targeted CPU tests passed**, with zero failures,
errors, or skips. The recorded suite took 16.54 seconds. Ruff checks and
`git diff --check` also passed.

The production collector and optimizer completed two real games with a small CPU
model: 22 collected transitions, one PPO update, and 16 optimizer examples.
Teacher action calls were forbidden. Separate interrupted and uninterrupted CPU
runs produced identical model, Adam, RNG, and cohort state.

Coverage includes completion-selected labels, observation schema and history
preservation, GAE boundaries, frozen-policy probabilities, incomplete-cohort
rejection, worker transport for both decks, lazy microbatch gradients, saved
trajectory compatibility, strict resume, seed separation, paired statistics, and
the evaluation-to-confirmation gates. Existing rollout, checkpoint, deck,
outcome-PPO, and batched-evaluation regressions are included.

Reproduce the bounded verification independently of the large parent checkpoints:

```sh
.venv/bin/python -m pytest \
  tests/test_complete_cohort.py \
  tests/test_complete_cohort_training.py \
  tests/test_complete_cohort_study.py \
  tests/test_rollout_buffer.py \
  tests/test_ppo_resume_checkpoint.py \
  tests/test_outcome_ppo_pipeline.py \
  tests/test_ppo_deck.py \
  tests/test_outcome_branch.py \
  tests/test_batched_eval.py
```

The local `eval/complete_cohort_v1` directory retains the frozen protocol,
verification log, JUnit results, and CPU smoke artifacts. Its verification source
digest is `05a732af69ec2563030c45788fb69ad96b4f37edd453992e675197260f7315c1`;
the JUnit artifact SHA-256 is
`a1fb3f7b8a5172ea152bfcff42590a8d0f9812ad00462482e36127c4edadd9db`.
Generated checkpoints and logs are excluded from this commit.

Four recipes are prepared: completion-window control and complete-label treatment,
each from both pinned update-42 parents. No new accelerator study or real
checkpoint evaluation panel was executed; the accelerator ledger remains at
zero and confirmation has not started. Simulated evaluation results in tests
exercise orchestration only. **Gameplay improvement remains unmeasured.**

See the [implementation, prior-art mapping, and experiment protocol](complete_cohort.md).
