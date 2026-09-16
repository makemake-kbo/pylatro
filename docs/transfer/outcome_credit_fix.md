# Completed-outcome credit for win-only PPO

The win-only PPO path can now use one directly supervised value and complete-game actor credit. This implements a consistent objective for the existing `+10 win / 0 loss`, gamma-one task. It does not start or resume training.

The old path adds a learned residual to `10 P(win)` and fits that residual to bootstrapped GAE returns with Huber loss. Those targets can preserve the old critic's errors. Huber also does not generally estimate a conditional mean. The new explicit path removes that objective and its compensation term:

```text
V_old(t) = 10 * P_old(win | observation/history at t)
G(t)     = sum of actual rewards from t through true termination
A_raw(t) = G(t) - V_old(t)
L_critic = outcome_loss_coeff * categorical_NLL(actual_final_outcome)
```

With this reward definition, every preceding decision in a winning game has return target 10; every preceding decision in a losing game has target 0. The collected value remains a frozen pre-update baseline. Future critic predictions, the 256-step logical boundaries, and lambda do not enter these return targets. Existing whole-cohort advantage normalization and PPO clipping remain in place. The empirical outcome is a sample of action quality; it does not establish that every action in a winning game was good.

The outcome probabilities now determine the entire value consumed by PPO, evaluation, and the critic-only path. The residual tower is bypassed, receives no gradients, and cannot compensate away a changed win prediction. Its registered parameters and Adam state remain present solely to preserve checkpoint compatibility. The inactive Huber objective is never called. Logs mark it disabled instead of displaying a zero loss as successful fitting.

This change separates critic correctness from baseline accuracy: the underlying on-policy Monte Carlo score-function estimator tolerates an inaccurate action-independent baseline, whereas future bootstrap errors can bias local credit. This does not make clipped, normalized, multi-epoch PPO unbiased or guarantee a gameplay gain. Full-episode estimates have higher variance. The [GAE derivation, equations 14–18](https://arxiv.org/html/1506.02438v6) provides the relevant estimator identity; [Huber regression analysis](https://arxiv.org/abs/2009.12755) explains the loss/conditional-mean distinction. Our [research assessment](credit_assignment_research_20260916.md) records evidence and earlier inconclusive interventions.

## Configuration and checkpoint handling

The new recipe is explicit:

```python
collection_mode = "complete_cohort"
outcome_label_mode = "complete_episode"
return_value_mode = "terminal_only"
advantage_estimator = "complete_episode_mc"
value_loss_coeff = 0.0
outcome_loss_coeff = 0.10  # positive proper-outcome supervision
gamma = 1.0
```

Validation requires terminal-only rewards, complete labels, positive finite outcome supervision, and no residual loss. Partial-game collection and incompatible reward settings fail before environments are created. `AgentConfig.return_value_mode` is persisted and must agree with PPO metadata. Known historical omissions restore the old modes; unrelated schema omissions remain errors.

Existing checkpoints and pinned studies retain `terminal_plus_residual` and `fragment_gae`. Switching an existing model requires an explicit new branch:

```sh
.venv/bin/python -m tools.prepare_outcome_credit \
  --parent /path/to/existing/complete-cohort-checkpoint.pt \
  --destination /path/to/new-branch/initial.pt
```

This command only prepares a file. It preserves weights, full Adam and RNG state, total steps, update count, cohort counters, seed cursor, the entire inherited seed plan, and evaluation reservations. It records the changed learning semantics, new output paths, parent hash, source hashes, and fresh run provenance. It refuses existing output/runtime paths. It neither adds training seeds nor assigns accelerator time. Subsequent checkpoints retain the branch record, and resumed training rejects source changes. The old study manifests and checkpoints are not rewritten.

The source-pin check is enforced when using the branch through `train_ppo`; direct fresh configurations remain supported. A later launch must use the saved full recipe, rather than the old study's pinned launcher. No production branch or long run was created during this implementation. The separate illegal-action failure remains unresolved.

## Directional verification

Tests exercise the desired direction, not merely optimizer activity:

- For one state with two wins in five outcomes, NLL's win-logit gradient is `p − .4`: negative below .4, zero at .4, positive above .4. The effective value is exactly `10p`, even when stored residual weights are nonzero.
- The legacy production Huber loss has a nonzero residual gradient at the correct mean 4 on those same binary outcomes. This is the minimal objective-mismatch example; live historical GAE targets were not binary Monte Carlo targets.
- A delayed win 600 decisions away retains its complete return target. Changing later values, lambda, collection chunk size, or slot-boundary offsets cannot alter it. Changing the pre-action value only changes the baseline term.
- Episodes cannot leak rewards across games; censored trajectories remain rejected. Targets are fixed across PPO epochs.
- A tiny production update never invokes Huber in terminal-only mode. Residual weights and optimizer state remain inactive. Uninterrupted and split/resumed runs produce identical model, Adam, RNG, and cohort state.
- Branching preserves actor action probabilities exactly while changing the critic semantics explicitly. Evaluation restores those semantics; mismatched metadata and source drift are rejected.

New target telemetry records completed returns, raw advantages, collected-critic bias/MSE against realized returns, and target bias/MSE against those returns. These are decision-weighted diagnostics. Realized-return MSE includes outcome noise and must not be relabeled as reducible critic error. Zero target MSE in Monte Carlo mode verifies the target definition, not learned gameplay strength.

Relevant tests: `test_outcome_value_mode.py`, `test_cohort_monte_carlo.py`, `test_outcome_credit_training.py`, and `test_outcome_credit_branch.py`, together with the existing cohort, critic, checkpoint, transfer, and PPO regression suites. All implementation verification uses CPU and bounded trajectories.

Verification on 2026-09-16: the combined regression suite passed 254 tests. After adding two effective-update-plan compatibility regressions, the final affected integration subset passed all 73 tests (256 distinct tests across those checks). Ruff and `git diff --check` passed. The stopped study's accelerator ledger is unchanged. Its historical research artifacts describe the pre-change implementation and retain their original source hashes.
