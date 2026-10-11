# World-model research applicable to the long-horizon study

**Deferred by explicit user instruction:** “until we figure out the existing
gaps dont bother with world models.” Do not pursue further world-model research,
design, implementation or experiments while the transfer gaps remain unresolved.
The historical proposals below are not active next steps. Completing a small
fitting diagnostic alone does not establish that those gaps have been explained.

Subsequent user steering refocuses the investigation on the unresolved
[pretraining transfer gap](pretraining_gap_focus_20260923.md). The proposals below
remain hypotheses; no world-model experiment has launched.

Research note responding to the user's September 23 question. This proposes a
separate follow-up; it does not modify the active observation experiment or
establish that a world model will improve win rate. No new model training,
gameplay, held-out comparisons or accelerator work was performed for this note.

## Research to borrow

| Research | Relevant mechanism | Proposed application |
| --- | --- | --- |
| [MuZero](https://www.nature.com/articles/s41586-020-03051-4) | Learn recurrent latent dynamics together with reward, policy and value predictions useful for planning. | Represent how actions change future opportunities, with a value target tied to winning. |
| [Stochastic MuZero, author explanation](https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/) | Separate the action's afterstate from subsequent chance outcomes and search over both. | Distinguish deterministic purchases/resource changes from uncertain draws, offers and effects. |
| [Sampled MuZero](https://proceedings.mlr.press/v139/hubert21a/hubert21a.pdf) | Plan over sampled candidate actions with a policy-improvement rule that accounts for sampling. | Use the existing grammar to propose a bounded set of legal actions; avoid enumerating all 17,327 flat IDs. |
| [EfficientZero](https://proceedings.neurips.cc/paper_files/paper/2021/file/d5eca8dc3820cad9fe56a3bafda65ca1-Paper.pdf) | Supervise temporal consistency between predicted and encoded future latent states. | Learn from transition sequences in both successful and unsuccessful games. Benefit on this structured-state task must be tested. |
| [DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) | Improve actor and critic using trajectories imagined by a learned world model. | A later stage if prediction and short planning survive simulator checks; it introduces a substantial new training pipeline. |

TD-MPC2 also offers useful latent-planning ideas, but its published core setting is
[continuous control](https://www.tdmpc2.com/). Our masked, combinatorial discrete
actions make the sampled/stochastic MuZero mechanisms a more direct starting
point. This is a judgment about fit, not an empirical algorithm comparison here.

## Cheapest representation diagnostic

Before implementing planning, isolate whether predictive training improves the
representation. Cache encoder features from completed training trajectories and
train matched small latent adapters with identical direct action-value readouts.
One arm additionally predicts action-conditioned future features; the control
uses the same value targets without that objective. Declare matched training
budgets and account for target-generation cost. At evaluation, discard the
prediction decoder and give both arms the same candidate actions and one-pass
readout, without search. Any ranking benefit then cannot come from extra
deployment planning compute. It would support predictive representation training,
not establish that dynamics prediction is uniquely better than every other
auxiliary objective.

Include measured consequences at later blind boundaries under a specified
continuation policy, alongside immediate correctness targets. Factual training
trajectories alone do not identify the value of unchosen actions; fresh ranking
validation needs separately reserved counterfactual supervision and seed-grouped
splits. The actor can remain frozen throughout this diagnostic.

An exact simulator makes the learning rationale especially important: transition
prediction could supply useful supervision from ordinary games and teach reusable
consequences. If simulator branching with the current critic already ranks actions
well, distilling those decisions may be cheaper. If its rankings are poor, a
learned transition model does not automatically fix the leaf-value estimates.
Neither outcome is known yet.

## Subsequent planning pilot

Keep the current actor frozen initially. Encode the observation plus the proposed
action's type, selected entities and targets. Learn an action-conditioned next
latent state, with separate observable consequence predictions and uncertainty
where appropriate. Use short real sequences to constrain multi-step predictions.
Immediate cash/inventory changes provide checks; delayed scoring capacity,
resources and survival at a later blind boundary are the decision-relevant tests.
Heuristic build scores must be distinguished from measured future outcomes.

Initially rank a small legal action set at early shop decisions under a fixed
continuation policy. Compare three approaches using the same candidate actions,
information access, train/validation split and declared resource accounting:

1. A direct action-value predictor with comparable capacity and data.
2. The action-conditioned dynamics/consequence model.
3. An exact simulator transition plus the current critic where that transition
   does not reveal hidden future information.

Prediction of immediate purchase costs alone is insufficient. The dynamics model
must help rank delayed consequences beyond the direct predictor before expanding
to imagined policy training or deeper search. For any later planner, measure
errors on actions it favors, not only average prediction error: optimization can
select precisely the forecasts that are wrong.

Once useful planning is demonstrated, policy distillation is a possible route
to retaining improvements without search on every decision. Compare deployment
compute explicitly so extra search is not mistaken for improved learned behavior.
Gains with search can still be useful; they answer a different comparison from
gains retained by the policy alone.

## Existing infrastructure and limitations

References prefixed NEWCOPY refer to
`eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate`.

- `NEWCOPY/src/pylatro_agent/training/cohort_io.py:17–47` stores aligned training
  observations, actions, rewards and final observations. Consecutive observations
  supply next-state targets; `training/cohort.py:89–102` packs action masks.
  Richer build targets may need deterministic replay.
- `NEWCOPY/src/pylatro_agent/env.py:266–309` supports boundary snapshot/restore;
  `transfer_diagnostics.py:260–262` supports an any-phase diagnostic clone.
- Prior `analysis/prepare_shop_counterfactual.py:155–199,245–257` reconstructs
  prefixes, checks observation equality and fixes legal action pairs before
  outcomes; `run_shop_counterfactual.py:139–181` branches under common continuation.
- `NEWCOPY/src/pylatro_agent/shop_eval.py:323–390` exposes candidate build/economy
  measurements, some of which are heuristics rather than realized outcomes.

**Cloning is not sampling the agent's uncertainty.** Snapshots retain engine RNG.
Existing counterfactual repeats vary continuation-policy RNG, not independently
sampled future worlds. Arbitrary midgame reseeding does not define a valid
distribution over hidden states consistent with prior observations. The existing
counterfactual results were inconclusive and their states are exposed development
material; they cannot become fresh validation by relabeling.

**Information access must be explicit.** The current tokenizer emits draw-pile
cards in engine order (`NEWCOPY/.../tokenizer.py:321–347`), and embeddings add
absolute positions (`embeddings.py:629–631`). This existing caveat was documented
in `docs/audits/2026-09-06-model-training-audit.md:172–177`. A player-visible
world-model experiment must canonicalize unknown order and audit derived inputs
in an isolated variant, with matched controls. Alternatively it must explicitly
inherit and disclose the current privileged observation regime. Never feed seeds,
RNG state, snapshot metadata or future outcomes into the predictor or planner.
The active frozen study is unchanged.

**Legal actions change with imagined state.** The current grammar and root mask
are reusable, but a root mask cannot be carried unchanged through imagined future
states. Deeper planning needs reliable action legality and chance handling.

Group all transitions, prefixes and counterfactual branches from one original
game seed into the same data split. Newly reserved validation must remain separate
from the current held-out panels and from exposed diagnostic states.

## Evidence required for the actual goal

Offline prediction or ranking improvement would only justify continuation.
Completion still requires fresh-game win-rate gains against matched controls and
evidence that better early decisions contribute. A stronger claim of delayed
planning needs choices with short-term costs and later payoffs under controlled
continuations. Model losses and simulator reconstruction alone cannot establish
that result.

The existing observation evaluation must finish under its original criteria.
Before any world-model run, freeze a separate data split, information regime,
resource budget, baselines, endpoints and outcome tests. Source feasibility was
reviewed by `review_state_encoding`; primary papers were checked by the root agent.
