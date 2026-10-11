# Translating the learning diagnosis into research and implementation

Research synthesis, September 27, 2026. This is a proposal grounded in the saved experiments and current source, not a new trained result. No training, gameplay evaluation, or checkpoint changes were performed for this note.

The strongest current diagnosis is a combination of poor transfer of learned corrections, loss of useful behavior during continuation, sparse decisive contexts, and a historical data-generating-rule mismatch. These resemble established research problems. The correspondence motivates controlled interventions; it does not identify a unique cause or guarantee a win-rate gain.

## What prior art says, and what we have already tested

| Our evidence | Relevant prior art | Concrete translation and limitation |
| --- | --- | --- |
| Training imitation improves while fixed held-out imitation worsens; ranking fits about 90% of training pairs and 44% held out | [Procgen, Cobbe et al. (2020)](https://proceedings.mlr.press/v119/cobbe20a.html) demonstrates the importance of training-environment diversity for procedural-game generalization | Compare a broader current-rule seed population against reuse of the current 1,000 games at matched presentations. New data must use the same engine and teacher settings as its control. This is not evidence that a particular game count is sufficient. |
| Changed encoder features can reproduce a fatal regression even with the old correction head | [Kumar et al. (2022)](https://arxiv.org/abs/2202.10054) studies feature distortion and worse out-of-distribution performance during fine-tuning; [Side-Tuning, Zhang et al. (2020)](https://arxiv.org/abs/1912.13503) learns a separate network alongside a fixed pretrained network | A separately trainable branch could learn new features without changing the established feature path. Freezing the entire encoder has already limited learning here. A frozen reference also cannot prevent a new branch from making bad final choices. |
| Corrections improve selected cases but lose clears where the base was already correct | [Experience Replay for Continual Learning, Rolnick et al. (2019)](https://arxiv.org/abs/1811.11682) uses retained experience to reduce forgetting; [Learning without Forgetting, Li and Hoiem (2016)](https://arxiv.org/abs/1606.09282) preserves previous outputs while adapting | Our KL-preservation experiment already tests a related distillation principle. Another generic KL loss is not a novel remedy. A distinct data intervention would retain broad current-rule examples and consequential base-correct comparisons as well as corrections, without using exposed evaluation cases as training data. |
| Small imitation errors can alter subsequent states | [DAgger, Ross et al. (2011)](https://proceedings.mlr.press/v15/ross11a.html) aggregates expert labels on states reached by the learner | Refresh student-state data iteratively and retain the aggregate. A previous finite correction-data replacement failed here. Also, our fixed teacher-state regression exists without autonomous state drift, so DAgger cannot be the whole explanation. |
| Exact action agreement, pair ordering, immediate clears, and final wins disagree | [AggreVaTe, Ross and Bagnell (2014)](https://arxiv.org/abs/1406.5979) extends interactive imitation using action costs; [UNREAL, Jaderberg et al. (2016)](https://arxiv.org/abs/1611.05397) improves representations using additional learning signals | Predict measured action consequences and eventually continuation cost, rather than supervising only teacher labels. Immediate hand score or clear status is not cost-to-go and must not be presented as a full AggreVaTe implementation. |
| Reusable card/Joker interactions fail to transfer reliably | [DouZero, Zha et al. (2021)](https://proceedings.mlr.press/v139/zha21a.html) encodes actions for value estimation in a combinatorial card game; [Relational Deep RL, Zambaldi et al. (2018)](https://arxiv.org/abs/1806.01830) uses entity interactions; [Set Transformer, Lee et al. (2019)](https://proceedings.mlr.press/v97/lee19d.html) models interactions in sets | Test a scorer that explicitly conditions its access to relevant entities on the proposed action. Our existing transformer and subset head already supply attention and structured actions: neither adding attention in general nor adding another generic residual is a new hypothesis. |

These are analogies across different experimental settings, not evidence that our same-task continuation is identical to a benchmark's domain shift or multitask catastrophic forgetting.

## Most defensible architecture candidate: a hand-consequence scorer

The proposed question is whether direct supervision of reusable action consequences, with explicit access to the interacting entities, transfers better than the existing teacher-label and pair-ranking supervision.

Current implementation:

- [agent.py](../../src/pylatro_agent/agent.py) already uses a transformer over typed observations.
- [action_grammar.py](../../src/pylatro_agent/action_grammar.py) constructs a global mean-pooled state summary and supplies contextual hand-card tokens to the subset head. Joker tokens are available in the same forward pass.
- [subset_residual.py](../../src/pylatro_agent/subset_residual.py) projects each card, sums selected-card features with count normalization, adds state/count context, and produces play/discard corrections. Its card tokens already contain transformer context; they are not isolated raw cards.
- [embeddings.py](../../src/pylatro_agent/embeddings.py) already embeds heuristic hand-candidate scores. Those are not an exhaustive, supervised consequence predictor for every legal subset.

A narrowly scoped implementation could add an offline `HandConsequenceHead` first:

1. Reuse the exact subset membership and legality tables. Encode the proposed subset, its action type, and selected versus retained cards.
2. Build a small action query that reads contextual Joker, hand-level, history and card tokens through cross-attention. Keep meaningful ordering and position information: Joker order and some card-order effects matter. Do not arbitrarily permute the whole observation or claim global set invariance.
3. Predict `log1p(immediate_score)` and immediate blind-clear probability, using the remaining blind requirement and resource context already visible to the policy. Treat these as immediate-consequence outputs, not eventual win estimates.
4. Use engine-generated training labels from the existing replay infrastructure. Deterministic cases can have exact targets. Stochastic effects require an explicitly defined target distribution; one saved-RNG realization is not an exactly knowable outcome from the observation. Never expose engine RNG or future draws as new predictor inputs.
5. Include teacher actions, base actions, represented alternatives and a bounded set of other legal actions. A model trained only on a preferred pair can still select an unassessed third action. Keep every branch from one game in the same data split.

Start with the policy and encoder unchanged and use the module only for prediction. This makes the first result a representation/supervision diagnostic, with no gameplay regression caused by deploying an untested scorer. A matched MLP using the current pooled subset representation should receive the same targets and examples. First compare that baseline to the relational scorer; do not conflate more supervision with a new architecture.

If frozen features constrain both scorers, a subsequent controlled change can give the new module a small independently trainable observation branch while leaving the base feature path fixed. This adapts Side-Tuning's separation principle. Our earlier frozen-backbone residual work already used part of this principle; the distinguishing change would be the new branch's own features, not just another output layer.

Only after held-out consequence ranking improves should a separate experiment integrate the scorer into the existing legal-action distribution. Immediate score should be a feature for a learned decision, not a universal max-score policy: discarding, saving resources, and Joker growth can trade current score for future value. Training-time simulator labels do not require simulator lookahead at deployment. An exact-score inference helper would be a different intervention with a different information/compute budget.

The expected signature is improved unseen-state score ordering and clear prediction, especially on last-hand decisions and held-out combinations, without losing already-correct clear choices. Failure to beat the matched simpler predictor would weaken the relational-readout hypothesis. Improved consequence prediction without gameplay improvement would instead point toward action integration, state coverage, or the mismatch between immediate and eventual value.

## Sequence and spending decision

1. Reuse the saved 0/1,000/2,000/4,000 checkpoints to distinguish late overfitting from corrections that never transfer. This remains the cheapest unresolved diagnostic in [research guidance](research_guidance_20260927.md). Reuse equivalent completed results if present. No new training is needed for this step.
2. If transfer peaks early and then deteriorates, test a stopping rule before adding capacity. If it stays weak despite strong training learning, compare coverage and consequence prediction. Decide between a data change and the scorer based on that evidence, rather than combining them in one fit.
3. For a data test, use current-rule demonstrations, mix games across batches, and retain baseline-correct as well as corrective contexts. Keep training presentations and control provenance explicit. The original 5,000 games are not a clean scale-up: [the corpus audit](original_corpus_feasibility_20260927.md) found obsolete cleanup actions and different pack transitions.
4. For the scorer test, reuse existing action replays, make an explicit deterministic/stochastic label audit, and benchmark one small batch before estimating cost. The existing subset implementation chunks activations; preserve bounded computation instead of running a full transformer separately for every subset.
5. Follow the existing single-exploratory-fit workflow. Advance only on the declared development criterion, including correction transfer and retention. Then evaluate sampled and greedy games and the early-policy intervention. Independent training and untouched confirmation belong to a selected final candidate.

No wall-clock improvement or training-time estimate is established for the proposed module. No numerical success threshold or new fit recipe has been selected in this research note. Those must be concrete before spending on a scientific run; the prior final win-rate standard remains unchanged.

## What the evidence does not currently support

- A larger transformer, longer context, or a full world model as the next default intervention. Relevant context was available in several diagnosed failures; a cause of the overall gap remains unresolved. Existing world-model deferral remains in effect.
- Repeating generic preservation/ranking combinations because they resemble published methods. Both principles have already been tested locally, including their combination, without the required gameplay result.
- Treating all card states as permutation invariant. The proposed shared action scorer must preserve semantically meaningful order.
- Interpreting a successful local score predictor as a solution to early shop strategy or long-horizon credit assignment.

The implementation priority is a more testable learning signal and action readout, alongside rule-aligned coverage. It is a targeted hypothesis, not a promise that any single paper supplies a drop-in fix.
