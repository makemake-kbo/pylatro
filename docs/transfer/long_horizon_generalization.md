# Long-horizon generalization experiment, 2026-09-21

The requested outcome is an improvement in full-game win rate on unseen games,
with evidence that learned early decisions contribute to those wins. Training
loss, critic parameter movement, and correct return targets do not establish
that outcome. This study is complete and failed its acceptance gates; the full
generalization claim remains unproven.
All 120 planned training updates are complete and audited under the logical-KL
recovery. All development evaluations are complete. The fixed selection rule
chose +20 updates. All 24 fresh confirmation panels, totaling 9,600 plays, are
complete, and the canonical win comparisons have passed independent verification.
Both sampling repeats are now audited across both replicas and matched controls.
MC and the original-objective legacy control both win 38.875%, terminal GAE
38.4375%, and the parents 35.75%. MC-minus-parent
is +3.125 points [+0.625, +5.625], a small gain on fresh games. MC-minus-GAE is
only +0.4375 [−2.1875, +3.1875], with one replica worse. The sampled acceptance
gates fail: neither gain reaches five points, and MC does not beat GAE in both
replicas. The complete greedy comparison passes the regression guardrails but
shows no resolved improvement. The legacy control also gains +3.125 points
[+0.6875, +5.5] over its parent. MC-minus-legacy is 0 [−2.5, +2.625], and
terminal-GAE-minus-legacy is −0.4375 [−3, +2.125]. Continued training under the
original objective reproduces the small fresh-game gain; neither experimental
change shows a resolved advantage over that control. The legacy control's greedy
rate is 48.875%; MC-minus-legacy is +3.625 points [0, +7.25], still including zero.
The 50 inherited updates are preserved with their original source history.
The reserved early-decision attribution test is complete and independently
audited: early MC wins 635/1,600 games against 609/1,600 for all-GAE, a gain of
1.625 points [−1.125, +4.4375]. Both replica point estimates are positive, but
the pooled interval includes zero, so the early-decision gate also fails.
The shop action-ranking
diagnostic is complete and inconclusive: preferred-action gain +4.6875 points
[−7.03125, +17.1875] on 32 reused training states, with all 260 trajectories
independently replayed exactly.
The completed run is
[`long_horizon_generalization_20260921_logical_kl`](../../eval/long_horizon_generalization_20260921_logical_kl/manifest.json).

## Diagnosis and repaired collection failure

The [saved-log diagnosis](../../eval/complete_cohort_long_20260915/analysis_20260916/README.md)
found complete terminal labels and optimizer exposure in every committed update,
but no resolved sampled-win improvement on the 50-seed development panel. For
decisions more than 100 actions from termination, the mean coefficient of the
terminal reward in fragmented GAE was below 1%. Critic bootstrapping can still
transmit information; this measurement identifies a testable credit bottleneck,
not proof that Monte Carlo credit will improve play.

The [completed-outcome credit implementation](outcome_credit_fix.md) already
provided an outcome-only value and full-episode Monte Carlo advantages. It had
not been trained on real games before this experiment.

The preceding run's illegal action was reproduced from update 140, with its
saved MPS RNG, on exactly seed 8005054 at step 153. The actual failure began four
decisions earlier: the last drawable cards were played below the blind target,
with one hand remaining. The controller stayed in hand play. After four Joker
sales, the observation had no legal action.

The controller now resolves an exhausted hand/deck as a round ending. This also
covers discards, card destruction, and empty opening decks. A qualifying
Mr. Bones resolves the exhausted round; it does not request another empty hand.
The installed game's `Game:update_selecting_hand` and `end_round` definitions
were inspected to verify these transitions. Proprietary source is not copied
into the repository. Existing ordinary hand-limit behavior is outside this fix.

The regression fixture preserves the 149-action path to exhaustion. Both the
Gym environment and compiled fast runner now produce a normal loss at Ante 5,
with zero win-only reward. Replaying the same frozen 64-game cohort completes
all games after the fix. These are collection-correctness results, not wins
gained by the model. All experimental arms and baseline evaluations use the
same corrected simulator.

Evidence is under `eval/long_horizon_generalization_20260921/legality/`:
`reproduction_before/result.json`, `reproduction_after/result.json`, and
`regression_tests.log`. The reproduction tool is
`python -m tools.reproduce_cohort_failure`.

## Matched experiment

Both parents are the last equally trained endpoint of the preceding run,
global update 122, chosen before seeing new evaluation games. Every branch
preserves model weights, Adam state, RNG, update count, consumed seed prefix,
and logical collection counters. New output directories preserve old studies.

| Arm | Value used by actor | Actor target |
| --- | --- | --- |
| `legacy_gae` | `10 P(win) + residual` | Fragmented GAE, lambda .97 |
| `terminal_gae` | `10 P(win)` | Fragmented GAE, lambda .97 |
| `terminal_mc` | `10 P(win)` | Actual complete-game return minus collected value |

The MC versus terminal-GAE comparison changes only the actor credit estimator.
The terminal-GAE versus legacy comparison tests the residual-value objective.
All arms retain complete categorical outcome supervision, terminal-only 0/10
rewards, gamma 1, LR 1e-5, 64 completed games per cohort, and the same optimizer
exposure. There is no teacher, reward shaping, replay, or action replacement
during training. The environment fix is shared by every arm.

The fresh-comparison preflights for
[replica 0](data/long_horizon_fresh_credit_seed0_preflight_20260922.json) and
[replica 1](data/long_horizon_fresh_credit_seed1_preflight_20260922.json) directly
verified identical initial model tensors, full optimizer state, Python/NumPy/CPU/MPS
RNG states, counters, architecture, and reward configuration between MC and terminal
GAE. Their recipes differ only in the advantage estimator, apart from output paths
and run identifiers. Both completed 1,280 training games, 327,680 optimizer examples,
and 520 optimizer steps per active parameter without rollback or early stopping.
These checks validate the matched treatment and training records; they supply no
gameplay result. Both fresh matched sampling repeats and the greedy comparison are
now complete and reported below. Both legacy sampling repeats and their greedy
panels are complete too.

The [value-control preflight](data/long_horizon_fresh_value_control_preflight_20260922.json)
also verifies exact initial weights, optimizer state, and RNG state between
terminal GAE and legacy GAE in both replicas. Their semantic recipe differences
are `return_value_mode` and `value_loss_coeff` (0 versus 0.25). Architecture is
otherwise identical, with the same reward configuration and planned optimizer
exposure. The four residual-head parameter tensors receive no updates in terminal
GAE, as intended. This comparison tests the combined value objective; it cannot
separate the effect of critic predictions from the auxiliary residual loss.

The first real MC update in the superseded resource pilot (replica 0, global
update 123) committed. A
[trajectory/checkpoint audit](../../eval/long_horizon_generalization_20260921/analysis/first_update_audit.json)
verified all 64 declared game seeds, 10,947 completed decisions, saved trajectory
hashes, 16,384 optimizer examples, and 26 Adam steps for each of 314 participating
parameters. The four unused residual-value parameters did not advance. No KL
rollback occurred.

Of those decisions, 5,067 were at least 100 actions before termination, and
2,667 distinct such rows received optimizer exposure. Their direct terminal
coefficient is 1.0. Reconstructing fragmented GAE on these exact trajectories
and frozen terminal-only values gives a mean direct coefficient of .00661.
This verifies the intended credit change; it does not measure action quality
or a gameplay gain. The initial MC targets equal the observed return by design.

Each arm has two inherited training replicas, each with 20 new updates and
1,280 fresh game seeds. Training seeds are paired across arms, distinct between
replicas. Every update records loss, completed outcomes, actual optimizer
exposure, and direct terminal credit/return errors by distance to termination.
Zero MC target error verifies the target definition; it is not a critic score.

The first experiment is bounded by a new 12-hour local MPS wall-time cap,
including training failures. This is a self-imposed experiment limit under the
active improvement goal. It does not reuse or modify prior compute allowances.
CPU gameplay evaluation runs separately, and existing MPS memory limits remain
unchanged. A `STOP` file requests termination after the current committed update.

The first study stopped cleanly after global update 125. Optimization took
410, 614, and 630 seconds; a native memory report showed a 47.4 GB footprint with
24.7 GB swapped out. Its checkpoints and source archive are preserved. Cache
cleanup between optimizer steps barely reduced peak memory and did not improve
the short replay time, so it was not applied to training.

An explicit resource revision restarted all six arms from their original
update-122 parents, using physical batch 160 with the same logical batch 640.
The partial first study is diagnostic evidence, not a candidate pool. Fresh
training seeds start at 310000000 and 310100000, and evaluation panels start
at 260000/270000/280000. These ranges passed the historical overlap audit.
The new ledger carries only the unspent portion of the original self-imposed
12-hour cap, including the cost of the diagnostic probes: 2,355.6 seconds spent,
40,844.4 seconds remaining. Preparation rechecked 709 historical manifests and
32,312 previously reserved seeds. Training and CPU development evaluation used
the revised, frozen source identity until the interruption described below.

The [resource and numerical checks](data/long_horizon_memory_revision_20260921.json)
used the same policy, Adam state, recorded rows, row order, and logical exposure:

| Physical batch | Recorded rows / epochs | Update seconds | Peak live tensor GiB | Peak recorded driver GiB |
| --- | --- | ---: | ---: | ---: |
| 320 | 1,280 / 2 | 22.15 | 21.77 | 25.00 |
| 160 | 1,280 / 2 | 22.98 | 10.40 | 11.71 |
| 160 | 8,192 / 2 | 125.37 | 10.40 | 12.79 |

On 64 saved observations, all greedy actions matched. Maximum selected-action
log-probability difference was 1.91e-6; maximum win-probability difference was
1.79e-7. Adam step counts and RNG states matched exactly. Weights were not
bitwise identical: physical batching changes floating-point accumulation order,
and a repeated batch-320 probe also showed small numerical differences. These
are bounded execution checks; the revised study must still demonstrate gameplay
gains. No system or MPS memory limit was changed.

The first revised real update has also committed and
[passed its checkpoint/trajectory check](../../eval/long_horizon_generalization_20260921_micro160/analysis/first_update_verification.json):
64 complete games, all 16,384 optimizer examples across 104 physical minibatches,
full outcome labels, no KL rollback, and zero MC target error. Collection took
141.4 seconds and optimization 130.6 seconds (273.5 seconds total). Its native
memory report showed a 20.5 GB footprint with no swapped pages. Of 4,179 decisions
at least 100 actions before termination, 2,346 distinct rows were optimized.
These checks do not establish a gameplay gain; the first candidate development
results are reported below.

The first +10 endpoint committed for `terminal_mc`, replica 0, at
global update 132. Its [endpoint audit](data/long_horizon_first_endpoint_20260921.json)
verified all 640 fresh game seeds and trajectory hashes, 103,669 completed
decisions, complete outcome labels, and all 163,840 optimizer examples without
rollback. The first replica's matched gameplay results are reported below.

MC replica 1 has also completed +10. Its
[endpoint audit](data/long_horizon_second_mc_endpoint_20260921.json) verifies
all 640 trajectory hashes, the exact fresh seed prefix disjoint from replica 0,
101,014 completed decisions, complete labels, and 163,840 optimizer examples
without rollback or early stopping. It checks the terminal-only MC checkpoint
recipe, seed-cursor and optimizer-counter advances, and logged zero MC target
error. Its sampled and greedy development results are reported below.

The [second terminal-GAE endpoint audit](data/long_horizon_second_gae_endpoint_20260921.json)
has also completed: 640 verified trajectory hashes, 100,542 decisions, full
outcome labels, and 163,840 optimizer examples without rollback or early
stopping. It verifies the same fresh game seeds and optimizer exposure as MC,
the terminal-only GAE recipe, and disabled residual-value loss. Both credit
arms are now trained through +10 in both replicas. The second terminal-GAE
replica has completed evaluation; the second legacy control failed before
committing its first new update. These
audits establish planned training exposure; gameplay effects still require
the completed comparisons and fresh validation.

The first terminal-GAE control update also committed with full optimizer
exposure. A [matched-cohort check](data/long_horizon_first_matched_cohort_20260921.json)
found identical recorded actions, rewards, values, log probabilities, slot
counters, observation scalars, and optimizer row draws across the two arms'
first 64 games (9,863 decisions). Their frozen policy hashes and game outcomes
also matched. The changed credit targets therefore start from the same recorded
experience. Raw advantage standard deviation was 4.217 for MC and 1.423 for GAE;
the larger spread does not itself establish better credit assignment. This
check applies to the first cohort only; subsequent on-policy trajectories may
diverge as intended.

The [first value-target comparison](data/long_horizon_first_value_target_comparison_20260921.json)
also verifies identical experience between terminal-GAE and legacy-GAE before
their first optimizer step. All saved arrays except recorded values match
exactly across 64 games and 9,863 decisions, including every observation field;
all 128 trajectory hashes, outcomes, slot counters, and optimizer draw hashes
were checked. The legacy residual adds .876 return points on average, or .925
on Ante-1–3 states. Reconstructing production GAE and whole-cohort normalization
changes advantage signs on 5.67% of decisions (5.08% at Antes 1–3), with an
overall normalized-advantage correlation of .989. Raw advantage standard
deviations are 1.423 for terminal-GAE and 1.541 for legacy-GAE. These changes
identify the initial learning-signal contrast; they do not identify correct
action credit. Later training also removes the residual-value loss, so this
diagnostic does not isolate that gradient effect or establish better gameplay.

## Frozen-critic fitting diagnostic

A separate CPU probe used the first revised training cohort: 64 complete games
from the frozen update-122 replica-0 parent. It cached the encoder's features,
kept the actor and encoder frozen, and fitted private copies of the outcome
head on 32 whole games. The other 32 games supplied validation only. None of the
main development, confirmation, or early-intervention seeds entered this probe.
The [compact evidence](data/frozen_critic_probe_20260921.json) records hashes,
learning curves, per-game paired intervals, and a categorical-support audit.

The initial random split contained 10 training wins and 17 validation wins.
After seeing that imbalance and the fitting curve, a post-hoc check swapped
the halves. Each direction used two fitting orders, fresh AdamW at LR 3e-4,
batch 320, and 400 updates. These are private-head diagnostics, not candidate
gameplay checkpoints. Losses below average states within each game, then games
and fitting orders; lower is better.

| Split direction | Training outcome NLL, initial → 400 | Validation outcome NLL, initial → 400 | Validation win Brier, initial → 400 |
| --- | ---: | ---: | ---: |
| Original 32/32 split | 1.704 → .770 | 1.303 → 1.753 | .166 → .214 |
| Post-hoc reverse split | 1.303 → .615 | 1.704 → 3.082 | .164 → .287 |

The validation NLL increases at 400 updates have paired, game-cluster 95%
intervals of [.185, .743] and [.598, 2.234]. On Ante-1–3 states, the increases
are .440 [.176, .728] and 1.055 [.315, 1.916]. This demonstrates that additional
fitting can improve training forecasts while worsening unseen-game forecasts
in this small cohort. It argues against treating more critic updates as an
automatic remedy. The reversed halves reuse the same 64 games; the intervals
condition on each fitted head and validation half. They are not independent
confirmation or evidence that this fitting schedule caused the PPO plateau.
The main PPO learning rate, exposure, and joint encoder updates differ.

The audit also reproduced a narrower representation defect. In seed 310000035,
recorded action 60 buys Hieroglyph, reducing Ante 4 to 3; the game later loses
at Ante 3. Both the Gym environment and compiled runner reproduce all 87
recorded actions and the same loss. The hazard head assigns zero probability
to death before the current Ante. Its clipped categorical NLL is therefore
16.118 with zero hazard-logit gradient on the five earlier Ante-4 rows. These
constant rows occur in both the initial and fitted scores; they do not account
for the measured increase after fitting.

Across the first five revised cohorts, this zero-support outcome affected
13 of 50,982 decisions in two of 320 games, with 18 of 81,920 optimizer examples
across both epochs. Ante decreases occurred in 16 games, most without a later
death below their previous Ante. The categorical support needs to accommodate
legal Ante decreases in a future explicit model revision. This small observed
frequency does not establish it as the plateau's cause. Binary win labels and
complete-game Monte Carlo returns remain valid. The running matched study's
policies, source hash, recipes, and selection rule are unchanged.

## Historical repeated-game return check

A [post-hoc return-baseline diagnostic](data/long_horizon_historical_return_baseline_variance_20260921.json)
uses the completed earlier outcome-PPO confirmation: 400 game seeds, three
sampled attempts per policy and seed, for its initial control and two selected
update-42 replicas. These are historical policies, not the current update-132
checkpoints. Input export hashes and all 3,600 sampled outcomes were checked.

For each 0/10 return, the diagnostic subtracts either the other two returns on
the same game seed, averaged, or the mean return on all 399 other seeds. Both
resulting residual collections have zero overall mean. Relative changes in
their empirical variance are:

| Historical policy | Same-seed baseline vs constant reference, relative variance change [95% interval] |
| --- | ---: |
| Initial control | +3.14% [−9.32%, +15.75%] |
| Selected replica 0 | −0.45% [−11.11%, +11.11%] |
| Selected replica 1 | −1.91% [−12.35%, +9.35%] |

The 5,000 bootstrap resamples retain all three attempts within each game seed
and recompute the constant reference. None of these comparisons resolves a
variance reduction. This exploratory scalar-return calculation does not test
the current learned state-value baseline, measure policy-gradient variance,
or establish a gameplay effect. It supplies no reason to change the running
protocol and does not identify the cause of the current plateau.

## Evaluation and attribution

Explicit seed lists in the frozen manifest are checked against inherited
training data and previous study manifests. New panels are:

- Development: 260000–260199, sampled repeat 0 and greedy, at +10 and +20.
- Confirmation: 270000–270399, sampled repeats 0/1 and greedy, after selection.
- Early-decision transfer: 280000–280399, sampled repeats 0/1, after selection.

The checkpoint endpoint maximizes the development MC-minus-terminal-GAE sampled
difference, with ties choosing the earlier endpoint. Confirmation uses frozen
checkpoint hashes and cannot feed training or checkpoint selection.

Gameplay acceptance requires at least +5 percentage points versus both matched
terminal-GAE and the frozen parent, paired 95% lower bounds above zero, positive
differences for both training replicas, and greedy lower bounds above -5 points.
Intervals resample game seeds after averaging repeats and replicas; they
condition on these two training replicas.

The early-decision intervention uses the MC policy in every phase while the
current Ante is 1–3 and the matched terminal-GAE policy at higher Antes. A legal
Ante decrease can reactivate the MC policy. Its control uses terminal-GAE
throughout. Both policies receive ordinary observations, with no teacher or hidden
future information. Acceptance on this separate panel requires a terminal-win
95% lower bound above zero and a positive point difference in each training
replica. This attributes a benefit to early decisions under a common later
policy. Component critic predictions in this hybrid are not estimates of the
hybrid's value.

This tests generalization to unseen procedural games for the full Ante-8 Blue
Deck, White Stake task. It does not establish transfer to other decks or games.
Even a positive early-policy result would not establish that the model explicitly
plans, distinguish strategic preparation from improved immediate survival, or
override a failed full-game acceptance gate. All original gates must pass.
An inconclusive or negative result keeps the original improvement goal open.

The superseded pilot's first reference panel (frozen parent replica 0,
sample 0, seeds 230000–230199) records 58 wins in 200 games, with no stalls or
censored outcomes. It is preserved as pilot evidence. The revised experiment
uses its own new reference panels; no cross-panel comparison establishes a gain.

## First candidate development results

At +10 updates, replica 0 of all three arms has completed sampled and greedy
development panels against the frozen parent. Each mode uses the same 200
seeds across policies. All eight completed panels have zero stalls or censored
outcomes.

| Policy | Sampled wins, repeat 0 | Greedy wins |
| --- | ---: | ---: |
| Frozen parent | 79/200 (39.5%) | 91/200 (45.5%) |
| `terminal_mc` | 72/200 (36%) | 100/200 (50%) |
| `terminal_gae` | 79/200 (39.5%) | 107/200 (53.5%) |
| `legacy_gae` | 80/200 (40%) | 95/200 (47.5%) |

Differences below are percentage points, with paired game-seed bootstrap 95%
intervals. The estimates condition on this single training replica.

| Comparison | Sampled difference [95% interval] | Greedy difference [95% interval] |
| --- | ---: | ---: |
| MC − parent | −3.5 [−10, +3] | +4.5 [−2, +11] |
| MC − terminal-GAE | −3.5 [−11, +4] | −3.5 [−10.5, +3.5] |
| Terminal-GAE − parent | 0 [−7.5, +7.5] | +8 [+1.5, +14] |
| Terminal-GAE − legacy-GAE | −0.5 [−8, +7] | +6 [−1, +13] |
| Legacy-GAE − parent | +0.5 [−7, +8] | +2 [−4.5, +8.5] |

The outcome-only GAE arm has a positive greedy development result: 29 paired
games were GAE-only wins and 13 were parent-only wins. Its sampled win count is
unchanged, with 27 wins in each discordant direction. This is a lead for further
validation, not proof of the requested generalization. Against matched legacy
continuation, the greedy point estimate favors terminal-GAE by six points, but
its interval includes zero: 33 games are terminal-GAE-only wins and 21 are
legacy-only wins. The value-objective effect therefore remains unresolved;
replication, fresh confirmation, and the early-decision intervention remain necessary.
Neither direct MC-minus-GAE interval establishes a benefit from replacing
fragmented credit with full-episode Monte Carlo.

The completed sampled comparison also does not establish a benefit from the
outcome-only value objective: terminal-GAE wins 30 games that legacy-GAE loses,
while legacy-GAE wins 31 that terminal-GAE loses. Legacy continuation itself is
unresolved against the parent. All three trained arms used the same 640 fresh
training seeds, 163,840 optimizer examples, complete outcome labels, and no KL
rollback at this endpoint. Unequal game counts or optimizer exposure do not
explain the sampled comparison. The intervals condition on one training replica
and are not adjusted for the multiple exploratory contrasts.

The [first-replica optimizer audit](data/long_horizon_first_optimizer_diagnostics_20260921.json)
checks all 30 update hashes against the training snapshot. All updates completed
both epochs without a soft stop or KL rollback. Mean final full-rollout KL is
.00349 for MC, .00307 for terminal-GAE, and .00319 for legacy-GAE; every update
is below .00402, versus the .015 soft-stop threshold. Global gradient clipping
at norm .5 affects 236/260, 227/260, and 243/260 optimizer steps respectively,
with mean gradient scale factors .755, .733, and .687. Clipping is frequent in
every arm, but these global norms do not separate actor and critic contributions
or establish whether clipping limits learning. Sample-weighted PPO ratio clip
fractions are 4.57%, 4.14%, and 4.04%; these are distinct from gradient clipping.
This audit does not motivate changing the frozen optimizer recipe mid-study.

The [sampled comparison metadata](data/long_horizon_first_credit_comparison_20260921.json),
[sampled outcome triplets](data/long_horizon_first_credit_comparison_games_20260921.jsonl),
[greedy metadata](data/long_horizon_first_credit_comparison_greedy_20260921.json), and
[greedy outcome triplets](data/long_horizon_first_credit_comparison_greedy_games_20260921.jsonl)
preserve all three policy comparisons independently of large checkpoints. The
audits verify checkpoint/source/outcome hashes, shared evaluation settings,
and identical MC/terminal-GAE model configurations.

The [full sampled comparison metadata](data/long_horizon_first_value_comparison_sampled_20260921.json),
[sampled four-policy game rows](data/long_horizon_first_value_comparison_sampled_games_20260921.jsonl),
[full greedy comparison metadata](data/long_horizon_first_value_comparison_greedy_20260921.json),
and [greedy four-policy game rows](data/long_horizon_first_value_comparison_greedy_games_20260921.jsonl)
add the matched legacy arm. These audits validate all four checkpoints, source
and artifact hashes, matched evaluation settings, and that the model
configurations differ only in the declared return-value mode. It also records
the hash of the training-exposure metadata snapshot; full trajectory hashing
remains part of the experiment verifier before endpoint selection.

This is one development replica and one sampling repeat. The
second replica, later endpoint, confirmation guardrails, and early-decision
attribution remain pending. The experiment and deterministic MC selection rule
continue unchanged. A promising secondary arm cannot silently replace the
declared treatment; pursuing it requires a prospective comparison and fresh
validation.

## MC development across both replicas

Both MC replicas have completed sampled and greedy development at +10, each
against its own frozen parent on the same 200 game seeds within each mode.
Sampled play uses repeat 0. All eight panels are complete with zero stalls or
censored outcomes.

Sampled play:

| Training replica | Parent wins | MC wins | MC − parent, pp [95% interval] |
| --- | ---: | ---: | ---: |
| 0 | 79/200 (39.5%) | 72/200 (36%) | −3.5 [−10, +3] |
| 1 | 76/200 (38%) | 71/200 (35.5%) | −2.5 [−10, +4.5] |
| Both, equally weighted | 155/400 (38.75%) | 143/400 (35.75%) | −3 [−8.25, +2.25] |

Greedy play:

| Training replica | Parent wins | MC wins | MC − parent, pp [95% interval] |
| --- | ---: | ---: | ---: |
| 0 | 91/200 (45.5%) | 100/200 (50%) | +4.5 [−2, +11] |
| 1 | 102/200 (51%) | 101/200 (50.5%) | −0.5 [−7, +6] |
| Both, equally weighted | 193/400 (48.25%) | 201/400 (50.25%) | +2 [−2.75, +6.75] |

Each combined interval averages both fixed training replicas within each game
seed before bootstrapping the 200 seeds. The 400 physical games per policy are
not treated as 400 independent seeds, and the intervals do not capture
variation across new training replicas. Both sampled replica differences are
negative. The greedy point estimate improves in replica 0 but not replica 1,
and the pooled greedy interval includes zero. Complete-episode credit has not
demonstrated a gain in either mode at this endpoint. The second replica's
legacy control, +20 endpoint, fresh confirmation and early-decision
attribution remain pending; selection and acceptance rules are unchanged.

The [sampled comparison metadata](data/long_horizon_first_replicated_mc_sampled_20260921.json),
[sampled paired game rows](data/long_horizon_first_replicated_mc_sampled_games_20260921.jsonl),
[greedy comparison metadata](data/long_horizon_first_replicated_mc_greedy_20260921.json),
and [greedy paired game rows](data/long_horizon_first_replicated_mc_greedy_games_20260921.jsonl)
preserve the pooled and separate results, with 400 paired rows per mode.
Each audit validates all four
checkpoints against their training/preparation evidence, shared evaluation
settings, source and artifact hashes, and the declared value-mode difference.

## Replicated credit comparison and milestone outcomes

The [complete +10 credit comparison](data/long_horizon_replicated_credit_development_20260921.json)
now includes both MC and terminal-GAE replicas and their parents. These are
the inherited development panels, with 200 shared seeds and two fixed training
replicas per policy in each mode. No recovery evaluation is counted as an
additional independent sample.

| Policy | Sampled wins | Greedy wins |
| --- | ---: | ---: |
| Parent | 155/400 (38.75%) | 193/400 (48.25%) |
| Terminal-GAE | 150/400 (37.5%) | 202/400 (50.5%) |
| Terminal-MC | 143/400 (35.75%) | 201/400 (50.25%) |

The pooled MC-minus-GAE difference is −1.75 percentage points [−7, +3.5] in
sampled play and −0.25 [−5.25, +4.75] in greedy play. GAE-minus-parent is −1.25
[−6.5, +4] sampled and +2.25 [−2.25, +6.75] greedy. All terminal-win intervals
include zero. Replicas are averaged within each seed before bootstrapping;
the intervals condition on these two trained replicas.

The recovery's [complete +10 report](data/long_horizon_complete_endpoint10_development_20260921.json)
now also contains both legacy replicas. Their pooled wins are 154/400 (38.5%)
sampled and 196/400 (49%) greedy. Terminal-GAE-minus-legacy is −1 percentage
point [−6.75, +4.5] sampled and +1.5 [−3, +6] greedy, leaving the residual-value
objective effect unresolved. The MC-minus-legacy differences are −2.75 points
[−8.25, +2.75] sampled and +1.25 [−4, +6.5] greedy. The
[export verification](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/complete_endpoint10_export_verification.json)
checks all 16 outcome artifact hashes and independently reproduces the report's
eight pooled contrasts from 200 seed clusters, averaging the two replicas
within each seed. The terminal-arm comparisons agree with the inherited report;
recovery evaluations add no independent seed observations.

An exploratory decomposition checks unconditional Ante-clear rates rather
than treating midgame survival as a win. Greedy terminal-GAE clears Ante 5 in
316/400 games (79%), versus the parents' 283/400 (70.75%): +8.25 points
[+4, +12.5]. Both replicas improve at this milestone: 81% versus 73.5% in
replica 0, and 77% versus 68% in replica 1. The advantage is smaller at Ante 8,
where the terminal-win interval includes zero. Sampled Ante-5 clear rates
are 67.5% for GAE and 70% for the parents, with no resolved benefit.

The [paired game rows](data/long_horizon_replicated_credit_development_games_20260921.jsonl)
and [Ante-5 outcome accounting](data/long_horizon_gae_ante5_win_accounting_20260921.json)
show how the greedy comparison adds up. GAE alone clears Ante 5 in 50 paired
runs, producing 26 wins; the parent alone clears it in 17 runs, producing 7
wins. Among 266 runs where both clear it, GAE records 176 wins and the parent
186. Those differences sum to the overall nine additional GAE wins. These
groups are selected after each policy's actions, so they do not identify a
causal later-policy effect or establish that comparable builds reached Ante 5.

All eight milestone thresholds are exploratory, with unadjusted pointwise
95% intervals. Equal or improved early survival does not establish better
early decisions: their build quality can affect much later outcomes. The
separate early-policy intervention remains necessary for attribution. The
next endpoint and original selection/acceptance rules remain unchanged.

## Late hand-phase diagnostic

The midgame pattern motivated a bounded
[teacher intervention](data/long_horizon_late_hand_diagnostic_manifest_20260921.json)
using the two fixed terminal-GAE +10 policies. It used the first 25 already
exposed development seeds, greedy actions, CPU batch 1, and two workers.
Each policy ran a student baseline and a hybrid that substitutes the existing
teacher only in `CHOOSE_ACTION` when the **current** Ante is at least 6.
All earlier actions and other phases remain under the learned policy. This
phase boundary includes consumables and Joker sales legal during hand play;
it is not restricted to selecting played or discarded cards.

All 100 physical games completed in 519.3 CPU wall seconds. The
[verified result](data/long_horizon_late_hand_diagnostic_20260921.json)
records 26/50 student wins (52%) and 28/50 hybrid wins (56%). The paired gain
is +4 percentage points [0, +10], averaging the two fixed replicas within
each of 25 game seeds before bootstrap. Each replica gains one net win;
their intervals are [0, +12] and [−8, +16]. The interval does not exclude zero.

The [paired outcomes](data/long_horizon_late_hand_diagnostic_games_20260921.jsonl)
contain three rescued losses and one new loss. Teacher actions differ from
student proposals in 39 games, with 384 replacements across 893 controlled
steps; 314 replacements select a different played subset. Trace verification
checks all phase boundaries, unchanged uncontrolled actions, and 6,339 matched
observations and proposals through the first replacement in each paired run.
The teacher's queries are checked for live-state/RNG isolation.

This small diagnostic does not establish a reliable late-hand benefit or
identify late hand play alone as the main reason midgame survival fails to
convert into wins. It also cannot rule out hand-policy weaknesses: the teacher
has privileged simulator access, is imperfect, and changes later visited
states. These are exposed development games and teacher-assisted outcomes,
not learned-policy generalization. No teacher actions enter training or
change endpoint selection, acceptance, or the untouched early-policy test.

## Early shop-phase diagnostic

A [post-hoc follow-up](data/long_horizon_early_shop_diagnostic_manifest_20260921.json)
uses the same 25 exposed seeds and two fixed terminal-GAE +10 policies. The
teacher replaces actions only in `SHOP` while the **current** Ante is at most
3. The student controls hands, packs, and every other phase. Legal Ante
decreases can reactivate the shop intervention. This tests early shop help
under the student's subsequent policy; it does not train that policy.

All 100 games completed in 576.2 CPU wall seconds, with no stalls or censoring.
The [verified result](data/long_horizon_early_shop_diagnostic_20260921.json)
records 29/50 hybrid wins (58%) versus the same 26/50 student wins (52%). The
paired difference is +6 percentage points [−8, +20], with fixed replicas
averaged within each of the 25 seed clusters. Replica differences are +8
points [−8, +24] and +4 [−16, +24]. These intervals do not establish a benefit.

The [paired outcomes](data/long_horizon_early_shop_diagnostic_games_20260921.jsonl)
contain seven rescued losses and four new losses. Teacher proposals replace
262 actions across 1,225 controlled steps, changing 49 of 50 hybrid games.
The most common replacement is buying to leaving the shop (69 actions),
followed by changing the purchased item (60). These action counts describe
the intervention, not which changes caused wins or losses.

Verification checks every intervention boundary and 524 shared observations
through the first replacement in each pair. All 50 student control games,
including their full 8,746 observation/action trace rows, exactly reproduce
the late-hand diagnostic's controls. The repeated controls do not increase
the number of independent game seeds.

Neither this early-shop result nor the late-hand result identifies a reliable
phase-specific improvement on this small panel. Their point estimates do not
establish that early shops are more important than late hands. Both use a
teacher with privileged simulator access on exposed development seeds;
neither is evidence of learned generalization. The frozen training study,
endpoint selection, and fresh confirmation and early-policy tests continue
unchanged.

## Early-shop gradient consistency

A [frozen-parent diagnostic](data/long_horizon_early_shop_gradient_consistency_manifest_20260921.json)
checks the learning signal directly in the first matched training cohorts.
Both replicas have 64 byte-identical MC/GAE trajectory pairs and identical
saved optimizer row draws. The probe evaluates all 1,370 early-shop decisions
in replica 0 and 1,513 in replica 1, using their original observations,
actions, values, and credit targets. It differentiates only the 1,935
parameters of the macro action-type head, covering choices such as buying,
leaving, rerolling, and selling. It does not measure item selection, the
encoder, critic gradients, or the full optimizer update.

For 1,000 fixed random partitions per replica, each half contains 32 whole
games. The primary comparison normalizes each half's advantages using all
of that half's decisions, then compares its early-shop gradient direction
with the other half. Positive cosine means their directions agree; negative
means they oppose. The secondary comparison uses the saved training-row draw
and the original shared full-cohort normalization.

| Replica | Credit | All eligible rows, independently normalized halves | Saved row draw, shared normalization |
| --- | --- | ---: | ---: |
| 0 | MC | −0.215 | +0.328 |
| 0 | GAE | +0.145 | −0.086 |
| 1 | MC | +0.458 | +0.567 |
| 1 | GAE | −0.422 | −0.133 |

These are median split cosines, not confidence intervals. The partitions
overlap and do not constitute 1,000 independent experiments. The
[recomputed results](data/long_horizon_early_shop_gradient_consistency_20260921.json)
also separate normalization from row selection: holding the original
normalization fixed while using all eligible rows changes the corresponding
medians by at most .024. Using the saved row draw changes agreement more
substantially. Full eligible-row versus saved-draw gradient cosines are
.830/.914 for MC and .657/.497 for GAE in replicas 0/1.

This does not establish that either estimator supplies more reliable or more
correct early-shop credit. The primary ranking reverses across replicas,
and the saved-draw ranking differs from the eligible-row ranking. Agreement
could reflect a consistently wrong direction, and disagreement could reflect
a small true gradient. It therefore does not justify selecting a new credit
method or batch size. The probe uses neither fresh games nor teacher labels
and cannot establish a learned win-rate gain.

The CPU probe completed in 90.3 seconds without optimizer steps or model
writes. Recorded action log probabilities reproduce within 1.51e−5, all
trajectory/checkpoint identities were checked, and both models remained
unchanged. The frozen gameplay experiment and acceptance rules are unchanged.

## Latest control result and optimizer interruption

Terminal-GAE replica 1 has completed both 200-game development panels at +10:
[sampled wins](../../eval/long_horizon_generalization_20260921_micro160/evaluations/development/update10/terminal_gae_seed1/sample0/summary.json)
are 71/200 (35.5%), versus its parent's 76/200 (38%);
[greedy wins](../../eval/long_horizon_generalization_20260921_micro160/evaluations/development/update10/terminal_gae_seed1/greedy/summary.json)
are 95/200 (47.5%), versus 102/200 (51%). Both panels have zero stalls or
censoring. The first replica's positive greedy point estimate therefore did
not repeat. These counts do not establish a replicated value-objective benefit.

The second legacy replica rejected its first attempted update, global 123,
because a physical 160-row minibatch had estimated KL .08066, above the .08
hard threshold. All 50 previously committed updates remain preserved. The
[isolated replay](data/long_horizon_legacy_seed1_kl_replay_20260921.json)
reproduced the rejection from the saved cohort and row order, and verified
that rollback restored both model weights and Adam state exactly.

At that same rejected policy state, the offending batch's mean KL was .08066;
pairing its rows with the adjacent batch gave .04104 over 320 rows, and the
640-row logical optimizer group averaged .02182. The full 8,192-row draw
averaged .00397. The 320-row figure is a reaggregation of recorded per-row
values, not a full optimizer replay with physical batch 320. This identifies
a hard guard whose result depends on physical memory batching. It does not
establish that the policy update is beneficial or that the threshold should
be relaxed. It motivated the measurement correction described below.

The failed training job and diagnostic replay remain charged to the existing
compute allowance. Frozen sources and failure evidence are archived. The
finishing supervisor stopped on the failed training dependency; confirmation
and early-transfer panels remain untouched.

## Logical-KL recovery

The hard guard now measures sample-weighted KL over the logical optimizer group,
both during training and in the epoch-end check. Physical batches bound memory.
The hard threshold remains .08 and the full-draw soft threshold remains .015.
Short final groups use their actual row counts, and rejection still restores
the complete pre-update model and Adam state. Per-physical and per-logical KL
values are both logged.

The [saved-cohort replay](data/long_horizon_logical_kl_replay_20260921.json)
completed all 16,384 optimizer examples and 26 Adam steps in 115.4 optimizer
seconds. Its largest physical mean was .08066, largest logical mean .02182,
final full-draw mean .005415, and final maximum logical mean .03062. It had no
rollback or soft stop. All 64 trajectory hashes and the exact saved draw and
epoch orders were checked. The diagnostic wrote no candidate checkpoint and
does not establish a gameplay gain.

An [accepted-prefix audit](data/long_horizon_logical_kl_prefix_audit_20260921.json)
verified 3,200 trajectory hashes and 819,200 optimizer examples across all 50
committed updates. Every accepted physical mean was below .08, as were the
containing logical means. In this physical-160/logical-640 recipe, epoch checks
keep the same physical chunks and NumPy order. The audit supports retaining
these accepted updates; it is not a replay of all 50 updates.

The [recovery declaration](data/long_horizon_logical_kl_recovery_20260921.json)
retains those five +10 prefixes, while the second legacy replica restarts from
its unchanged update-122 learning state. New checkpoint copies preserve model,
Adam, RNG, update count, seed cursor, and collection counters exactly; their
metadata records both the inherited source and the new execution source.
The recovery manifest pins 3,374 evidence files, including original checkpoint
hashes and exact copies of accepted cohort artifacts. Original failure logs,
sources, checkpoints, and compute ledger remain preserved.

Seventy planned updates remain. The new ledger carries 28,222.96 seconds from
the original 12-hour allowance, after all previous training, failures, and
diagnostic replays. Development evaluation reruns the already exposed seeds
under the new source identity. Selection, acceptance gates, and untouched
confirmation/early-transfer panels are unchanged. No diagnostic weights enter
the candidate pool.

Validation recorded
[202 passing tests](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/logical_kl_validation.log)
covering optimizer behavior, cohort training, and study integrity, including
physical-batch invariance, uneven groups, exact rollback, RNG preservation,
and strict recovery provenance. Ruff and diff checks passed.

The [first real recovered update](data/long_horizon_first_logical_kl_update_20260921.json)
has also committed for legacy replica 1, global update 123. Collection exactly
reproduced all saved arrays from the failed attempt's 64 games and 9,990
decisions, including optimizer row draws. Optimization then completed all
16,384 examples and 26 steps without rollback or early stopping; final
full-draw KL was .005415. Checkpoint policy hash, seed cursor, collection
counters, and Adam step advances were checked against the saved update.
This is a successful recovery, not a measured win-rate gain.

All six branches have now completed +10 training. The
[matched endpoint audit](data/long_horizon_all_arm_endpoint10_20260921.json)
verifies all 3,840 trajectory hashes, the declared seed prefixes, complete
outcome labels, and 983,040 optimizer examples across 60 updates. The recovered
legacy replica contributed 100,281 decisions and 163,840 optimizer examples;
all 318 Adam parameter states advanced 260 steps, with no rollback or early
stop. Its checkpoint policy hash matches the final logged update. Training
has now completed the +20 MC endpoint for replica 0 and advanced to its
terminal-GAE control. Complete gameplay panels and the original validation
gates remain necessary.

The [first repeated baseline panel](data/long_horizon_recovery_baseline_reproduction_20260921.json)
also completed. All 200 sampled games reproduce the original 79 wins, and
every saved non-timing field matches exactly, including action counts, steps,
milestones, and critic forecasts. This checks inference reproducibility after
the recovery; the repeated exposed development seeds are not fresh evidence.
Its [greedy counterpart](data/long_horizon_recovery_baseline_greedy_reproduction_20260921.json)
has also completed, reproducing 91 wins in 200 games and every non-timing
field exactly.
The [second replica's sampled baseline](data/long_horizon_recovery_baseline_seed1_sampled_reproduction_20260921.json)
now reproduces all 200 games and all non-timing fields exactly as well, with
76 wins.
The [second greedy baseline](data/long_horizon_recovery_baseline_seed1_greedy_reproduction_20260921.json)
has now reproduced all 200 games and 102 wins exactly as well. All four
baseline panels—800 games across both replicas and modes—match every original
non-timing field. Evaluation has advanced to the trained +10 candidates.

## First +20 training endpoint

The [replica-0 MC endpoint audit](data/long_horizon_first_endpoint20_training_20260921.json)
verifies global update 142: all 1,280 planned game seeds and trajectory hashes,
211,766 collected decisions, complete outcome labels, and 327,680 optimizer
examples across 20 updates. Every update consumed its full exposure without
KL rollback or early stopping. All 314 active Adam parameter states advanced
520 steps; four unused residual-head states did not advance. The saved policy
digest and cohort counters match the final update log. This verifies completed
training, not better gameplay.

Its already-planned development evaluation completed alongside the main
evaluation queue under a recorded
[execution intent](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/parallel_first_mc20_intent.json).
It uses the same 200 development seeds, sampled repeat 0 and greedy mode,
CPU batch 32, four environment workers, four inference threads, and the frozen
source/checkpoint. The standard evaluator validates and skips completed panels
when its queue reaches this endpoint. This advances the existing 400-game
workload; it adds neither another seed panel nor an independent replication.
Endpoint selection, confirmation, and learned early-policy attribution remain
unchanged and pending.

The already-audited legacy replica-1 +10 endpoint also completed evaluation
concurrently, under a separate
[execution intent](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/parallel_legacy1_u10_intent.json).
This advances the missing 400-game legacy-control comparison with the same
planned settings. There are at most three concurrent CPU evaluations including
the original queue; each writes a distinct policy/endpoint directory. The
original queue will validate and reuse both advanced panels when reached.

The [replica-0 terminal-GAE +20 audit](data/long_horizon_terminal_gae_seed0_endpoint20_training_20260921.json)
now verifies the matched control: 1,280 complete games, 213,789 decisions,
327,680 optimizer examples, all trajectory hashes, and 520 Adam steps without
rollback or early stopping. Its planned sampled and greedy development panels
have completed alongside the main queue and exploratory early-policy test.
The [execution record](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/parallel_gae0_u20_execution.json)
was written after launch: an assertion about the main queue's position was
stale, and the launch was not gated on that check's exit code. Subsequent
verification confirmed the correct checkpoint, source, seeds, settings,
distinct output directories, and three-evaluation concurrency limit. The
active job was preserved; no panel or training settings changed.

The [replica-0 legacy-GAE +20 audit](data/long_horizon_legacy_gae_seed0_endpoint20_training_20260921.json)
has now completed as well: 1,280 complete games, 216,428 decisions, 327,680
optimizer examples, and 520 steps for all 318 active Adam parameter states.
Every trajectory hash and planned seed was checked; all outcomes were labeled,
with no rollback or early stop. All three first-replica arms now have audited
+20 endpoints. The first replica's legacy control has also completed both
development evaluation modes.

The [replica-1 MC +20 audit](data/long_horizon_terminal_mc_seed1_endpoint20_training_20260921.json)
now verifies 1,280 complete games, 205,913 decisions, and 327,680 optimizer
examples. All trajectory hashes and planned seeds match, all outcomes are
labeled, and the 314 active Adam parameter states advanced 520 steps; four
unused residual-head states did not advance. There was no rollback or early
stop. Both MC replicas have completed development evaluation.

The [replica-1 terminal-GAE +20 audit](data/long_horizon_terminal_gae_seed1_endpoint20_training_20260921.json)
also verifies full training exposure: 1,280 complete games, 205,352 decisions,
327,680 optimizer examples, and 520 steps for each of 314 active Adam parameter
states. All planned seeds, trajectory hashes, completed labels, policy digest,
and checkpoint counters match. No update rolled back or stopped early. This
brings audited updates to 110/120. Both of its development evaluation modes
have now completed.

The [final legacy +20 audit](data/long_horizon_legacy_gae_seed1_endpoint20_training_20260921.json)
verifies 1,280 complete games, 205,252 decisions, 327,680 optimizer examples,
and 520 steps for all 318 active Adam states, with every planned seed,
trajectory hash, label, policy digest, and counter checked. No update rolled
back or stopped early. All six +20 endpoints are now audited: 120 updates,
7,680 completed trajectory games, and 1,966,080 optimizer examples. Game seeds
are paired across arms; those trajectory counts are not independent seed counts.
The training supervisor exited successfully. The final legacy development
evaluation remains in progress.

## First +20 development results

The first replica's
[complete sampled panel](data/long_horizon_mc_endpoint20_seed0_sample0_20260921.json)
records 91/200 MC wins at +20 (45.5%), compared with 72/200 at +10 (36%) and
79/200 for its parent (39.5%). All panels use the same 200 development seeds
and sampling repeat 0, with no stalls or censoring. Checkpoint identities,
source identity, and evaluation settings were verified before comparison.

The paired +20-minus-+10 difference is +9.5 percentage points, with a pointwise
95% interval of [+2, +17]. Against the parent it is +6 points [−1.5, +13.5],
which still includes zero. The
[paired game rows](data/long_horizon_mc_endpoint20_seed0_sample0_games_20260921.jsonl)
contain 35 rescued parent losses and 23 new losses, or 41 rescues and 22 new
losses against +10. Recomputing 10,000 seed-paired bootstrap samples directly
from the exported rows reproduces both intervals.

The [greedy panel](data/long_horizon_mc_endpoint20_seed0_greedy_20260921.json)
has also completed, with no stalls or censoring:

| Policy | Sampled wins | Greedy wins |
| --- | ---: | ---: |
| Parent | 79/200 (39.5%) | 91/200 (45.5%) |
| MC +10 | 72/200 (36%) | 100/200 (50%) |
| MC +20 | 91/200 (45.5%) | 100/200 (50%) |
| Terminal-GAE +10 | 79/200 (39.5%) | 107/200 (53.5%) |
| Terminal-GAE +20 | 80/200 (40%) | 102/200 (51%) |
| Legacy-GAE +20 | 70/200 (35%) | 105/200 (52.5%) |

The greedy +20-minus-parent difference is +4.5 points [−2.5, +11.5], while
+20-minus-+10 is 0 points [−7, +7]. Equal total wins do not mean unchanged
behavior: the [paired greedy rows](data/long_horizon_mc_endpoint20_seed0_greedy_games_20260921.jsonl)
contain 24 rescued +10 losses and 24 new losses. Both intervals and all counts
were independently recomputed from the exports.

The sampled improvement over +10 is a positive development result, but neither
mode establishes a gain over the parent. Both matched control modes have now
completed as described below; fresh confirmation and the reserved learned
early-decision intervention remain pending.
Reused parent and +10 panels do not add independent game seeds. The original
pooled MC-versus-GAE endpoint-selection rule and all acceptance gates remain
unchanged.

The second MC replica's
[sampled +20 panel](data/long_horizon_mc_endpoint20_seed1_sample0_20260921.json)
is now complete: 81/200 wins (40.5%), versus its parent's 76/200 (38%) and
71/200 (35.5%) at +10. The paired parent comparison is +2.5 percentage points
[−5.5, +10], with 34 rescued losses and 29 new losses. Against +10 it is
+5 points [−2.5, +12.5]. Its
[greedy +20 panel](data/long_horizon_mc_endpoint20_seed1_greedy_20260921.json)
records 102/200 wins, matching the parent's 102/200 and comparing with 101/200
at +10. The paired differences are 0 points [−6.5, +6.5] against the parent
and +0.5 [−5.5, +6.5] against +10. Both modes contain all 200 planned games
without stalls or censoring, with verified checkpoint identities and matching
evaluation settings. These are pointwise development intervals. Both MC
replicas have positive sampled point estimates against their parents, but
neither establishes that gain. The second matched sampled control is now
reported below, together with its greedy comparison; fresh tests remain necessary.

The first replica's completed legacy control supplies a separate check of the
value-objective change. In the
[sampled comparison](data/long_horizon_value_endpoint20_seed0_sample0_20260921.json),
terminal-GAE wins 80/200 versus legacy-GAE's 70/200. Its paired difference is
+5 percentage points [−2.5, +12.5], with 35 rescued losses and 25 new losses.
In the [greedy comparison](data/long_horizon_value_endpoint20_seed0_greedy_20260921.json),
terminal-GAE wins 102/200 versus legacy-GAE's 105/200: −1.5 points [−8.5, +5.5],
with 24 rescues and 27 new losses. Neither mode resolves the value-objective
effect. This comparison changes both the value used by GAE and the residual
value loss; it cannot identify either component's separate effect.

Legacy continuation itself changes sampled wins by −4.5 points [−12.5, +3.5]
against the parent, but greedy wins by +7 points [+0.5, +13.5]. The latter is a
positive pointwise development result for this fixed replica, showing that a
greedy improvement is not specific to MC. It does not establish replicated or
fresh generalization, and the sampled comparison remains unresolved. Both
reports verify equal completed training exposure, model/settings identities,
and independently recomputed paired intervals; reused panels add no new seeds.

The [matched +20 sampled comparison](data/long_horizon_matched_endpoint20_seed0_sample0_20260921.json)
records 80/200 wins (40%) for terminal-GAE, versus MC's 91/200 (45.5%). Both
models received 1,280 complete training games and 327,680 optimizer examples,
with no rollback or early stop. The paired MC-minus-GAE difference is +5.5
percentage points [−2.5, +13.5]: 40 GAE losses become MC wins, while 29 GAE
wins become MC losses. The interval includes zero. Terminal-GAE changes by
only +0.5 points from both its parent [−6.5, +8] and its +10 endpoint [−7, +8].
These results do not yet distinguish a reliable credit-estimator benefit from
development-panel variation.

The [five-policy paired rows](data/long_horizon_matched_endpoint20_seed0_sample0_games_20260921.jsonl)
include the parent and both credit arms at +10 and +20. The audit verifies
checkpoint recipes, equal evaluation settings, complete uncensored panels,
training-audit identities, and all input hashes. Independent 10,000-draw paired
bootstrap calculations reproduce all five reported intervals. This is one
fixed training replica on exposed development seeds; the intervals are
pointwise and do not replace the planned pooled selection or fresh tests.

The [matched +20 greedy comparison](data/long_horizon_matched_endpoint20_seed0_greedy_20260921.json)
records 102/200 terminal-GAE wins (51%), versus MC's 100/200 (50%). MC-minus-GAE
is −1 point [−8, +6], with 24 rescues and 26 new losses. Terminal-GAE-minus-parent
is +5.5 points [−1.5, +12.5], and GAE +20-minus-+10 is −2.5 [−9.5, +4.5]. All
three intervals include zero. The
[greedy paired rows](data/long_horizon_matched_endpoint20_seed0_greedy_games_20260921.jsonl)
and checkpoint/training checks pass the same independent audit as sampled play.
The positive sampled MC point estimate therefore has no corresponding greedy
advantage over matched GAE in this replica.

The second legacy replica's
[+10 sampled control](data/long_horizon_legacy_seed1_endpoint10_sample0_20260921.json)
has also completed: 74/200 wins (37%), versus its parent's 76/200 (38%). The
paired difference is −1 percentage point [−8, +6], with 24 rescued losses and
26 new losses. Both panels have no stalls or censoring. The
[exported game pairs](data/long_horizon_legacy_seed1_endpoint10_sample0_games_20260921.jsonl)
independently reproduce the interval and counts. Its
[greedy control](data/long_horizon_legacy_seed1_endpoint10_greedy_20260921.json)
also completed: 101/200 wins (50.5%), versus the parent's 102/200 (51%), paired
−0.5 points [−7.5, +6.5]. The
[greedy game pairs](data/long_horizon_legacy_seed1_endpoint10_greedy_games_20260921.jsonl)
contain 24 rescues and 25 new losses, with no stalls or censoring. Neither mode
shows a resolved gain at +10. The completed +20 legacy-control comparisons are
included in the complete development report below.

## Replicated +20 sampled comparison

The [second matched sampled control](data/long_horizon_matched_endpoint20_seed1_sample0_20260921.json)
has completed with 81/200 GAE wins (40.5%), exactly matching MC's total. The
paired MC-minus-GAE difference is 0 points [−7.5, +7.5], with 28 MC-only wins
and 28 GAE-only wins. Both algorithms increase from 71/200 at +10 to 81/200 at
+20 in this replica. Equal totals do not imply identical behavior, but the
sampled point estimate supplies no MC advantage here.

The [pooled sampled report](data/long_horizon_replicated_endpoint20_sample0_20260921.json)
combines the two audited replicas on 200 shared game seeds:

| Policy | Sampled wins |
| --- | ---: |
| Starting policies | 155/400 (38.75%) |
| MC +10 | 143/400 (35.75%) |
| MC +20 | 172/400 (43%) |
| Terminal-GAE +10 | 150/400 (37.5%) |
| Terminal-GAE +20 | 161/400 (40.25%) |

MC-minus-GAE at +20 is +2.75 percentage points [−2.75, +8.25], and MC-minus-parent
is +4.25 [−1, +9.5]. Both intervals include zero. MC's increase from +10 is
+7.25 points [+2, +12.5], while GAE's is +2.75 [−2.25, +8]. Thus longer training
improves MC relative to its earlier checkpoint on these development games,
without establishing a gain over either matched GAE or the parents. A positive
interval for one training trajectory and an unresolved interval for another
are not themselves evidence that the trajectories differ.

The [exported seed clusters](data/long_horizon_replicated_endpoint20_sample0_games_20260921.jsonl)
average the two fixed training replicas within each game seed before 10,000
paired bootstrap draws. Independent recomputation verifies all five contrasts,
and the source reports' model, training, and gameplay hashes are checked again.
The 400 games per policy are 200 independent seed clusters, not 400 independent
seeds or additional evaluations. These pointwise development intervals do not
replace the fixed endpoint-selection rule, fresh confirmation, or reserved
early-policy attribution test. All legacy evaluations are now complete.

## Replicated +20 greedy comparison

The [second matched greedy panel](data/long_horizon_matched_endpoint20_seed1_greedy_20260921.json)
records 102/200 MC wins and 91/200 GAE wins. MC-minus-GAE is +5.5 points
[−1.5, +12.5], with 31 rescued losses and 20 new losses. GAE-minus-parent is
−5.5 points [−12.5, +1], and GAE +20-minus-+10 is −2 points [−8.5, +4.5].
These intervals all include zero.

The [pooled greedy report](data/long_horizon_replicated_endpoint20_greedy_20260921.json)
has 202/400 MC wins (50.5%), 193/400 GAE wins (48.25%), and 193/400 parent wins
(48.25%). MC-minus-GAE is +2.25 points [−3, +7.5], and MC-minus-parent is
+2.25 [−2.5, +6.75]. MC's change from +10 is +0.25 [−4.5, +5], while GAE's
is −2.25 [−6.75, +2.25]. No greedy comparison establishes an improvement.
As in sampled play, all intervals average the two fixed replicas within the
200 shared game seeds. Input hashes, matched exposure, and the
[paired clusters](data/long_horizon_replicated_endpoint20_greedy_games_20260921.jsonl)
were independently checked. Fresh confirmation and early-policy attribution
remain necessary.

## Complete development report and frozen selection

The [complete +20 report](data/long_horizon_complete_endpoint20_development_20260921.json)
contains all four policies and both fixed replicas:

| Policy | Sampled wins | Greedy wins |
| --- | ---: | ---: |
| Starting policies | 155/400 (38.75%) | 193/400 (48.25%) |
| Terminal MC | 172/400 (43%) | 202/400 (50.5%) |
| Terminal GAE | 161/400 (40.25%) | 193/400 (48.25%) |
| Legacy GAE | 153/400 (38.25%) | 203/400 (50.75%) |

Terminal-GAE-minus-legacy is +2 percentage points [−3.00625, +7.25] sampled
and −2.5 [−7, +2] greedy. Both intervals include zero, leaving the value-objective
bundle effect unresolved. The second legacy replica wins 83/200 sampled and
98/200 greedy games. Its matched GAE differences are −1 [−8, +6] sampled and
−3.5 [−10, +3] greedy, as recorded in the
[sampled](data/long_horizon_value_endpoint20_seed1_sample0_20260921.json) and
[greedy](data/long_horizon_value_endpoint20_seed1_greedy_20260921.json) comparisons.

An [independent export audit](data/long_horizon_complete_endpoint20_verification_20260922.json)
verified all 16 outcome artifacts, every win count, all eight pooled paired
contrasts, and the frozen selection hashes. Uncertainty uses 200 game-seed
clusters with the two training replicas averaged within each seed.
The [selection record](data/long_horizon_endpoint_selection_20260922.json)
chooses +20 because its development sampled MC-minus-GAE difference is +2.75
points, versus −1.75 at +10. This is the original selection rule, independent
of whether the development interval excludes zero. Selection does not establish
generalization. The live coordinator has passed its real process-isolation
regression and started the unchanged fresh confirmation panels.

## Replicated fresh matched comparison

The [complete sampled comparison](data/long_horizon_fresh_replicated_credit_sampled_20260922.json)
includes both fixed sampling repeats and both training replicas. All policies
play the same 400 confirmation seeds, disjoint from training and development.
The +20 endpoint was selected before confirmation began.

| Policy | Replica 0 wins | Replica 1 wins | Combined win rate |
| --- | ---: | ---: | ---: |
| Starting policy | 277/800 | 295/800 | 35.75% |
| Terminal GAE +20 | 317/800 | 298/800 | 38.4375% |
| Terminal MC +20 | 307/800 | 315/800 | 38.875% |

| Paired comparison | Difference, percentage points | Pointwise 95% interval |
| --- | ---: | ---: |
| MC minus terminal GAE | +0.4375 | [−2.1875, +3.1875] |
| MC minus parent | +3.125 | [+0.625, +5.625] |
| Terminal GAE minus parent | +2.6875 | [+0.0609375, +5.375] |

Both continued-training arms have positive pooled intervals against their parents
on these fresh games. The matched MC-minus-GAE interval includes zero, and its
point estimates differ in direction
across replicas (−1.25 points in replica 0; +2.125 in replica 1). These parent
comparisons combine additional training with the outcome-only value objective;
they do not distinguish those shared changes by themselves. The completed sampled
legacy-control comparison below now reproduces the parent gain with the original
training objective.

The frozen sampled acceptance conditions **fail**. MC improves by less than five
points against both controls and does not improve over GAE in both replicas.
A positive parent interval does not satisfy those conditions. The completed greedy
results below pass the regression guardrails. The reserved early-policy test remains
pending; it cannot reverse these completed sampled-gate failures. All planned evaluations continue
without changing thresholds, checkpoint selection, or the original goal.

The [exported paired clusters](data/long_horizon_fresh_replicated_credit_sampled_games_20260922.jsonl)
contain all three policies' outcomes for both replicas and both repeats. Four
outcomes per policy are averaged within each of **400 game-seed clusters** before
10,000 bootstrap draws with seed 1729. There are 1,600 plays per policy, not 1,600
independent seeds. These intervals quantify procedural-game uncertainty conditional
on the two trained replicas. They are pointwise intervals; earlier interim
inspections did not change selection, samples, or stopping.

All twelve underlying panels are complete without stalls or censoring. Checkpoint,
source, selection, evaluation, training, and raw-outcome hashes passed the
individual audits and were rechecked by the pooled calculation. An independent
bootstrap agrees with the production statistics for all three contrasts. The
second-repeat audits for
[replica 0](data/long_horizon_fresh_credit_seed0_sample1_20260922.json) and
[replica 1](data/long_horizon_fresh_credit_seed1_sample1_20260922.json) record
MC-minus-GAE differences of −2.75 points [−8, +2.25] and +2.5 [−3, +7.75].

The preserved [first-repeat snapshot](data/long_horizon_fresh_replicated_credit_sample0_20260922.json)
was +1 point [−2.753125, +4.875] for MC-minus-GAE. Its larger point estimate is
superseded by the complete two-repeat comparison above. The architectural
hypothesis remains open: these data establish a modest fresh-game improvement
over the parents, not that better early decisions or the credit change caused it.
The requested full generalization claim remains unproven.

## Complete fresh value-control comparison

The [complete sampled value-control report](data/long_horizon_fresh_replicated_value_sampled_20260922.json)
adds both legacy replicas and both sampling repeats on the same 400 confirmation
seeds. Legacy GAE wins 295/800 games in replica 0 and 327/800 in replica 1:
622/1,600 overall, or **38.875%**, the same aggregate win rate as MC. Their
individual game outcomes differ; equal totals do not establish policy equivalence.

| Paired comparison | Difference, percentage points | Pointwise 95% interval |
| --- | ---: | ---: |
| Legacy GAE minus parent | +3.125 | [+0.6875, +5.5] |
| Terminal GAE minus legacy GAE | −0.4375 | [−3, +2.125] |
| MC minus legacy GAE | 0 | [−2.5, +2.625] |

The legacy-minus-parent point estimates are positive in both replicas: +2.25 and
+4 points. Terminal-GAE-minus-legacy changes direction across replicas: +2.75 and
−3.625 points. Continued training under the original objective therefore produces
a small gain on these unseen procedural games. Neither the full-episode credit
treatment nor the outcome-only value objective shows a resolved advantage over
that control. The intervals still allow modest differences; this is not evidence
that the methods are equivalent or that a particular architectural defect caused
the plateau. The original MC sampled acceptance gates remain failed, and learned
early-decision attribution remains unproven.

The [paired clusters](data/long_horizon_fresh_replicated_value_sampled_games_20260922.jsonl)
contain four outcomes per policy for each of 400 game seeds. The audit checks
the new control panels against the frozen checkpoints, recipes, evaluation
settings, and training preflight. It independently recomputes all six comparisons
for each replica and the pooled result: 18 paired estimates and bootstrap
intervals. The three previously published pooled contrasts reproduce exactly.
All sixteen sampled panels are complete without stalls or censoring, and the
report and paired exports are byte-identical to the audited artifacts. These
pointwise intervals are conditional on the two fixed trained replicas. No games,
selection changes, or stopping-rule changes were introduced by the audit.
The legacy greedy comparisons are complete below. The reserved early-policy test
is running and has no completed comparison yet.

## Complete fresh greedy comparison

The [complete greedy control report](data/long_horizon_fresh_replicated_value_greedy_20260922.json)
now covers all four policies and both fixed replicas on all 400 confirmation seeds.

| Policy | Replica 0 wins | Replica 1 wins | Combined win rate |
| --- | ---: | ---: | ---: |
| Starting policy | 195/400 | 206/400 | 50.125% |
| Terminal GAE +20 | 206/400 | 191/400 | 49.625% |
| Terminal MC +20 | 213/400 | 207/400 | 52.5% |
| Legacy GAE +20 | 186/400 | 205/400 | 48.875% |

| Paired comparison | Difference, percentage points | Pointwise 95% interval |
| --- | ---: | ---: |
| MC minus terminal GAE | +2.875 | [−0.75, +6.5] |
| MC minus parent | +2.375 | [−0.875, +5.75] |
| Terminal GAE minus parent | −0.5 | [−3.75, +2.75] |
| MC minus legacy GAE | +3.625 | [0, +7.25] |
| Terminal GAE minus legacy GAE | +0.75 | [−2.625, +4.125] |
| Legacy GAE minus parent | −1.25 | [−4.5, +2.125] |

No pooled MC interval excludes zero. The MC-minus-GAE and MC-minus-parent lower
bounds exceed −5 points, so the
frozen greedy regression guardrails pass; this does not rescue the failed sampled
acceptance conditions. The [paired clusters](data/long_horizon_fresh_replicated_value_greedy_games_20260922.jsonl)
average both replicas within each of 400 game seeds before bootstrap. Independent
statistics and source, checkpoint, settings, raw-outcome, and export checks pass.
All 18 estimates and intervals (six contrasts, each pooled and per replica) were
independently recomputed; the three earlier pooled contrasts reproduce exactly.
MC-minus-legacy is +6.75 points [+1.75, +11.75] in replica 0 but only +0.5
[−4.5, +5.5] in replica 1. This does not establish a replicated greedy advantage.
The original-objective control's positive sampled gain is not reproduced as a
positive greedy gain.

The matched individual audits record MC-minus-GAE differences of
[+1.75 points in replica 0](data/long_horizon_fresh_credit_seed0_greedy_20260922.json)
[−3.25, +7] and
[+4 points in replica 1](data/long_horizon_fresh_credit_seed1_greedy_20260922.json)
[−1.25, +9.25]. The
[second replica's parent comparison](data/long_horizon_fresh_mc_parent_seed1_greedy_20260922.json)
is +0.25 points [−4.5, +5], with 47 rescued losses and 46 new losses.

## Complete confirmation verification

The [canonical confirmation report](data/long_horizon_complete_confirmation_20260922.json)
contains all 24 completed panels and the frozen +20 endpoint. Its
[independent verification](data/long_horizon_complete_confirmation_verification_20260922.json)
checks every outcome artifact against the completed audits, matches all four
canonical comparisons in both modes for each replica and the pooled result,
and independently recomputes the original gameplay thresholds. That covers 24
paired win estimates and intervals. Training evidence is unchanged from the
audited development report, and the original selection rule still chooses +20.

All four canonical gameplay gates fail. The sampled gain is below five points
in every comparison; other failures remain as documented above. All four greedy
regression conditions pass. This verifies the outcome of the fixed experiment;
it does not establish the requested generalization claim. The reserved
early-policy intervention also failed its gate, as documented below. The completed
shop action-ranking diagnostic is inconclusive and does not establish a new
policy improvement.

## Complete early-decision attribution and final outcome

The [completed early-transfer report](data/long_horizon_early_transfer_20260922.json)
covers all eight panels: 400 reserved game seeds, two fixed training replicas,
and two sampling repeats for each policy. The hybrid uses MC in every phase
while the current Ante is at most 3 and the matched GAE policy afterward;
the control uses that same GAE policy throughout. An Ante decrease can reactivate
the early policy.

| Comparison | Hybrid wins | All-GAE wins | Paired gain, percentage points | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Both replicas | 635/1,600 (39.6875%) | 609/1,600 (38.0625%) | +1.625 | [−1.125, +4.4375] |
| Replica 0 | 317/800 | 308/800 | +1.125 | [−2.75, +5.125] |
| Replica 1 | 318/800 | 301/800 | +2.125 | [−1.75, +6.125] |

The [independent audit](data/long_horizon_early_transfer_verification_20260922.json)
verified all 3,200 uncensored terminal outcomes, exact paired seed/repeat coverage,
frozen source and checkpoint hashes, evaluation settings, and actual routing
counts. Because each original hybrid object was reused across repeats, the audit
subtracts cumulative route snapshots and checks each increment against that
panel's action count. It independently reconstructs the intervals with 10,000
seed-cluster bootstrap draws after averaging repeats and the two fixed replicas.
The [paired game export](data/long_horizon_early_transfer_games_20260922.jsonl)
contains the evidence; the audit generated no new games.

Both replica point estimates are positive, but the pooled lower bound is below
zero. The required early-decision gate therefore fails. The
[final report](data/long_horizon_final_results_20260922.json) records all three
main gates as false: gain over matched GAE, gain over the frozen parent, and
early decisions improving final wins. The experiment is complete; it does not
establish a learned long-horizon advantage. A separate matched observation
experiment tests the verified input omissions under the same acceptance bar.

## First fresh confirmation comparison

The [combined sampled result for replica 0](data/long_horizon_fresh_mc_parent_seed0_sampled_20260922.json)
now includes both fixed sampling repeats on all 400 confirmation seeds. The
+20 endpoint was already frozen by the development selection rule.

| Policy | Wins across both repeats | Win rate |
| --- | ---: | ---: |
| Starting policy | 277/800 | 34.625% |
| MC +20 | 307/800 | 38.375% |

The combined paired difference is +3.75 percentage points, with a pointwise 95%
bootstrap interval of [+0.25, +7.125]. Both repeats are averaged within each game
seed before resampling: these are **400 seed clusters**, with 800 games per
policy. The [exported clusters](data/long_horizon_fresh_mc_parent_seed0_sampled_games_20260922.jsonl)
contain both policies' two outcomes per seed and reproduce the estimate.

The [first sampling repeat](data/long_horizon_fresh_mc_parent_seed0_sample0_20260922.json)
records 162/400 MC wins versus 139/400 parent wins: +5.75 points
[+0.75, +10.75], with 65 rescued losses and 42 new losses. The
[second sampling repeat](data/long_horizon_fresh_mc_parent_seed0_sample1_20260922.json)
records 145/400 versus 138/400: +1.75 points [−3.25, +6.75], with 55 rescued
losses and 48 new losses. The second repeat's interval includes zero. Its
[400 paired rows](data/long_horizon_fresh_mc_parent_seed0_sample1_games_20260922.jsonl)
and the [first repeat's pairs](data/long_horizon_fresh_mc_parent_seed0_sample0_games_20260922.jsonl)
are exported separately. The combined result is smaller than the first snapshot.

All four panels are complete, with no stalls or censoring. Checkpoint and source
hashes, exact confirmation seeds, evaluation settings, completed training exposure,
and frozen selection were verified. Independent 10,000-draw seed bootstraps with
seed 1729 agree with the production calculations for each repeat and their
combination. The pooled audit rechecked both source reports and all raw artifacts.

The [completed greedy comparison](data/long_horizon_fresh_mc_parent_seed0_greedy_20260922.json)
records 213/400 MC wins (53.25%) versus 195/400 parent wins (48.75%). The paired
difference is +4.5 points [−0.25, +9.5], with 57 rescued losses and 39 new losses.
The [400 greedy pairs](data/long_horizon_fresh_mc_parent_seed0_greedy_games_20260922.jsonl)
and independent bootstrap reproduce the result, with the same checkpoint,
settings, source, and training checks as sampled play. Neither panel has stalls
or censoring. The interval includes zero, so it does not establish a greedy
benefit. Its lower bound is above the study's −5-point guardrail for this fixed
replica. The complete pooled guardrails above now also pass.

This is a positive preliminary result on games disjoint from training and
development, conditioning on one fixed training replica. Its +3.75-point difference
is below the study's five-point target at this interim stage. Its interval is not
adjusted for inspecting multiple partial confirmation panels; the study continues
with all original panels and acceptance criteria. The parent received no additional
training, so this contrast does not isolate the credit estimator. The complete
matched sampled results above show no resolved MC advantage over GAE. The pooled
greedy guardrails pass; the reserved early-policy intervention remains pending. This result
does not establish an architectural fix or show that improved early decisions
caused the terminal-win gain.

The [second replica across both sampling repeats](data/long_horizon_fresh_mc_parent_seed1_sampled_20260922.json)
records 315/800 MC wins (39.375%) versus 295/800 parent wins (36.875%):
+2.5 points [−1.25, +6.5]. Its
[second repeat](data/long_horizon_fresh_mc_parent_seed1_sample1_20260922.json)
is 162/400 versus 144/400, +4.5 points [−0.75, +9.75]. Both individual replicas
have positive parent point estimates, but replica 1's combined interval includes
zero. The pooled parent comparison and matched-credit limitations appear above.

## Architectural follow-up

A separate [observation counterexample](pack_enhancement_observation_20260922.md)
confirms a concrete input defect: swapping equal-cost Glass and Mult cards between
two Standard-pack slots leaves all 14 policy observation arrays identical, while
claiming those slots produces different permanent deck cards and downstream
scores. Enhancement identity is omitted from pack tokens and their embedding.
The paired constructed fixtures establish lost decision information, not its
frequency, win-rate impact, or responsibility for the plateau. No frozen source,
policy or active evaluation was changed.

Two more [consumable fixtures](consumable_observations_20260922.md) reproduce
identical complete policy observations with different action effects: The Fool
creates Mercury or Hermit depending on prior use order, and selling Mercury pays
$9 or $1 depending on acquisition timing relative to Gift Card payouts. These
are missing current-state fields, not a measured explanation for the win plateau.
All three corrections are implemented in an isolated source copy with explicit
checkpoint and tokenizer compatibility; the live study remains unchanged. The
[candidate verification](data/observation_fix_candidate_verification_20260922.json)
records 110 targeted Python checks, additional checkpoint edge checks, exact
legacy parameter/output preservation, and byte-identical Python/native tokenizer
observations across 73 synthetic fixtures (1,022 arrays). Native tokenizer tests
also passed. Independent static review found no blocking source issue. These
checks establish implementation behavior, not a learning or win-rate gain.
The record also discloses two accidentally selected tiny CPU training fixtures
with mocked evaluation; they used temporary artifacts and did not modify the
study. Live ingestion still lacks the Fool target, and resale values above
32767 saturate. The candidate has not been applied to the primary source.

The [matched learning experiment](observation_generalization_plan_20260922.md)
uses identical corrected architectures, hiding only the three new inputs in its
control. All 32 prescribed training updates, final functional control proofs and
the amended training audit and endpoint lock are complete. Held-out evaluation
is running; its interrupted greedy panels were archived and restarted at whole
panel boundaries as documented in the linked plan. No generalization gain has
yet been established.

The completed [early-shop representation probe](representation_probe_20260922.md)
tests whether averaging all contextual token states obscures useful build
information. It uses 1,280 saved games and equal-sized frozen-feature predictors
with disjoint whole-game training, validation, and test splits. Equal weighting
of token types changes held-out win NLL by +0.00276 [−0.00179, +0.00743] versus
the existing mean, with slightly worse point estimates in both replicas. The
prespecified probe gate fails. This result supports neither changing pooling
nor claiming an architectural solution. It leaves action-conditioned delayed
consequences and richer learned representations as open questions. No source
policy, training recipe, gameplay panel, or selection criterion changed.

The completed [paired early-shop diagnosis](shop_counterfactual_20260922.md)
forced a Joker purchase or leaving from 32 selected training states, followed by
the same frozen continuation policy for four repeats per action. Preferences
were fixed before outcomes. Policy-preferred-minus-other is +4.6875 percentage
points [−7.03125, +17.1875]; buy-minus-leave is +4.6875 [−7.03125, +16.40625];
critic-preferred-minus-other is +6.25 [−5.46875, +17.96875]. All intervals include
zero. The primary classification is unresolved, so the diagnostic establishes
neither poor ranking nor reliable alignment on these pairs. The critic's larger
point estimate does not establish superiority over the actor.

The independent auditor replayed all 256 primary continuations and four duplicate
controls, covering 37,030 decisions. All recorded actions were legal, and every
observation, reward, terminal flag, and outcome matched exactly. All three
statistics were independently reproduced and their exports verified. The result
still concerns only 32 reused training states with a restricted binary choice;
it does not identify an architectural root cause or demonstrate a new learned
policy gain. No new policy was trained for this diagnostic.

## Exploratory learned early-policy test

The completed first-replica MC panels motivated a parent-relative test on the
same 200 exposed development seeds, sampled repeat 0. The baseline uses the
starting policy throughout. The experimental arm uses MC +20 for all phases
at **current** Ante 3 or lower and the starting policy elsewhere. Both arms
use the same routing wrapper; legal Ante decreases can reactivate the early
component. There are 400 new physical evaluation games and no additional
independent seed panel. The test completed in 1,920.0 CPU wall seconds under
its 90-minute limit, with batch 32, four workers, and four inference threads.
It used no accelerator time.

Its [prepared manifest](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/early_mc20_parent_development/manifest.json)
pins the models, source, scripts, seeds, and post-hoc motivation. The initial
MC branch and the original legacy parent have exactly equal state dictionaries;
their only configuration difference is the critic's return-value mode. Their
actor outputs matched exactly on 32 saved real observations. The early/late
routing also matched exact component calls with matching subgroup batch shapes
and consumed no sampling randomness.

An initial preflight comparison against full 32-row component calls found a
small numerical mismatch and was preserved before revising the reference.
Matching subgroup batch shapes resolves that comparison exactly; full-batch
differences up to 9.54e−6 remain recorded. The core routing implementation and
numerical precision were unchanged. The completed test also compared its
all-parent gameplay with the original parent panel.

The [completed all-parent control check](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/early_mc20_parent_baseline_reproduction.json)
now reproduces all 200 original games exactly for wins, action counts, game
length, final/maximum Ante, cleared Antes, and final Jokers: 79 wins and 34,469
decisions. The wrapper routed 12,044 early and 22,425 continuation decisions,
matching the total game steps. Critic forecasts are excluded because the value
modes differ. This supports the control setup.

The [completed hybrid result](data/long_horizon_early_mc20_parent_development_20260921.json)
is 74/200 wins (37%), versus the all-parent control's 79/200 (39.5%). The paired
difference is −2.5 percentage points [−10, +5], with 28 rescued losses and 33
new losses. All 400 games completed without stalls or censoring; both base
models remained unchanged. The hybrid routed 11,976 early and 22,456 later
decisions, accounting for all 34,432 game steps. The
[exported pairs](data/long_horizon_early_mc20_parent_development_games_20260921.jsonl)
and [manifest](data/long_horizon_early_mc20_parent_development_manifest_20260921.json)
preserve the inputs and independently reproduced interval.

This does not demonstrate a benefit from learned early choices with the parent
continuation. It also does not establish harm, or prove that the full model's
higher sampled point estimate comes only from later play: early choices can
interact with the continuation policy. The test uses learned policies without
teacher access or further training. It remains exploratory and parent-relative
on exposed data, not the reserved fresh MC-versus-GAE attribution test. Endpoint
selection and acceptance criteria are unchanged.

## Continuation dependence diagnostic

The early-only result motivated completion of the four policy combinations on
the same exposed 200 seeds, with the same current-Ante gate and sampling repeat.
The new 400-game CPU run uses MC +20 throughout through the common wrapper,
then uses the parent for early decisions and MC +20 later. Combined with the
two completed panels, this estimates the early-policy effect under each fixed
continuation, the later-policy effect under each fixed early policy, and their
paired interaction. The all-MC wrapper is also compared with the original full
MC panel to identify any numerical routing differences.

The [prepared plan](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/continuation_mc20_parent_development/manifest.json)
pins the prior result as its post-hoc motivation, all five contrasts, the
interaction formula, input hashes, models, source, and script. Both new routing
combinations matched exact component calls on 32 saved observations with
matching subgroup batch shapes and consumed no sampling randomness. The run
uses batch 32, four workers, four inference threads, a 90-minute CPU wall cap,
and zero accelerator time. It supplies no training labels or model-selection
input. These are additional physical evaluations of existing seeds, not a new
independent panel or fresh confirmation.

The [full-MC wrapper control](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/continuation_mc20_baseline_reproduction.json)
has completed and reproduces the original 91/200 wins exactly on the checked
gameplay fields: wins, steps, final/maximum Ante, cleared Antes, action counts,
and final Jokers. Its 11,964 early and 23,000 continuation routes account for all
34,964 decisions. The remaining arm and the
[independent four-policy audit](data/long_horizon_continuation_mc20_parent_development_20260921.json)
have now completed. The additional 400 games took 1,703.8 CPU wall seconds,
with no stalls or censoring and no model changes.

| Early policy, current Ante <= 3 | Later policy | Wins |
| --- | --- | ---: |
| Parent | Parent | 79/200 (39.5%) |
| MC +20 | Parent | 74/200 (37%) |
| Parent | MC +20 | 82/200 (41%) |
| MC +20 | MC +20 | 91/200 (45.5%) |

With MC continuation fixed, the early-policy difference is +4.5 percentage
points [−3, +12], with 34 rescues and 25 new losses. With parent early decisions
fixed, the later-policy difference is +1.5 points [−6, +9]. Neither establishes
a benefit. With MC early decisions fixed, switching the continuation raises
wins by +8.5 points [+1, +16], but those two point estimates were already known
before this follow-up; the repeated full-MC control adds no independent seeds.
The interaction is +7 points [−3.5, +17.5], so continuation dependence is also
unresolved. These are pointwise intervals on a post-hoc development diagnostic,
not fresh confirmation of an early-policy gain or a phase-specific bottleneck.

The [four-policy game pairs](data/long_horizon_continuation_mc20_parent_development_games_20260921.jsonl)
and [execution manifest](data/long_horizon_continuation_mc20_parent_development_manifest_20260921.json)
record all inputs. The audit verifies matching evaluation settings across the
four panels, all hashes and route totals, unchanged base models, full-MC control
reproduction, and independent recomputation of every contrast and interaction.

## Execution

The first user-requested process pause was resumed from the same in-memory evaluation
processes; no games were restarted. The
[pause record](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/process_pause_1790078242.json)
and [resumption record](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/process_resume_1790091251.json)
identify the affected processes and suspension duration. Raw elapsed evaluation
times include suspension and must not be interpreted as active compute time.
The reporting supervisor's wall-clock timeout expired during suspension. Its
existing child audit finished after resumption, and a
[recorded reporting recovery](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/fresh_report_resume_1790091397/recovery.json)
verified that artifact and completed the one remaining pooled report. This changed
no experiment source, policy, seed, setting, or selection criterion and ran no new
games. Legacy-control evaluation and the queued early-policy tests retain their
original owners and settings.

A [second pause](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/process_pause_1790093337.json)
was also [resumed on the user's instruction](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/process_resume_1790096325.json).
All 15 existing processes retained their identities and memory. Before resumption,
the four saved outcome files, 231 source files, selection, and helper versions were
verified unchanged. Both ongoing control panels subsequently advanced with their
previous output prefixes intact, and all three owners refreshed their status.
This suspension lasted about 2,988 seconds. The same timing caveat applies to both
pauses. The waiting diagnostic and audit timers
had not started, so no additional timeout recovery was needed.

The commands below record the already running recovery's launch.

From the repository root, using `.venv/bin/python`:

```sh
python -m tools.long_horizon_study prepare-recovery \
  --root eval/long_horizon_generalization_20260921_logical_kl \
  --recover-from eval/long_horizon_generalization_20260921_micro160
python -m tools.long_horizon_study train-all --root eval/long_horizon_generalization_20260921_logical_kl
python -m tools.long_horizon_study evaluate-all --root eval/long_horizon_generalization_20260921_logical_kl
```

Run `train-all` and `evaluate-all` as separate supervised processes. Evaluation
waits for atomic endpoint checkpoints and checks the live trainer handle.
`training_status.json`, `evaluation_status.json`, and `compute.json` record
processes and progress; job stdout is in `jobs/`. Training TensorBoard files are
under each arm's `logs/tensorboard/` directory. Completed outcomes, forecast
traces, and summaries are stored under `evaluations/`.

Once development is complete, `select`, `confirm`, `early-transfer`, and
`finalize` freeze the endpoint and generate the confirmation evidence and final
report; pass the same explicit `--root` to each command. Incomplete panels cannot
produce a passing result. Source changes
require an explicit new branch. Existing outputs and interrupted jobs require
inspection before recovery.

The predecessor's
[finishing supervisor](../../eval/long_horizon_generalization_20260921_micro160/analysis/finish_study.py)
stopped on the failed training dependency before invoking any of those four
commands. Its manifest, status, and stage logs are in `finish_supervisor/`
under the active run. It does not restart failed or partial jobs.
A completed experiment can still fail the generalization gates; that result
requires further model work under the original improvement goal.

The recovery's original
[finishing supervisor](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/finish_study.py)
was intentionally replaced while it was waiting without an active child.
The [transition record](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/parallel_finish_transition.json)
preserves its state, the tested replacement hashes, the reversible pause,
verification that the replacement was live, and the original process's confirmed
exit. Training and development evaluation continued throughout the handoff.

The active [confirmation coordinator](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/finish_parallel_confirmation.py)
observed both development/training supervisors complete, passed the existing
process-isolation regression, and invoked the unchanged selection command.
It is scheduling eight whole confirmation arm/replica
jobs with at most three concurrent processes. Each job retains the full seed
order, two sampling repeats, greedy mode, batch 32, four workers, and four
inference threads. The sampler uses a panel-wide RNG, so panels are not split
across processes. The core `confirm` command subsequently validates and reuses
the completed outputs before core `early-transfer` and `finalize` run.

All [18 scheduler and ownership tests](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/parallel_confirmation_scheduler_tests.json)
passed. They cover bounded concurrency, unique output ownership, exact CLI job
plans, stopping dispatch and draining active jobs on failure, and requiring a
verified coordinator handoff. The real
[process-isolation regression](../../eval/long_horizon_generalization_20260921_logical_kl/finish_parallel_supervisor/concurrency_regression.log)
also passed before confirmation began. The coordinator preserves
failed outputs without automatic retries. Its live status is in
[`finish_parallel_supervisor/status.json`](../../eval/long_horizon_generalization_20260921_logical_kl/finish_parallel_supervisor/status.json).
This changes execution overlap; the model-selection rule, fresh panels, and
acceptance criteria remain the frozen protocol.
