# Pretraining transfer: fitting, fresh imitation, and wins

Training the encoder improves teacher-action fitting and fresh-state prediction
under this recipe. It also improves greedy wins over the matched frozen encoder;
the sampled win difference remains unresolved. This is one diagnostic training
replica. The broader long-horizon generalization goal remains unproven.

The [frozen protocol](../../eval/pretraining_transfer_20260923/stage2_protocol_frozen_v1.md)
and [complete result](../../eval/pretraining_transfer_20260923/analysis/stage2_results_v1/result.json)
cover both fixed endpoints and both gameplay modes. No endpoint or mode was
selected from these outcomes. World-model work remains deferred.
A compact [machine-readable summary](data/pretraining_transfer_summary_20260923.json)
retains the comparisons, uncertainty, family metrics and source/audit hashes.

## Fixed comparison

Both arms started from the same corrected supervised seed-0 checkpoint, using
v13 observations and the corrected runtime. Each trained on all 202,141 decisions
from 1,000 historical teacher games for 4,000 updates of 320 examples: 1,280,000
presentations, about 6.33 passes. Both trained all actor heads; one additionally
trained the encoder and embeddings. Critics stayed fixed. The objective was
unit-weight full-action behavior cloning with identical row order, fresh AdamW,
learning rate 3e-5, a 100-update warmup, and disabled dropout.

Fresh teacher-state evaluation covered 26,045 decisions from 128 reserved games.
Autonomous evaluation covered the same 128 seeds per policy and mode, totaling
768 games. The task was Blue Deck, White Stake, Ante 8. Sampling used the fixed
global Torch stream seeded with 0, temperature 1 and batch size 32. All panels
finished without stalled or censored games.

## Fitting and fresh-state imitation

| Policy | Training agreement | Training NLL | Fresh agreement | Fresh NLL |
| --- | ---: | ---: | ---: | ---: |
| Initial | 82.16% | 0.672312 | 81.82% | 0.685193 |
| Frozen encoder | 82.45% | 0.647823 | 82.02% | 0.670901 |
| Trainable encoder | 85.83% | 0.496954 | 82.72% | 0.639106 |

The primary trainable-minus-frozen comparison improves training agreement by
3.3838 percentage points and fresh agreement by 0.7026 points. Fresh
decision-weighted NLL falls by 0.031796, with a paired whole-game 95% bootstrap
interval of [-0.038712, -0.024599]. The equal-game NLL difference is -0.032371
[-0.039348, -0.024980].

Much of the training advantage does not carry through to these fresh states.
The trainable arm's training-to-fresh agreement gap is 3.11 points, versus 0.43
for the frozen arm. This pattern is consistent with overfitting or distribution
shift; it does not identify their relative contributions. Historical runtime
corrections can alter teacher trajectories, and the bounded replay check does
not establish parity across the whole historical corpus.

| Teacher decision | Frozen training | Trainable training | Frozen fresh | Trainable fresh |
| --- | ---: | ---: | ---: | ---: |
| Play | 67.94% | 75.14% | 65.50% | 67.99% |
| Discard | 48.85% | 57.94% | 49.72% | 52.76% |
| Joker sale | 20.67% | 37.69% | 18.76% | 25.21% |
| Pack claim | 83.12% | 85.14% | 83.15% | 83.71% |

Fresh Joker-sale NLL is 2.228695 trainable versus 2.221834 frozen: the higher
sale agreement does not produce lower sale NLL. Errors remain substantial,
including on training states. The improving training curve does not establish
convergence or an irreducible error floor. Alternative actions may also be
harmless; exact disagreement alone does not measure gameplay regret.

## Autonomous wins

| Policy | Greedy wins | Sampled wins |
| --- | ---: | ---: |
| Initial | 41/128 (32.03%) | 20/128 (15.63%) |
| Frozen encoder | 37/128 (28.91%) | 27/128 (21.09%) |
| Trainable encoder | 51/128 (39.84%) | 33/128 (25.78%) |

| Paired difference, percentage points | Greedy, 95% interval | Sampled, 95% interval |
| --- | ---: | ---: |
| **Trainable minus frozen: primary** | **+10.94 [2.34, 19.53]** | **+4.69 [-3.91, 13.28]** |
| Frozen minus initial: secondary | -3.13 [-11.72, 5.47] | +5.47 [-2.34, 13.28] |
| Trainable minus initial: secondary | +7.81 [0.00, 16.41] | +10.16 [3.13, 17.19] |

All intervals use 10,000 paired whole-game seed resamples with
`numpy.default_rng(1729)`. They are diagnostic intervals. The primary greedy
comparison excludes zero; the primary sampled comparison does not. The positive
secondary sampled comparison includes continued training as well as encoder
adaptation. Outcomes are paired by game seed, without per-seed common random
numbers in the global sampling stream. Both modes remain part of the result.

The matched teacher won 95/128 games (74.22%). It can access the full simulator,
scoring and copied RNG, so this is a descriptive reference rather than an
information-matched upper bound for the student.

## What this explains and what remains

The earlier corrected-head work identified an action-representation restriction.
This experiment establishes that allowing the encoder to adapt closes another
part of the later corrected-head fitting gap and improves fresh-state NLL, with
a positive greedy win comparison against the frozen control. It does not
reconstruct the original 5,000-game pretraining: that training also updated its
encoder. The original corpus's verified whole-game win rate was 74.56%; the
logged roughly 79.3% measure weighted decisions rather than games.

Independent training replication, fresh confirmation, and a controlled test of
the contribution from early decisions remain necessary. Longer context and
world-model changes are not justified by these results. The subsequent
[confirmation attempt](pretraining_confirmation_20260923.md) preserved that
contrast and planned a fixed-continuation early-policy test, but ended incomplete
after host sleep consumed its wall-clock allowance. It adds a repeated training
benefit at update 1,000 and no fresh gameplay evidence. Remaining teacher-state
errors and train-to-fresh gaps stay explicit.

## Verification

The actual owner session 83749 exited 0 before fresh results were opened. Its
[terminal receipt](../../eval/pretraining_transfer_20260923/analysis/stage2_owner_terminal_83749_1790197312378024000.json)
is pinned with SHA256
`2bba4e20a5c9fa810284afd621f05d65bb94149966ef372fc1f548dfcb98839d`.
The prepared analyzer exited 0 and checked all five phases, fixed endpoints,
artifact pins, eight training snapshots, fresh imitation, all six gameplay panels,
and paired statistics. The complete result has SHA256
`955bf6d2514eaf2d179cae53146fde9a93298575ac406f52c605813d8e293778`.

The [independent autonomous audit](../../eval/pretraining_transfer_20260923/analysis/stage2_independent_autonomous_audit_1790197588603954000.json)
passed 54 checks. It reproduced all six win counts and every paired interval,
verified all 110,572 action-trace records against per-seed decision counts, and
checked actual panel pins and settings. The independent fresh-imitation audit
also [passed](../../eval/pretraining_transfer_20260923/analysis/independent_stage2_fresh_imitation_audit_v1.json):
all three policies have identical ordered identities for 26,045 decisions across
128 games, and independently recomputed arithmetic and whole-game bootstrap
intervals match. Saved parameter movement reports do not
independently reconstruct tensor or optimizer operations; analysis-time hashes
do not create retrospective execution provenance for previously unpinned metrics.
