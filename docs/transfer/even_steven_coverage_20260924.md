# Coverage and fitting of the Even Steven failure

The subsequent [trajectory audit](fitting_trajectory_20260924.md) measures every
saved training snapshot and all four intermediate checkpoints on the exposed
held-out panel. Training fit continues improving late, while held-out gains
shrink and the diagnosed fatal pair choice remains wrong. It qualifies the
next-step interpretation below without changing these coverage counts.

The original corpus contains the ingredients of the diagnosed scoring interaction,
but very few examples of applying it with one hand remaining. Across all **5,000
original games and 1,126,644 decisions**, there are **236,278 plays**, of which
**4,491 (1.90%)** occur with one hand left. Among **834** ordinary visible-hand
states with Even Steven and both even and odd pairs available, **19** have one
hand left. Only **one** of those nineteen has a single-pair teacher action.

There is also incomplete fitting in the later corrected-policy training run.
Of three training examples where the teacher chooses the pair of 4s with a pair
of 7s available, the trainable model still chooses the 7s in two. Replaying those
choices confirms lower immediate scores. However, neither substitution causes
an immediate loss in its training state, and one still clears the blind.

These findings establish limited coverage of the diagnosed decision context
and remaining imitation errors. They do not show how much either causes the
full teacher–student win gap. Generalizing learned card/Joker interactions to
unseen setups remains the desired behavior; absence of an exact setup alone
does not explain why that generalization fails. World-model work remains deferred.

The [complete summary](../../eval/even_steven_coverage_20260924/summary_v1.json)
and [compact report data](data/even_steven_coverage_20260924.json) retain all
strata, policies, per-game counts, likelihood metrics and training examples.

## Scope and definitions

Three distinct datasets are kept separate:

- The original pretraining corpus: 5,000 games, 1,126,644 decisions, 3,728 wins.
  This scan measures teacher-state coverage, without new original-model inference.
- The later Stage 2 training corpus: 1,000 games, 202,141 decisions. The saved
  starting, frozen-encoder and trainable-encoder predictions are available.
  Every training row was presented six or seven times during the matched run.
- The already-exposed Stage 2 held-out panel: 128 games, 26,045 decisions, with
  saved predictions for the same three policies. It is not fresh confirmation.

The predicates were fixed before the counts. “Ordinary visible hand” means that
every hand token has a known rank and suit, the vocabulary's base-card ID, and
no encoded enhancement, edition, seal, debuff, hidden face, permanent bonus or
forced selection. This is a token-level predicate, not a reconstruction of
every hidden state value. Both even and odd pairs being available means at least
two cards of an even rank 2/4/6/8/10 and at least two of an odd rank 3/5/7/9.

A “single-pair pattern” means that exactly one selected rank appears twice and
all other selected ranks are distinct. It is a structural description; some
Jokers or blind rules can change the actual poker-hand classification. Pair-rank
agreement distinguishes choosing the same pair with different filler cards from
choosing a different pair. It is not a replacement for measuring gameplay costs.

## Coverage across both training corpora

Each cell is **play decisions / distinct games**. The three-Joker setup is
exactly Banner, Space Joker and Even Steven, all active, with an ordinary visible
hand. Its absence below does not claim that the inventory never occurs in any
other phase or with modified cards.

| Predicate | Original 5,000 games | Later 1,000 training games | Exposed 128 held-out games |
| --- | ---: | ---: | ---: |
| All recorded plays | 236,278 / 5,000 | 47,149 / 1,000 | 6,113 / 128 |
| Active Even Steven | 8,668 / 672 | 1,915 / 147 | 227 / 18 |
| Active Even Steven, ordinary visible hand | 5,067 / 624 | 1,143 / 142 | 133 / 16 |
| Above, with both even and odd pairs | 834 / 399 | 176 / 87 | 20 / 12 |
| Above, specifically pairs of 4s and 7s | 68 / 60 | 9 / 9 | 1 / 1 |
| Exact three-Joker setup | 0 / 0 | 0 / 0 | 4 / 1 |
| Exact setup and both even and odd pairs | 0 / 0 | 0 / 0 | 1 / 1 |

The later training run supplies 12,159 row presentations of the 1,915 Even
Steven plays, including 1,124 presentations of the 176 mixed-pair states and
57 presentations of the nine 4s-and-7s states. Repeated presentations do not
increase the number of distinct states or games.

The original corpus includes eighteen single-pair-of-4s teacher choices within
the 68 states containing both 4s and 7s; none occurs with one hand remaining.
The later corpus includes three such choices, also all with multiple hands left.
Other hands and configurations can still teach aspects of the same scoring rule;
these counts do not make exact matching examples a prerequisite for learning.

The last-hand subdivision was declared after replaying the three later training
examples, so it is a descriptive follow-up rather than an independent hypothesis
test. The same predicates were then applied to the entire original corpus.

| Ordinary Even Steven states with both even and odd pairs | Original corpus | Later training | Exposed held-out |
| --- | ---: | ---: | ---: |
| All states | 834 | 176 | 20 |
| One hand remaining | 19 | 1 | 1 |
| Single-pair teacher pattern with one hand remaining | 1 | 0 | 1 |
| Specifically pair of 4s with 7s available and one hand remaining | 0 | 0 | 1 |

The original last-hand single-pair example is seed 2005251, step 85, with a pair
of 10s selected. The held-out single-pair example is the previously diagnosed
seed 850000015, step 28: the teacher's pair of 4s scores 1,989 while the model's
pair of 7s scores 795 against an 800-point requirement.

## Fitting of the later corpus

Each cell lists exact teacher-action matches for **starting / frozen / trainable**
policies, followed by the shared number of states. Empty training strata are
explicitly empty, not zero-accuracy measurements.

| Predicate | Later training matches | Exposed held-out matches |
| --- | --- | --- |
| All recorded plays | 31,700 / 32,031 / 35,430 of 47,149 | 3,980 / 4,004 / 4,156 of 6,113 |
| Active Even Steven | 1,128 / 1,132 / 1,230 of 1,915 | 125 / 128 / 126 of 227 |
| Active Even Steven, ordinary visible hand | 705 / 708 / 759 of 1,143 | 76 / 77 / 76 of 133 |
| Above, with both even and odd pairs | 77 / 80 / 83 of 176 | 12 / 13 / 14 of 20 |
| Above, specifically pairs of 4s and 7s | 1 / 2 / 3 of 9 | 0 / 0 / 0 of 1 |
| Exact three-Joker setup | No states | 2 / 2 / 1 of 4 |
| Exact setup and both even and odd pairs | No states | 0 / 0 / 0 of 1 |

On the broad Even Steven stratum, trainable versus frozen agreement is
**64.23% versus 59.11% in training**, but **55.51% versus 56.39% held out**.
Teacher NLL improves in both: 1.7215 to 1.3690 in training and 1.7918 to 1.7100
held out. This is a descriptive comparison across 147 training games and only
18 held-out games, not evidence of a statistically resolved regression.

In the 176 ordinary mixed-pair training states, 47 teacher actions have a
single-pair pattern. The starting, frozen and trainable policies choose that
same pair rank in 16, 15 and 13 cases respectively. The trainable policy picks
an odd pair in thirteen cases where the teacher picks an even pair. These
structural disagreements alone do not establish inferior scores or lost games.
All likelihood, exact-action and pair-pattern metrics are retained, including
the positive held-out mixed-pair agreement result and the negative pair-rank counts.

## Actual consequences on three training examples

All three later training examples with the teacher choosing a pair of 4s were
replayed, including the one learned correctly. All **607 decisions across the
three complete games** reproduce under the archived Stage 2 runtime: every
cached input array, action and terminal outcome matches. Repeating each recorded
action from a copied state reproduces its full factual post-state. Saved policy
choices are evaluated from the same state and RNG/card-identity counters.

| Training seed / step | Presentations | Hands left | Points needed | Teacher score | Starting / frozen score | Trainable score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000542 / 90 | 7 | 3 | 5,226 | 4,440 | 3,388 | 3,388 |
| 4000602 / 40 | 6 | 4 | 680 | 3,528 | 1,248 | 1,248 |
| 4000708 / 70 | 7 | 4 | 1,650 | 11,826 | 936 | 11,826 |

The first two rows establish lower-scoring training-state choices after repeated
supervision. The first clears in neither branch; the second clears in both.
Neither is an immediate loss. On the third row, encoder training changes the
choice to the teacher action and immediately clears the blind, whereas the
starting and frozen policies do not clear on that action. This is a local
training-state improvement, not a new autonomous win result.

## Verification and implications

The corrected Stage 2 scan validates all saved prediction identities, legality,
likelihood arithmetic, aggregate metrics, exposures and immutable input hashes.
An independent implementation checks the raw-token predicates, complete Even
Steven row membership, per-game counts and pair-pattern arithmetic: **11,282
checks pass**. The original scan verifies all 20 Parquet shards, every one of
the 1,126,644 seed/step identities, and the exact original-generation hashes for
tokenizer, constants, subset mapping, vocabulary and game data.

A separate original-corpus audit independently rereads all row/game totals and
the last-hand denominators, and recounts all 8,668 saved Even Steven rows. Its
full original token classification remains worker evidence, with matching source
semantics and a known positive control. The three-game cost audit additionally
checks saved choice identities and score products; it is not a second independent
engine replay. All worker processes actually exited zero and are terminal.

The first coverage scan used token ID 0 for ordinary cards; the actual base-card
ID is 1. A known-case control exposed this error. That result and its validation
failure remain preserved; its narrower zero counts are invalid. The corrected
version derives the ID from the pinned vocabulary and passes both the known-case
control and the independent raw-token check. Only corrected counts appear here.

The evidence supports testing two existing gaps separately: fitting the scoring
interactions already represented in the data, and transferring those interactions
to resource-constrained combinations. Corpus win rate does not quantify either
ability. These results do not isolate data scarcity from optimization or
representation, prove convergence, or establish that any particular weighting
or additional-data intervention will improve wins. The broader replicated fresh
win and consequential early-decision requirements remain unmet. The failed
Stage 3 attempt and its unopened fresh-outcome gate were not accessed.
