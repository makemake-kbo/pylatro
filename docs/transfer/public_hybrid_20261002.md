# Public-information hybrid gameplay study

**KEEP: Fixed paired greedy confirmation criteria.**

Greedy confirmation: **hand assistance won 253/400 (63.25%), versus 200/400 (50.00%) for the neural baseline**. The paired gain was **+13.25 percentage points**, with a 95% game-bootstrap interval of **[+8.00, +18.50]**. The incumbent is `ppo_replica1`. The primary criterion passed; the sampled panel is secondary.

Blue Deck, White Stake, Ante 8; CPU FP32, four workers, one Torch thread per worker; temperature 1 and independent per-game policy RNG. Neural weights were frozen throughout. All arms used public-information-v2 observations, legal masks, sampled derived features, and the same public-sample Joker ordering. Historical win rates are reference results under older semantics.

An earlier runtime attempt was invalidated because the simulator's visual-only Amber Acorn approximation exposed the concealed Joker order. Its 16 pilot games, 384 baseline games, and 234 partial architecture games are retained separately and excluded from every result below. Its architecture outcomes were not opened. Public-information-v2 canonicalizes the known owned roster, samples concealed order for simulation, and disables ordering and indexed Joker actions while that order is unknown. All five replacement seed panels are fresh and disjoint from the invalidated attempt, with identical panel sizes and search settings. The original 48-hour clock remains in force.

A second attempt stopped during confirmation when the legacy continuation heuristic proposed card slot 16, outside the existing 16-card action vocabulary. That attempt completed 16 pilot, 384 baseline, 1,024 architecture, and 218 confirmation games; its one failed game was recorded, never counted as a loss. Confirmation win outcomes were not opened. The invalidated architecture panel had 123 neural-only, 145 hand-only, 116 shop-only, and 149 combined wins out of 256; these are unconfirmed historical results, excluded from the replacement analysis below. The correction restricts heuristic proposals to addressable card slots while retaining the full hand for engine transitions and scoring, and bounds concealed Aura masks to the same vocabulary. All five panels were reserved again under a fresh V3 seed namespace, excluding both earlier reservations. Observation semantics remain public-information-v2. Search settings and selection rules did not change. A new technical-phase allowance was recorded explicitly; the original overall 48-hour and architecture deadlines were retained. All prior evidence is preserved under `runs/public_hybrid_20261002/invalidated_attempt_2/`.

The technical pilot completed 16 games and 3,385 decisions, with 347 private-state perturbation checks, 0 illegal actions, and 15 searches falling back without a completed paired comparison. It projected 25.31 hours for the full study using the combined arm's measured cost for every game. Pilot wins did not select search settings.

## Baseline selection: 128 common seeds

| Arm | Wins | Rate | Changed decisions | Searches | Fallbacks | Mean / p95 / max decision seconds | Worker hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_replica0/off | 58/128 | 45.31% | 0 | 0 | 0 | 0.059 / 0.077 / 0.184 | 0.385 |
| ppo_replica1/off | 66/128 | 51.56% | 0 | 0 | 0 | 0.058 / 0.076 / 0.186 | 0.377 |
| supervised/off | 41/128 | 32.03% | 0 | 0 | 0 | 0.059 / 0.077 / 0.117 | 0.344 |

Selected: **ppo_replica1**.

## Architecture comparison: 256 separate seeds

| Arm | Wins | Rate | Changed decisions | Searches | Fallbacks | Mean / p95 / max decision seconds | Worker hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_replica1/off | 127/256 | 49.61% | 0 | 0 | 0 | 0.055 / 0.069 / 0.131 | 0.717 |
| ppo_replica1/hand | 171/256 | 66.80% | 7,845 | 16,488 | 10 | 0.227 / 0.945 / 3.118 | 3.449 |
| ppo_replica1/shop | 146/256 | 57.03% | 2,428 | 18,950 | 14 | 0.566 / 2.149 / 3.089 | 7.188 |
| ppo_replica1/both | 155/256 | 60.55% | 10,681 | 37,761 | 46 | 0.752 / 2.327 / 3.095 | 11.038 |

Selected: **hand**.

| Paired contrast | Gain (percentage points) | 95% game-bootstrap interval | Gained / lost games |
| --- | ---: | --- | ---: |
| both | +10.94 | [+3.91, +17.97] | 57 / 29 |
| hand | +17.19 | [+11.33, +23.05] | 57 / 13 |
| shop | +7.42 | [+1.17, +13.67] | 45 / 26 |

## Greedy confirmation: 400 untouched seeds

| Arm | Wins | Rate | Changed decisions | Searches | Fallbacks | Mean / p95 / max decision seconds | Worker hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_replica1/off | 200/400 | 50.00% | 0 | 0 | 0 | 0.058 / 0.074 / 0.111 | 1.221 |
| ppo_replica1/hand | 253/400 | 63.25% | 12,028 | 25,228 | 11 | 0.236 / 0.985 / 3.176 | 5.505 |

| Paired contrast | Gain (percentage points) | 95% game-bootstrap interval | Gained / lost games |
| --- | ---: | --- | ---: |
| assisted minus baseline | +13.25 | [+8.00, +18.50] | 85 / 32 |

## Secondary sampled comparison: 128 additional seeds

| Arm | Wins | Rate | Changed decisions | Searches | Fallbacks | Mean / p95 / max decision seconds | Worker hours |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo_replica1/off | 52/128 | 40.62% | 0 | 0 | 0 | 0.059 / 0.072 / 0.139 | 0.387 |
| ppo_replica1/hand | 67/128 | 52.34% | 4,110 | 8,247 | 0 | 0.248 / 1.041 / 3.190 | 1.832 |

| Paired contrast | Gain (percentage points) | 95% game-bootstrap interval | Gained / lost games |
| --- | ---: | --- | ---: |
| assisted minus baseline | +11.72 | [+1.56, +22.66] | 32 / 17 |

## Accounting and interpretation

| Panel | Completed games | Recorded errors |
| --- | ---: | ---: |
| pilot | 16 | 0 |
| baseline | 384 | 0 |
| development | 1024 | 0 |
| confirmation | 800 | 0 |
| sampled | 256 | 0 |

Final verification accounted for all **2,480 games** and independently recomputed every published summary and paired statistic. All 169 frozen scientific source files and all three checkpoint hashes matched. The 928 reserved seeds are unique and disjoint from 1,856 seeds reserved for invalidated attempts, as well as the retained historical exclusions. The original clock elapsed **17.95 hours**, including invalidated attempts and interruptions, within its 48-hour cap. Final-study game execution used 33.10 aggregate worker hours; this excludes invalidated attempts. The broad preflight passed 149 tests using the compiled engine path; seven reporting/resume regression tests subsequently passed. Outcome hashes and these checks are saved in `outcome_hashes.json` and `final_verification.json`.

The maximum measured planning interval was 3.028 seconds, including timer and cleanup overhead around the three-second planning budget. Decision latency additionally includes observation construction and neural inference, as shown above.

The fixed architecture advances an assisted arm only at a greedy gain of at least five percentage points. Confirmation additionally requires a paired 95% game-bootstrap interval strictly above zero, no information-boundary violations, and no illegal or silently dropped games. Bootstrap intervals use 10,000 resamples of complete paired games with generator seed 1729. Baseline ties prefer PPO replica 0, then replica 1, then supervised; assisted ties prefer fewer assisted phases, then measured latency.

Search uses at most six hand candidates or four shop plans and four shared sampled worlds. Only equally completed sample sets are compared; deadline handling can retain fewer than four complete sets. A fallback means no completed comparison, in which case the neural proposal is retained. Changed decisions count overrides of the model proposal at the same public decision state. Continuation caps are 60 hand actions and 100 shop actions. The three-second limit covers planning; reported decision latency also includes neural inference and public observation construction. Cost is aggregate worker wall time, not metered CPU time or financial expenditure.

Teacher agreement and forecast error were not advancement gates. The study measures actual terminal wins; it does not claim that unchanged neural weights learned new strategy. No outcome-selected coefficient sweep or additional auxiliary-loss experiment follows from these results.

See [player instructions and historical findings](README.md), [incumbents](../../incumbents/registry.json), [demonstrations](../../datasets/registry.json), and the [cleanup receipt](../../archives/experiments_20261002/receipt.json). Seed reservations, frozen source/configuration hashes, the frozen runtime archive, per-game outcomes and latency arrays, and verification logs are in `runs/public_hybrid_20261002`.

All 400 confirmation pairs finished before a NumPy boolean caused JSON summary serialization to stop. A separately hashed reporting adapter converts NumPy scalars to native JSON values. The study resumed saved games without replaying them; all frozen scientific source files, gameplay settings, and seed lists remained unchanged. The interruption and adapter hash are preserved in `reporting_interruption.json` and `reporting_adapter.json`.

Frozen source identifier: `90e31e8fd66b727447f3ebfdf19f8337d4a24398103ea06f68c578a1585f8daf`.
