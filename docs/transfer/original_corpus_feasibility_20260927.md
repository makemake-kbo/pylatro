# Original-corpus imitation and runtime-domain checks

**The complete corrected-base census is verified, and it exposes a material difference between the original and current data-generating rules.** The original corpus contains 97,809 post-claim empty-pack cleanup actions and 16,146 claim transitions that add the claimed consumable to inventory. Both signatures are absent from the later training and exposed evaluation corpora. A larger-data intervention must account for these differences. No new model was trained and no new gameplay improvement was measured.

The [completed combined-objective study](ranking_preservation_development_20260927.md) failed its development readiness screen. Recent fitting experiments used a later 1,000-game corpus. This study measured the same corrected Stage 2 base on every original record before selecting another fit. Its weights inherit original pretraining; this is not an untouched generalization population or a convergence result.

| Population | Decisions | Exact-action agreement | Teacher NLL |
|---|---:|---:|---:|
| Original 5,000 games | 1,126,644 | 84.32% | 0.571218 |
| Later 1,000 training games | 202,141 | 85.83% | 0.496954 |
| Exposed 128 evaluation games | 26,045 | 82.72% | 0.639106 |

With all three populations reweighted to the evaluation set's action-family proportions, agreement is **83.27% / 85.77% / 82.72%**, and NLL is **0.610642 / 0.498008 / 0.639106**. Seventeen original and six later consumable-sale records have no corresponding evaluation family and are excluded only from this reweighting; all raw results remain available. Reweighting adjusts family proportions, not within-family difficulty or exposure.

The original corpus's weak action families resemble the exposed evaluation population more closely than the recent training population. This is a fitting/transfer distinction, not proof of forgetting or an architectural limit: the same corrected model was not measured here before its earlier continuation.

The corpus is also from an older runtime. Generation-pinned `shop.py` allowed a pack consumable to enter inventory when its preferred immediate use was unavailable. The current source requires a legal immediate use and deals a temporary hand for Tarot/Spectral packs. The later training generation uses the current `shop.py` hash. The historical runner could leave its phase at BOOSTER_PACK after a claim had cleared `state.pack`; the current runner and controller resolve that transition to SHOP. Tokenizer and action mappings are unchanged, so format compatibility did not reveal these transition-rule differences.

A separate 48.26-second record scan, without engine or model imports, measured:

| Recorded signature | Original 5,000 games | Later 1,000 games | Exposed 128 games |
|---|---:|---:|---:|
| Empty-pack skip immediately after a claim | 97,809 | 0 | 0 |
| Claim followed by inventory increasing by the claimed consumable ID | 16,146 | 0 | 0 |

The empty-pack signature is recorded phase 3, no visible pack items, and a skip action; every such original row immediately follows a claim. These rows are **8.68% of the original corpus**, with 99.9949% model agreement. Excluding them for analysis changes original agreement from 84.32% to **82.83%**, and NLL from 0.571218 to **0.625476**. The exposed evaluation values are 82.72% and 0.639106. The inventory signature compares consecutive observations, including the complete multiset of consumable IDs; it is not a per-instance provenance trace. The pinned source establishes that the old banking branch exists. These counts do not quantify its win-rate impact.

This changes the next research decision: the original corpus cannot be treated as a clean increase in current-rule training data. Dropping cleanup records would remove metric inflation but would not undo trajectories produced under the old banking and pack-hand rules. A data-scale test should use demonstrations generated under the evaluation rules, with the same teacher/settings as its control. The existing transfer problem on current-rule states remains; neither these artifacts nor the teacher's corpus win fraction explains the whole autonomous gap.

The read-only census verified all 20 Parquet shard hashes and all 1,126,644 records across 5,000 games. Per-game row counts, consecutive step identities, recorded outcomes, maximum Ante and fixed Blue Deck/White Stake/Ante 8 settings agree with metadata. Every teacher action is in range and legal under its recorded mask; all masks contain only zero or one. The corpus contains 3,728 wins (74.56%).

All 14 recorded observation fields have the same native shapes and dtypes as the corrected model's later training cache. Tokenizer version is 13. Five relevant source files match byte-for-byte between the original generation inventory and the frozen current source: game data, vocabulary construction, constants, observation schema and tokenizer.

A second full scan reconstructed the vocabulary, compared every recorded constant, and checked all 17,327 action mappings and round trips against frozen code. It decoded every observation array in all 1,126,644 records: all values are finite, all masks are binary, and the declared categorical ranges pass. The ordered identity digest matches the first census. This scan completed in 135.91 seconds, with an observed process exit of zero. Continuous integer features have no arbitrary imposed bounds. The existing hand-level embedding saturates at level 15; 7,846 token entries exceed that cap. That is recorded separately from malformed categories and does not establish a cause of the transfer failure.

The 5,000 seed identities are not contiguous: their range is 2000000–2080230. They are disjoint from the later 1,000 training seeds and the currently exposed teacher-state, autonomous and early-decision development panels. The exact recorded seed list must be retained; its minimum and maximum are not a substitute for that list. This does not reserve any untouched confirmation seeds.

Using the existing native cache layout would require 59,007,979,500 data bytes, about 55 GiB before small array headers and metadata. Available disk space exceeded twice that estimate at admission. The new measurement reads one row group at a time and allocates no such cache. It preserves the actual shard/row-group order and the recorded identity digest.

The first identity/action census completed in 60.60 seconds under a fixed 300-second budget, without importing a model. An initial inspection attempt stopped because a PyArrow dtype-conversion helper required unavailable pandas. That attempt and its source are retained. Replacing that conversion with an explicit Arrow dtype mapping allowed the census to complete; no dependency was installed and no experiment was retrained.

The new direct observation-to-tensor path reproduces 52 archived decisions across 13 action families: every global action mode matches, and the maximum teacher-NLL difference is 4.77e-7. The saved reward version 17 and full fingerprint also match the loaded checkpoint. The census reads no rewards or return targets and computes no critic loss.

The existing reference populations were recounted from all 202,141 later-training predictions and 26,045 exposed held-out predictions, checking every identity against the pinned caches and every aggregate against the previously audited metrics. This used standard-library arithmetic and no model inference. For this same corrected base, raw agreement is 85.83% on later training and 82.72% held out. Giving both populations the held-out action-family proportions changes that gap only from 3.11 to **3.06 percentage points**. The difference in family mix explains little of the observed gap; differences in state difficulty and exposure within families remain possible.

| Teacher action family | Original-corpus agreement | Later-training agreement | Exposed held-out agreement |
|---|---:|---:|---:|
| Play subset | 69.45% | 75.14% | 67.99% |
| Discard subset | 50.85% | 57.94% | 52.76% |
| Sell Joker | 24.62% | 37.69% | 25.21% |

These three families account for 77.29% of the agreement gap and 85.96% of the NLL gap after reweighting. Early decisions (Ante below 4) show a raw agreement gap of 3.19 points, and later decisions 3.07 points. These are descriptive population differences, not paired state effects, proof of a unique bottleneck, or gameplay losses. The reweighting retains all held-out decisions; the only excluded later-training family is six consumable sales absent from that held-out population. The same fixed-family comparison rule is recorded before opening the original-corpus result.

The frozen measurement covered every original record with the corrected Stage 2 base (`a43e73ab…`), fp32/MPS, batch 32, evaluation mode, temperature 1, and no optimizer. It completed in 3,214.4 seconds under the 7,200-second limit, with an observed exit of zero. It saved teacher likelihood and exact global-action agreement by family, game, early/late Ante and teacher outcome. A separate checker reread all original identities, scalars and action masks, checked every predicted and teacher action, and reconstructed all 5,019 summary groups. Verification completed in 51.57 seconds and also exited zero. The comparison export exited zero after both checks. All processes are terminal.

For original early decisions (Ante below 4), raw agreement is 83.69% over 314,970 records, with NLL 0.611296; later decisions are 84.56% over 811,674 records, with NLL 0.555666. These raw strata include the historical domain differences described above. Low aggregate imitation alone does not justify a broader-data fit. A new recipe has not been selected. The independent confirmation and long-horizon win-rate criteria remain unchanged, and world-model work remains deferred.

Evidence: [complete admission data](data/original_corpus_feasibility_20260927.json), [audit source](../../eval/original_corpus_feasibility_20260927/audit_v1.py), [actual successful exit](../../eval/original_corpus_feasibility_20260927/audit_terminal_v1.json), [retained initial attempt](../../eval/original_corpus_feasibility_20260927/audit_attempt1_terminal_v1.json), [corpus provenance and research scope](pretraining_gap_focus_20260923.md).

New evidence: [complete value/mapping scan](../../eval/original_corpus_feasibility_20260927/values_v1.json), [observed successful exit](../../eval/original_corpus_feasibility_20260927/values_terminal_v1.json), [frozen measurement protocol](../../eval/original_corpus_feasibility_20260927/measurement_protocol_v1.json), [model compatibility](../../eval/original_corpus_feasibility_20260927/measurement_v1/compatibility.json), [independent checker](../../eval/original_corpus_feasibility_20260927/verify_measurement_v1.py).

Reference comparison: [complete recount](../../eval/original_corpus_feasibility_20260927/reference_metrics_v1.json), [actual successful exit](../../eval/original_corpus_feasibility_20260927/reference_metrics_terminal_v1.json), [fixed-family arithmetic](../../eval/original_corpus_feasibility_20260927/reference_comparison_v1.json). The [teacher-information diagnosis](teacher_cache_gap_20260924.md) remains a separate limitation: draw order is partly visible, a historical cache bug has a measured narrow label effect, and the broader information-sufficiency panel was not completed. Passing format checks does not resolve that question.

Completed results: [three-corpus comparison](data/original_corpus_imitation_20260927.json), [full independent verification](../../eval/original_corpus_feasibility_20260927/verification_v1.json), [measurement exit](../../eval/original_corpus_feasibility_20260927/measurement_terminal_v1.json), [verification exit](../../eval/original_corpus_feasibility_20260927/verification_terminal_v1.json), [comparison exit](../../eval/original_corpus_feasibility_20260927/comparison_terminal_v1.json), [pack-domain record scan and source pins](../../eval/original_corpus_feasibility_20260927/pack_domain_v1.json), [pack-domain scan exit](../../eval/original_corpus_feasibility_20260927/pack_domain_terminal_v1.json).
