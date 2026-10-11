# Measure whether prospective supervision transfers to sales

After the single [prospective-purchase fit](replacement_auxiliary_experiment_20260929.md)
and its audit have both exited successfully, compare its fixed 1,000-update
endpoint with the immutable base and the saved ordinary 1,000-update control.
This measurement adds no fit and changes none of the frozen spending criteria.
The [numeric recovery correction](replacement_auxiliary_numeric_recovery_20260929.md)
preserves both incomplete attempts; only the first completed, fully audited
endpoint is eligible for this scan.

Forward every row of the same 202,141-row training and 26,045-row development
caches for all three policies. Preserve their original consecutive MPS FP32
batches of 32, including the final partial batch. Use the original supervised
collation and likelihood path, eval mode, temperature1 and hand mixture0.5.
Future targets remain outside the model inputs. Save every identity, teacher
action, flat mode and full-action NLL. At all eligible pre-sale rows also save
the ten raw purchase logits, future target, offer support, auxiliary NLL and
prospective mode. Check all base outputs against their archived predictions
and ordinary development outputs against the admitted saved-prefix scan.

The serial audit independently checks every saved identity and legal mode,
all archived controls, and float64 masked normalization of all prospective
distributions. It also re-forwards all 26,045 development rows and the first
320 original-layout training rows for each policy using direct cached
observation tensors instead of supervised collation. Both paths call the same
underlying policy distribution function; this is an independent inference
and input-construction check, not an independently implemented decoder.
Modes must match exactly;
likelihood differences must be at most 1e-5 and purchase-logit differences at
most 2e-5. Check unchanged model hashes and immutable inputs. The training
scan outside those first 320 rows receives a saved-record audit, not an
independent second native inference pass. All development spending decisions
receive the independent native check.

Report training and development results for all decisions, SHOP sales,
SHOP purchases, non-sale SHOP decisions, early/later SHOP sales, and early
discards. Report prospective NLL and exact choice on all eligible labels and
on the multiple-offer subset. Keep singleton supports in the former only;
they have zero NLL and no learning signal. Early means current Ante<4,
including Ante0 and re-entry, consistent with prior diagnostics.

The frozen development screen is the conjunction of all nine comparisons:

- Future-offer NLL on the 469 multiple-offer choices is lower than base and
  ordinary1k (two checks).
- Teacher-action NLL on the 716 SHOP sales is lower than each control, and
  exact sale matches exceed each control (four checks).
- Exact SHOP purchase matches are at least each control (two checks).
- Non-sale SHOP NLL is no worse than ordinary1k (one check).

Report paired source-game bootstrap intervals for development NLL and exact
agreement differences in these four groups. Use 10,000 resamples of the 128
complete development games, NumPy default_rng(1729), preserving each group's
decision-weighted denominator in every resample. These are descriptive
marginal 95% intervals, not extra gates or a multiplicity-adjusted success
claim. Report all failures. No threshold, endpoint, coefficient or panel
selection follows the results. Passing permits preparation of the separate
SHOP-only fresh-game test; it does not demonstrate long-horizon improvement.

The admitted probability scan took 400.263664 seconds for 159,296 forwarded
rows. At that rate, 684,558 measurement rows forecast 1,720.09 seconds;
budget ceil(2×forecast+120) = 3,561 seconds. The audit's 79,095 native rows forecast 198.74
seconds; budget ceil(2×forecast+300), with the extra allowance covering the
saved-record recount and bootstrap. The protocol computes the exact integer
budgets from the pinned historical result (audit: 698 seconds). Each owner has
30 seconds cleanup grace. Use the tested pause-aware owner and worker clocks
from the [pause recovery amendment](replacement_auxiliary_pause_recovery_20260929.md),
subtracting only explicit user-requested pauses. There is no extension, retry
or timing pilot. Measurement and audit
run serially after the fit owner has exited. Poll about ten minutes apart;
routine heartbeats stay in logs. Admit scientific results only after the
entire scan owner actually exits successfully.
