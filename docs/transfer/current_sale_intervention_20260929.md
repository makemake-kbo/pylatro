# Does assistance on enabling sales improve current-policy games?

The saved sequence census finds that about 94–95% of teacher SHOP sales
immediately enable an offered Joker purchase. On development replacement
sequences, the base matches the sale in 24.48% of states but the next purchase
after the teacher's sale in 92.58%. Whole-shop teacher assistance previously
raised wins from 11/64 to 29/64 on the matched current-policy panel. Test
whether assistance specifically on teacher-proposed sales captures useful
gameplay headroom, including an early-only intervention. No training occurs.

Reuse the exact 64 exposed game seeds, immutable current base and v13 runtime
from the completed phase diagnostic. Reuse its admitted student and whole-shop
results and traces; do not rerun those games. Run one new sampled game per
seed for each of two arms:

- `sale`: query the teacher during SHOP. Execute its proposal only if that
  proposal is a Joker sale; otherwise execute the student sample.
- `early_sale`: apply the same rule only while current Ante is below 4,
  including Ante zero and any later return below 4. Do not query in other
  states. Hands, packs and all other phases remain student controlled.

Always draw the student action before any teacher query, including steps where
the teacher will replace it. Preserve CPU FP32 batch1, temperature1, mixture.5,
one CPU thread, Blue deck/White stake/Ante8, 2,000 total engine steps, UID reset1
and per-game Torch/Python/NumPy seeds from SeedSequence([game_seed,0,1729]).
The search/grow-scalers teacher receives a private state copy. Its queries
must leave live state, model inputs, UID counter and policy RNG unchanged.
A teacher-proposed sale already matching the student is controlled but is
not counted as a replacement. This intervention also permits direct beneficial
sales; it does not assert that every assisted sale is a replacement. It does
not suppress student sales when the teacher proposes another family.

Record every proposal, executed action, phase/Ante, input/state/RNG hashes and
terminal result. Run four fixed spawn shards within one owned measurement,
then four within the serial native audit. Reproduce every new trace and
intervention rule independently. Check each new arm's unchanged prefix against
the saved student trace, and both new arms against each other until their first
different action. Their first difference must be a sale intervention at Ante4
or later. Preserve failures and censoring; no replacement seeds or retries.
Only whole-owner actual exit zero admits the result.

Report wins and paired differences for sale−student, early_sale−student,
sale−early_sale and whole_shop−sale. Use the existing 10,000 paired seed
bootstrap draws with seed1729 and marginal exploratory 95% intervals. All 256
new/reused episodes must be uncensored for prioritization. The primary question
is sale−student; early_sale−student is the fixed secondary horizon question.
Do not use which arm wins to redefine the intervention or claim a trained gain.

A positive primary difference with a paired lower bound above zero supports
designing one exploratory replacement-learning experiment. A positive early
lower bound supports targeting early enabling decisions specifically. An
unresolved or nonpositive result does not identify these teacher sales as a
demonstrated win bottleneck; reconsider the next measurement before a fit.
Do not automatically extend the seed panel or change assistance eligibility.
All final learned win-rate, retention, early-attribution, independent-fit and
untouched-confirmation requirements remain unchanged.

Budget from the admitted whole-shop arm: its 64 games consumed 1,677.6184
summed worker seconds. Conservatively charge both new arms the same cost:
two times that sum divided by four workers gives 838.8092 seconds per phase.
Freeze each measurement/audit allowance at ceil(2×838.8092+120) = **1,798
seconds**, below a 2,400-second ceiling, with 30 seconds owner cleanup grace.
Expected combined time is about 28 minutes using this conservative forecast.
No timing pilot or automatic extension. Check a live owner roughly ten minutes
apart, keeping heartbeat output in artifacts. No other native job runs beside
this owned study.
