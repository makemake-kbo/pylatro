# Test prospective purchase supervision before changing the architecture

Teacher help on SHOP sales improves the current policy by +20.31 points on
the exposed matched panel; early-only help remains unresolved. Most teacher
sales enable the next Joker purchase. The current policy's purchase logits
are computed before the sale, but current-action legality excludes that
future purchase from the behavior-cloning target. Test whether explicitly
supervising this prospective choice helps the shared representation learn
when and what to sell. A competing explanation is that knowing the desired
offer is insufficient to learn the old/new Joker relation.

Use the existing transformer, heads, observations and 202,141-row training
corpus unchanged. On the 5,183 recorded sale states whose next consecutive
teacher action buys the same offered Joker, newly legal after the sale, add
cross-entropy for that next offer using the **existing shop-buy logits at the
pre-sale state**. Normalize this auxiliary distribution over all present Joker
offers, including currently unbuyable ones. Do not alter the policy's actual
legal action distribution. There are 3,494 nontrivial training choices and
469 development choices; singleton offer sets have zero auxiliary loss and
gradient. Future actions are labels only, never model inputs.

One controlled change: add this auxiliary loss to ordinary full-action NLL.
For each original logical batch, sum both per-row losses and divide by 320.
The auxiliary coefficient is exactly one per eligible row; absent labels
contribute zero. Do not renormalize by the number of eligible rows, upsample
sales, change the corpus, add a head, or tune coefficients after results.

Train **one model for exactly 1,000 updates** from the immutable current base.
Reuse ordinary replica0's exact row stream, seed0, logical batch320/micro32,
MPS FP32, eval-mode dropout, AdamW(3e-5, betas .9/.999, eps1e-8, decay0),
100-update linear warmup, no gradient clipping and frozen critic. Actor,
embedding and backbone remain trainable. This is 320,000 presentations,
matched to the saved ordinary 1,000-update checkpoint. That prefix had better
recorded consequence retention than ordinary at 2,000 or 4,000 updates.
Fix the endpoint now; no duration selection or second exploratory fit.

Check the label arrays against the complete cached sequences and admitted
support census. Verify the masked loss and gradient analytically on toy
logits before training. On the first real optimizer update, save auxiliary
logits/support/targets and their gradients: the auditor must recover
`(probability - one_hot) / 320` on eligible sale rows and zero off support.
No separate model-training pilot is needed. Verify row order/exposure against
the ordinary checkpoint, all update accounting, optimizer steps and parameter
groups, unchanged critic, and loadable endpoint. Preserve failures; no retry.

Then measure base, ordinary1k and the auxiliary endpoint on the fixed training
and development caches, preserving original MPS batch32 inference layout.
Audit labels, modes, likelihoods and prospective distributions. The fixed
development spending screen requires all of:

- Lower next-Joker NLL on all 469 nontrivial replacement choices than both
  base and ordinary1k.
- Lower teacher-action NLL on all 716 SHOP sale states than both controls,
  with more exact sale matches than both controls.
- At least as many exact SHOP purchase matches as each control, and no worse
  mean teacher NLL on non-sale SHOP states than ordinary1k.

These are deterministic point-estimate spending checks on exposed development
states. Report paired game intervals as uncertainty, including all failures.
They do not establish equal action value or substitute for eventual wins.
An auxiliary prediction improvement without sale transfer fails this test.

If the screen passes, integrate the learned policy only during SHOP, using
the unchanged base in all other phases. Apply identical routing to the
ordinary1k control, isolating the supervision change and preserving the base
hand policy. Freeze the subsequent small game protocol before its outcomes;
compare sampled wins, greedy retention and an early-SHOP intervention on
fresh development games. Final qualification must still include the original
ordinary control as well as the matched shorter control, the established
fresh-win and strict learned early-attribution requirements, and one final
independent training run with untouched confirmation. No confirmation is
commissioned for this exploratory fit.

The ordinary1k snapshot took 2,108.2762 seconds. Budget this single fit at
ceil(2×2,108.2762+300) = **4,517 seconds**, with 30 seconds owner cleanup grace;
its saved-checkpoint/accounting audit receives 600 seconds. The forecast is
about 35 minutes before audit. Pin the separate inference budget from existing
scan costs before launching it. No timing pilot or automatic extension.
Serialize native jobs and check live owners roughly ten minutes apart. World
models and unrelated architecture/data changes remain deferred.
