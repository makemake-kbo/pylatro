# Standard-pack enhancement is missing from policy inputs

A deterministic fixture confirms that the policy cannot distinguish Glass from
Mult enhancements when choosing Standard-pack cards with otherwise identical
attributes. Swapping the enhancements between two slots leaves **all 14 policy
observation arrays exactly equal**, including tokens, scalars, legal-action masks
and played-hand history. Claiming the same slot produces a different permanent
deck addition.

For this pair, a policy using only these observations must produce the same action
distribution. It therefore cannot adapt its slot preference to the swapped
enhancements. An independent read-only review found no invalidating mistake in
the reproduction and confirmed this limited conclusion.

This establishes an input-representation defect. Its frequency in normal games,
effect on win rate, and contribution to the current training plateau remain
unmeasured. No model was changed or trained.

## Reproduction

The [isolated reproducer](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/diagnose_pack_enhancement_observation.py)
constructs a shop fixture, opens a Standard pack through the engine, and replaces
two offers with equal-cost ten-of-spades cards. The two cases differ only in which
of those slots contains Glass or Mult. Remaining offers and the preceding
constructed state are held fixed. Both claims are legal.

| Pack fixture | Slot 0 | Slot 1 | Single-card score after claiming slot 0 / slot 1 |
| --- | --- | --- | --- |
| Glass first | Glass | Mult | 30 / 75 |
| Mult first | Mult | Glass | 75 / 30 |

Each claim increases the deck from 52 to 53 cards and preserves the chosen
enhancement. The score calculation demonstrates a downstream engine difference;
it does not estimate terminal-win value or establish which card is optimal for a
whole game. These are constructed legal-offer fixtures, not sampled reachable
full-game prefixes.

The [result and source hashes](data/pack_enhancement_observation_collision_20260922.json)
record equality of every policy input. The complete arrays are preserved in
[Glass-first](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/pack_enhancement_observation_collision/glass_first.npz)
and [Mult-first](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/pack_enhancement_observation_collision/mult_first.npz)
artifacts; the archives are byte-identical. The diagnostic ran in approximately
0.75 seconds, with zero policy inference calls, training updates or full-game
evaluations. Frozen study source hashes were checked before and after execution.

## Cause and implications

[`Tokenizer._encode_pack_card`](../../src/pylatro_agent/tokenizer.py) assigns all
Default/Enhanced playing-card offers item ID zero and type five. It encodes rank,
suit, price, edition and seal, but drops enhancement identity. The
[`ShopEmbedding`](../../src/pylatro_agent/embeddings.py) also has no enhancement
input. Diagnostic environment information contains the enhancement, but the
policy receives only the observation arrays checked here.

Consequently, additional training or a larger transformer cannot recover this
distinction from the present observation. A representation change must expose
the enhancement and make it available to the policy's embedding. Existing
checkpoint compatibility and tokenizer versions must be handled explicitly.

The active evaluation remains unchanged so that its results remain interpretable.
A future fix will need direct observation/embedding checks and fresh matched
training and full-game evaluation. Fixing this omission alone would not prove
long-horizon generalization.
