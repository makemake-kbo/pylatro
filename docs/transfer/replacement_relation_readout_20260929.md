# Coupled replacement readout

All quantities are evaluated at temperature one, matching the fixed experiment.
Let `m` be macro logits, `s_i` the original owned-Joker sale logits, and `J`
the legal Joker-sale slots from the unchanged action mask. Let `o_j` be present
Joker-offer slots from tokens and attention, independent of current buy legality.

The readout has `W_state:128→64` with bias, `W_owned:384→64` without bias,
`W_offer:384→64` without bias and `w_out:64→1` with bias: 57,473 parameters.
The output layer starts exactly zero; the earlier projections use seed-zero
PyTorch initialization. Existing model parameters load unchanged.

```
e_ij = w_out(GELU(W_state(state) + W_owned(joker_i) + W_offer(offer_j)))
p_j  = softmax(existing_shop_buy_logits restricted to present Joker offers)
r_i  = sum_j p_j * e_ij
```

The gate is current phase SHOP, encoded free-Joker capacity `<=0`, at least
one present Joker offer and at least one legal Joker sale. Invalid sale slots
and rows outside the gate receive exactly zero residual. Empty offer/sale
supports use a dummy arithmetic entry, then mask their result to zero; they
must not create NaNs or gradients. Future targets and teacher actions are
excluded from the readout inputs and gate. Meaningful contextual positions
remain in each owned/offer vector.

```
s'_i = s_i + r_i
delta = logsumexp_{i in J}(s_i + r_i) - logsumexp_{i in J}(s_i)
m'_sell = m_sell + delta
```

All other macro and conditional logits are unchanged. This yields
`P'(sale_i) ∝ P(sale_i) exp(r_i)` and `P'(a) ∝ P(a)` for non-sale actions,
with one common normalization. Zero residual reproduces the entire native
policy exactly. A positive constant residual raises the sale-family mass
while preserving conditional sale odds. A varying residual changes both.

Use one transient grammar-head hook around the existing, unmodified native
distribution call. Its only additional external input is the visible legal
action mask. Compute the pair features from the head's current contextual
inputs and its existing state projections; do not rerun the backbone. Attach
the new module under `action_grammar_head.replacement_relation` so the existing
actor parameter grouping and checkpoint state dictionary include it. Remove
the hook in `finally`, including on failure. Keep uncorrected grammar output
available for the fixed within-model ablation. A study-specific loader must
attach the extension before strict checkpoint loading; never silently discard
its weights in the legacy loader.

Before training, verify the following without optimizer steps:

- CPU float64 and MPS float32 joint probabilities agree with a separately
  enumerated flat-action tilt; gradients agree with `P'(sale_i)-1[teacher=i]`.
- Zero residual, empty support, illegal slots and inactive gates behave
  correctly; masked offers and illegal sales have zero gradients.
- Constant residual changes whether to sell; differing pair scores change
  which Joker to sell. Gradients reach offered and owned features when the
  output layer is nonzero. At initialization, upstream zero gradients are
  expected because the output weights are zero.
- Native original-layout training observations reproduce all original logits,
  modes and action NLL with the neutral extension; old parameters and critic
  stay unchanged. The hook is removed after calls and exceptions.
- The extension is strictly loadable from its saved state, and the native
  action distribution agrees with direct tilting for a nonzero readout.

On the first actual training update, verify the purchase auxiliary gradient
at the neutral readout as before. Independently verify the new output-bias
gradient: over active rows, sum `(P(sale)-1[teacher sale])/320`, allowing only
the predeclared FP32 numerical tolerance. After learning begins, action NLL
can also backpropagate into purchase preference through the coupled readout;
the old auxiliary-only logit-gradient identity no longer applies there.
