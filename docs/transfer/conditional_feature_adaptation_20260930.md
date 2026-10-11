# Adapt only the small readout's features

The converged 5,000-game output fit fails seven of thirteen fixed development checks. It improves sale likelihood versus the 1,000-game fit but loses three full sale matches. Its remaining feature gradient is locally favorable on development: all four projection-block cosines are positive, and the fixed negative training-gradient derivative has a descriptive game interval entirely below zero. Familiar Joker identities and identity pairs also retain a large transfer gap. These findings motivate learning the existing readout's features; they do not prove that doing so will improve a policy.

## Single controlled change

Start from the admitted broader-data output checkpoint. Unfreeze only `state_proj.weight`, `state_proj.bias`, `owned_proj.weight`, and `offer_proj.weight` under `action_grammar_head.replacement_relation`: **57,408 parameters**. Keep the 64 output weights and common output bias fixed. The encoder, every old head, purchase proposal and critic remain fixed. There is no width, context, input, label, data-population or policy-factorization change.

Use exactly the same 26,782 eligible sale states from the same 5,000 current-rule games, including all outcomes. Reuse the already audited frozen encoder features and purchase logits. The training objective is the same conditional teacher NLL summed over these states and divided by 1,013,555 original corpus rows. Training reads no development labels. No new game collection or teacher query is required.

The old convex Newton solver applies to the 64 output weights only. For nonlinear feature fitting, use the installed PyTorch L-BFGS implementation on CPU float64 with full-batch gradients accumulated in fixed 256-row chunks, preserving original-then-additional row order. The one fixed recipe is `lr=1`, `max_iter=50`, `max_eval=200`, `history_size=10`, `line_search_fn='strong_wolfe'`, `tolerance_grad=1e-8`, `tolerance_change=1e-12`. Permit at most 201 closure evaluations to account for the line-search boundary; exceeding this fails the run. Call the optimizer once, preserving its curvature history across its internal iterations. Its training-only numerical stopping rules may stop earlier. Use its final endpoint, without a checkpoint, epoch, coefficient or optimizer sweep. Record actual iterations, evaluations, losses, gradient norms and optimizer state. This is a bounded nonlinear fit, not a claim of global convergence or matched computational exposure.

## Technical admission before fitting

First perform a zero-update check over the complete training population. CPU float64 logits and loss must reconstruct the admitted fixed-feature endpoint; its 57,408 feature gradients must match the independent cached-feature calculus. Check parameter freezing and row identities. Exercise L-BFGS on a small synthetic quadratic separately; that changes no policy parameter. Pin the installed optimizer source and Torch version. This preparation performs zero policy optimizer steps.

Preflight is capped at 600 active seconds plus 30 seconds of owner grace. Fit allowance is derived before optimization as `ceil(2 * 202 * max(measured_complete_gradient_seconds, 1) + 120)`, capped at 3,600 seconds; if that projection exceeds the cap, stop for a technical implementation decision before fitting. The separate endpoint/native audit allowance is 900 active seconds plus 30 seconds of owner grace. Explicit user pauses suspend clocks. Native/model jobs remain serialized; long jobs are checked about ten minutes apart.

## Endpoint audit and unchanged decision

Export only the four feature tensors to float32 in the existing native checkpoint. Require all other tensors and policy metadata to equal the parent. Independently reconstruct initial, terminal and deployed conditional objective/gradients with the existing NumPy calculus, then replay all 26,782 training rows through native inference. Retain the existing 2e-5 native logit/NLL bounds. Record any near-boundary float64/native mode disagreement; native choices remain authoritative. The deployed training loss must be finite and lower than the parent. A failed audit is a technical failure and cannot qualify a policy.

After successful whole-fit and audit admission, run the same full teacher-state scan and independent native audit. Keep all thirteen original development checks against base, ordinary1k, auxiliary and disabled readout. Add the completed broader-data fixed-feature model as a descriptive control; it replaces no original control. Predict lower sale NLL and more exact full sale matches than that model. A failure stops this endpoint before gameplay or confirmation.

Only a qualifying endpoint advances to the original fresh-game sampled gain, greedy retention, early-policy attribution and one final independent-fit/untouched-confirmation requirements. No learned long-horizon or win-rate gain is established.

[Broader-data result](conditional_output_diversity_results_20260930.md) · [Remaining-gradient evidence](conditional_diversity_gradient_results_20260930.md) · [Identity coverage](conditional_context_coverage_results_20260930.md)
