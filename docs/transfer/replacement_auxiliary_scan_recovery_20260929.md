# Preserve completed predictions and correct the control hash convention

Scan owner24868 exited1 (`ee1220`, observed12:42:09UTC). It completed both
training/development prediction files for the auxiliary and base policies,
then failed the ordinary1k pre-inference state-hash assertion. The failed
measurement took1,178.1981 active seconds. It produced no admitted scan result;
no performance summary or spending decision has been inspected.

The ordinary hash came from the duration study, whose `tensor_digest` sorts
state-dictionary entries by name. The new scan's `tensor_hash` iterates model
state-dictionary order. Both include the same names, dtype, shape and bytes,
but in different orders. CPU checkpoint verification (`4d54d5`) computed both
digests for all three immutable checkpoints. It reproduces the historical
ordinary hash exactly under the sorted convention and all existing auxiliary
and base hashes under the model-order convention. No tensor changed.
The ordinary model-order digest is
`2f6b045bd137b9fa2428f827d251ec93dc3576da5f795956b37d72071e92adde`;
its preserved sorted digest is
`829582b3df55c8039977eb27a150c34c84785b6a2087caa44b4bc539d771c7d6`.
The verification loaded checkpoints but performed no model forward or fit.

Version the scan and auditor. Use the verified model-order digest for all
three policies while retaining the original protocol, failed-owner receipt,
and both hash conventions. This corrects a provenance representation error;
it does not relax the unchanged-model requirement, prediction tolerances,
spending checks, checkpoint choices, data panels or batch layout.

Reuse the four complete auxiliary/base prediction files as unadmitted raw
artifacts. Freeze their hashes, verify every identity and legal action, check
all base archived predictions, and retain all prospective labels. Both models'
post-scan hash checks were reached and passed before the ordinary model's
pre-scan assertion, as established by the failed source location and complete
worker log. Complete only the missing ordinary1k scan. The existing audit
still re-forwards every development row and the first320 training rows for
all three policies and audits every saved record and prospective distribution.
No result is admitted until the corrected whole owner and both phases exit0.

There are456,372 reused predictions and228,186 new measurement forwards. The
same pinned historical cost projects573.3638seconds for the remaining model.
Budget ceil(2×forecast+120)=1,267seconds, retaining the698-second audit budget.
The failed measurement plus this new allowance stays below the original
3,561-second measurement budget. Keep the tested pause-aware lifecycle and
30-second cleanup grace. No additional fitting, outcome-dependent selection,
automatic further retry or performance-threshold change is authorized by
this infrastructure correction.
