# Dense label collection: validation after owner exit 1

The original collection owner (PID77635) exited with code 1, observed by the
already attached Darwin kernel exit watcher. Preserve that result. Both native
workers have direct-parent wait receipts with exit 0 and no error: training
PID77636 and development PID98877. The original complete manifest exists and
references those receipts. The saved owner log contains no traceback. The exact
cause of the owner's finalization failure is not established; a lost terminal
pipe is a hypothesis, not an observed exception.

The original collection is not admitted on those facts alone. A separate pure
validation owner will run the unchanged, protocol-pinned
`shop_dense_labels_verify_v1.py` once on the existing complete outputs. It must
verify all original source and cache pins, all 228186 factual observation hashes,
all 64769 selected inputs and labels, the complete 1128-game population, native
filter and gain arithmetic, fresh controls, and archived pilot controls. Its
900-second active budget plus 30 seconds of owner grace is fixed before execution.
Explicit user pauses are excluded by the existing pause-aware controller.

Admission requires the new validation owner to exit 0 and its original audit
worker to exit 0. The admission record must identify itself as a post-collection
validation owner, retain the original exit-1 observation and native worker
receipts, and reference the unchanged collection manifest plus full audit.
It must never represent the original owner as having exited 0.

No collection source, protocol, raw label, worker receipt, or failure record is
overwritten. There are no native queries, simulator runs, model forwards,
optimizer steps, label repairs, population changes, or restarts in this step.
The training recipe, all 13 policy checks, fresh-game comparisons, early-policy
attribution requirement, and final independent confirmation remain unchanged.
Passing this validation establishes usable labels, not learned generalization.
