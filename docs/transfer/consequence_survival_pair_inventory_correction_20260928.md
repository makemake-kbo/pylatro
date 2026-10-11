# Correct target-file visibility before recounting

Inventory v1 exited with code 1 at its final emitted-row count assertion. The writer was still open and had not been flushed before the script opened a second reader. All earlier source/label/pair checks passed, but the reader missed buffered trailing rows. No scientific result was written or admitted.

Preserve v1's helper, protocol, emitted targets, worker log and failed owner receipt. Version 2 explicitly flushes the writer before the same recount and uses new output, protocol and execution paths. It pins the failed v1 evidence and retains the exact original labels, prediction arrays, pair definition, counts, tolerances and ranking statistics. This is a standard-library bookkeeping correction with no model fit, inference, new simulator outcome, gate change or scientific retry.
