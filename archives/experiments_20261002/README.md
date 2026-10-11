# Experiment retirement, October 2–3, 2026

All removals were restricted to explicit inventory files beneath old `eval/`
directories. Original checkpoints, production code, tests, and unrelated
working-tree edits were preserved.

| Artifact | Purpose |
| --- | --- |
| `inventory.jsonl` | Every original artifact's path, size, allocation, timestamp, keep/archive/delete assignment, and destination |
| `evidence.tar.gz` | Historical source, configuration, JSON/JSONL/CSV evidence, final results, terminal outcomes, reservations, and October 2 review findings |
| `working_tree_source.tar.gz` | Source, tests, tools, scripts, and reports captured before implementation |
| `working_tree.patch` | Original tracked working-tree changes, including binary diff metadata |
| `working_tree_status.txt` | Original tracked and untracked path inventory |
| `preservation_verified.json` | Archive and inventory hashes, verified member count, and unopened-outcome handling |
| `removed.jsonl` | Compact per-file deletion receipt with original size and preservation destination |
| `before.json`, `receipt.json` | Measured storage before and after retirement |

All three retained model files were copied, hashed against their original,
and strictly loaded. Every relocated dataset file was independently hashed;
counts were checked at 5,000, 1,000, and 4,000 games. Every evidence archive
member was streamed back and hashed against its source before deletion. The
working-tree source archive was also streamed to verify readability.
Reserved outcomes were archived as opaque bytes without statistical inspection.

Archive members retain their old repository-relative paths. Restore a named
member to an isolated directory when needed, for example:

```sh
mkdir -p /tmp/pylatro-history
tar -xzf archives/experiments_20261002/evidence.tar.gz \
  -C /tmp/pylatro-history \
  eval/model_play_review_20261002/complementarity.json
```

Evidence remains available, but removed checkpoints, optimizer recovery state,
regenerated arrays, and compiled duplicate runtimes prevent full replay of some
experiments. The stable registries are `incumbents/registry.json` and
`datasets/registry.json`. The [current status](../../docs/transfer/README.md)
replaces the old running chronology; the previous README is in the source archive.
