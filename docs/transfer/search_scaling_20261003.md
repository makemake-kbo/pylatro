# Search scaling study, October 3, 2026

Completed verdict: **retain the four-world incumbent**. Development wins were
58/96 incumbent, 56/96 moderate, and 62/96 high64. The largest observed gain,
+4.17 points (paired 95% interval [-4.17, +12.50]), missed the fixed five-point
advancement threshold. Confirmation was not run; its 256 reserved seeds remain
unused. The gameplay study finished in 5.66 hours with no recorded errors.
See the [combined results and diagnosis](../../runs/search_scaling_20261003/report.md).

The player accepts `--search-samples {4,16,32,64}` with `--planning-budget`.
Four worlds and three seconds remain the default. The recommended incumbent
command in `public_hybrid_20261002.md` changes only after successful confirmation.

The separate runner is `tools/search_scaling_study.py`. Its output directory is
`runs/search_scaling_20261003/`; the generated `report.md` is the combined results
report. The existing study, runtime archive, checkpoints, registry and datasets
are preserved.

```sh
.venv/bin/python tools/search_scaling_study.py prepare
.venv/bin/python tools/search_scaling_study.py run
.venv/bin/python tools/search_scaling_study.py status
.venv/bin/python tools/search_scaling_study.py report
.venv/bin/python tools/finalize_search_scaling_report.py
```

The finalizer regenerates the report's additional descriptive notes from the
completed artifacts. It performs no gameplay or selection; its separate source
hash receipt identifies this reporting work after the scientific run.

`prepare` runs correctness tests, checks compiled execution, audits retained and
archived training/evaluation metadata without inspecting reserved outcomes,
reserves disjoint panels, and freezes hashes. It does not start gameplay.
`run` starts the immutable 24-hour clock and resumes atomic completed games.
Use the same output directory to resume; interruptions never reset the clock.
A process lock prevents concurrent study coordinators. Four spawned workers
use CPU FP32 and one Torch thread each. An error or incomplete fixed panel
cannot advance and is never imputed as a loss.

All 400 previous confirmation pairs receive descriptive analysis. Diagnostic
replays use all 32 regressions and SHA-256 ascending selections of 16 shared
losses, eight rescues and eight shared wins. Both policies replay these 64 seeds.
Only diagnostics write compressed public decision traces. The public snapshot,
proposal, candidate plans, selected action, completed paired sample scores,
stop reason, resources, and observed public transition are recorded after the
planning timer. Live private RNG, future draws and private card identifiers
never enter trace records.

Final-blind sensitivity diagnostics hold the recorded neural proposal fixed
and reevaluate the first eligible hand decision at all four search profiles,
with sample prefixes beginning at 0 and 10,000. These simulated continuation
results do not establish whether an actual game was winnable.

Eight new pilot seeds run under each profile: 4/3s, 16/3s, 32/10s and 64/10s.
Pilot win fields remain sealed until profile selection. Timing-only selection
prefers 64 worlds if the projected 96-seed, three-arm development plus worst-case
256-pair confirmation fits the remaining clock with 25% margin; otherwise it
uses 32 if feasible. The projection includes two maximum-game scheduling tails.
No smaller panels are substituted.

Development advances the highest-win challenger only at a gain of at least
five percentage points, with ties resolved by lower mean decision latency,
then fewer requested worlds. Confirmation adoption additionally requires the
paired 95% bootstrap interval to be strictly positive and all integrity checks
to pass. Bootstrap uses 10,000 paired game draws and seed 1729. Sampled-policy
evaluation, training, new continuations, live play and shop redesign are deferred.
