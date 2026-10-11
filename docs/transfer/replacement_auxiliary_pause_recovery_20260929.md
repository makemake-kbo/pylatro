# Recover the user-interrupted fit without changing the experiment

The user paused the current work after 123 complete updates. The worker,
owner and caffeinate process groups were suspended at 08:48:14 UTC. On the
user's request to continue, the same processes resumed at 11:18:26 UTC after
9,012.662 seconds of suspension. Both the worker's real-time alarm and the
owner's elapsed-time budget expired during the pause. The whole owner exited
1 (tool completion `89f372`); the worker records a deadline exception. No
1,000-update checkpoint or development result exists from this attempt.

The pause implementation was inadequate: SIGSTOP retained memory but did
not suspend the two timeout mechanisms. Preserve all v1 scripts, protocol,
partial logs, pause/resume records and failure receipts. They are not
admissible model results. No checkpoint-selection decision is based on them.

The user's continuation request authorizes continuing the same experiment.
Make an explicit infrastructure exception to the original no-retry rule:
restart this interrupted fit from its identical immutable base, initialization
seed and row stream, with identical architecture, labels, objective, optimizer,
1,000-update endpoint and development criteria. The 123 interrupted updates
must be recomputed because this attempt had not saved a partial checkpoint.
This is one replacement execution of an interrupted attempt, not a second
experimental recipe or confirmation fit. Report both attempts and their cost.
No other retry or automatic budget extension is authorized by this amendment.

Version the trainer, auditor and protocol. Replace the real-time alarm with
a separate process owner enforcing the same fixed active-time budget. Both
owner and worker subtract only recorded intervals explicitly requested by
the user. The versioned pause/resume helper appends those intervals and
suspends/resumes the existing owned process groups. It must write the resume
event before sending SIGCONT. Use wall time consistently for elapsed time
and recorded pauses. The worker also checks the remaining active budget at
each update and before completion. An unpaused native hang remains bounded
by the separate owner. Keep cleanup grace and serial fit/audit ordering.

Before relaunch, test this lifecycle on disposable non-model processes:
pause for longer than the entire active-time allowance and verify that the
same owner and worker finish successfully after resume, with unchanged
process identities and no real-time alarm. Check clock arithmetic for
multiple/overlapping-phase intervals and verify an unpaused overrun is
still stopped. These tests create no policy, optimizer or training pilot.

In the actual replacement fit, compare the first 123 recorded objectives,
learning rates, row counts and loss components with the preserved prefix;
permit at most 1e-5 absolute numeric error. Recheck this in the final audit.
The original row-order, exposure, initial-policy, auxiliary-gradient,
optimizer and frozen-critic checks remain mandatory. Admission still
requires the replacement owner and its fit/audit children to exit 0.
All final long-horizon win-rate, early-attribution and final-only independent
confirmation requirements remain unchanged.
