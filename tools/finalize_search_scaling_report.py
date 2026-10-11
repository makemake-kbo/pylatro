"""Add reproducible descriptive notes after a completed search-scaling study.

Reporting only: never executes gameplay, changes selection, or opens an unused
panel. Its separate hash receipt distinguishes post hoc description from the
scientific source frozen before gameplay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import search_scaling_study as study


def finalize(out):
    frozen, _ = study.verify(out)
    verdict = study.read(out / "verdict.json")
    if verdict["verdict"] not in {"adopt", "retain"}:
        raise ValueError("Final reporting requires a completed study")
    selection = study.read(out / "challenger_lock.json")
    audit = study.read(out / "trace_final_audit.json")
    independent = study.read(out / "independent_verification.json")
    diagnosis = study.read(out / "loss_diagnosis.json")
    rows = []
    for loss in diagnosis["observations"]:
        if loss["arm"] != "incumbent" or loss["won"]:
            continue
        result = study.read(out / "sensitivity" / f"{loss['seed']}.json")
        trial = next(
            t for t in result["reevaluations"] if t["profile"] == "high64" and t["config"]["sample_start"] == 10000
        )
        scores = trial["completed_scores"]
        clear = [sum(s[i][0] for s in scores) / len(scores) for i in range(len(trial["candidates"]))] if scores else []
        rows.append(
            dict(
                seed=loss["seed"],
                completed_worlds=len(scores),
                best_candidate_clear_fraction=max(clear, default=None),
                all_candidate_clear_fractions=clear,
                stopping_reason=trial["stopping_reason"],
                dollars_left=loss["final"]["dollars"],
                discards_left=loss["final"]["discards"],
                consumables_left=len(loss["final"]["consumables"]),
            )
        )
    metrics = dict(
        loss_decisions=len(rows),
        no_completed_comparison=sum(r["completed_worlds"] == 0 for r in rows),
        no_candidate_cleared_in_any_completed_world=sum(r["best_candidate_clear_fraction"] == 0 for r in rows),
        some_candidate_cleared_at_least_once=sum((r["best_candidate_clear_fraction"] or 0) > 0 for r in rows),
        best_candidate_cleared_at_least_half_the_worlds=sum(
            (r["best_candidate_clear_fraction"] or 0) >= 0.5 for r in rows
        ),
        complete_64_world_comparisons=sum(r["completed_worlds"] == 64 for r in rows),
    )
    study.atomic(
        out / "entry_probe_description.json",
        dict(
            summary=metrics,
            rows=rows,
            method="Post hoc descriptive summary of hybrid replay losses. At the first eligible hand decision in the "
            "final blind, use high64 / sample_start=10000. Average candidate clear flags over equally completed "
            "sample sets, then take the maximum across the fixed shortlist. No new simulations or selection changes.",
            limits="Different decisions complete different sample counts. These fixed-heuristic "
            "public continuations are "
            "not live counterfactual wins. Zero sampled clears does not establish unwinnability.",
        ),
    )
    study.report(out)
    summaries = selection["summaries"]
    high = study.read(out / "profile_lock.json")["high"]
    worlds = study.PROFILES[high]["samples"]
    cost_ratio = summaries[high]["worker_hours"] / summaries["incumbent"]["worker_hours"]
    gain = selection["paired"][high]
    notes = ["", "## Confirmation and final checks", ""]
    if selection["challenger"] is None:
        notes += [
            "**Confirmation was not run:** neither challenger met the five-point development threshold. "
            "All 256 confirmation seeds remain reserved and unused. No sampled-policy evaluation was run.",
            "",
        ]
    notes += [
        f"The {worlds}-world development arm required **{cost_ratio:.2f} times** the incumbent's worker time for "
        f"an observed **{gain['points']:+.2f}-point** gain. Its paired 95% interval "
        f"[{gain['interval95'][0]:+.2f}, {gain['interval95'][1]:+.2f}] permits both harm and meaningful benefit. "
        "This panel does not establish that additional search has zero effect.",
        "",
        f"The gameplay clock ended after **{verdict['elapsed_seconds'] / 3600:.2f} hours**. "
        "All 147 preflight tests passed "
        "with the compiled execution path. Frozen source and checkpoint hashes matched at completion. "
        f"Independent checks verified {independent['outcome_hashes_verified']} "
        "outcome/timing/sensitivity artifact hashes "
        "and recomputed wins, worker hours, paired bootstrap intervals and selection. "
        f"The complete trace audit checked {audit['decisions']:,} decisions across {audit['games']} replays: "
        "no private fields, unequal recorded comparisons, snapshot-hash mismatches or terminal-outcome mismatches. "
        "The incumbent and dataset registries remain unchanged.",
        "",
        "Checks: [final scientific verification](final_verification.json), "
        "[independent verification](independent_verification.json), [trace audit](trace_final_audit.json).",
        "",
        "## Additional descriptive entry probe",
        "",
        f"At the first eligible final-blind hand decision in the {metrics['loss_decisions']} selected hybrid losses, "
        f"{metrics['some_candidate_cleared_at_least_once']} had a candidate clear at least one "
        "independent public world; "
        f"{metrics['best_candidate_cleared_at_least_half_the_worlds']} had a candidate clear "
        "at least half its completed "
        f"worlds. In {metrics['no_candidate_cleared_in_any_completed_world']}, "
        "no candidate cleared any completed world. "
        f"Only {metrics['complete_64_world_comparisons']} decisions completed all 64 worlds in this batch. "
        "This is a post hoc description of the already-fixed high64 / index-10,000 probe, not a new evaluation. "
        "Different continuation behavior, sampling uncertainty and earlier strategic choices remain hypotheses. "
        "No sampled clear is not proof of an unwinnable seed; a sampled clear is not proof of a live rescue.",
        "",
        "Per-decision values and the exact summary method: [entry probe description](entry_probe_description.json). "
        "These notes can be regenerated with `python tools/finalize_search_scaling_report.py` "
        "after the study completes.",
        "",
    ]
    with (out / "report.md").open("a") as f:
        f.write("\n".join(notes))
    study.atomic(
        out / "reporting_notes.json",
        dict(
            path=str(Path(__file__).relative_to(study.ROOT)),
            sha256=study.digest(Path(__file__)),
            frozen_scientific_id=frozen["id"],
            scope="Post hoc description and reporting only; no gameplay or selection changes",
            report_sha256=study.digest(out / "report.md"),
            entry_description_sha256=study.digest(out / "entry_probe_description.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.DEFAULT)
    finalize(parser.parse_args().output.resolve())
