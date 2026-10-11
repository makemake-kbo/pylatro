"""Render the fixed study's saved results without changing its selection rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(out):
    def read(name):
        path = out / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    verdict = read("verdict") or {"verdict": "INCOMPLETE", "reason": "Required fixed panels have not all completed."}
    lines = [
        "# Public-information hybrid gameplay study",
        "",
        f"**{verdict['verdict']}: {verdict['reason']}.**",
        "",
        "Blue Deck, White Stake, Ante 8; CPU FP32, four workers, one Torch thread per worker; "
        "temperature 1 and independent per-game policy RNG. Neural weights were frozen throughout. "
        "All arms used public-information-v2 observations, legal masks, sampled derived features, "
        "and the same public-sample Joker ordering. Historical win rates are reference results under older semantics.",
        "",
    ]
    confirmation = read("confirmation_result")
    architecture = read("architecture_selection")
    if confirmation and architecture:
        model, mode = architecture["baseline"], architecture["selected"]
        assisted, baseline_arm = confirmation["arms"][f"{model}/{mode}"], confirmation["arms"][f"{model}/off"]
        effect = confirmation["paired"]
        low, high = effect["interval95"]
        lines[4:4] = [
            f"Greedy confirmation: **{mode} assistance won {assisted['wins']}/{assisted['games']} "
            f"({100 * assisted['wins'] / assisted['games']:.2f}%), versus "
            f"{baseline_arm['wins']}/{baseline_arm['games']} "
            f"({100 * baseline_arm['wins'] / baseline_arm['games']:.2f}%) for the neural baseline**. "
            f"The paired gain was **{effect['points']:+.2f} percentage points**, with a 95% game-bootstrap "
            f"interval of **[{low:+.2f}, {high:+.2f}]**. The incumbent is `{model}`. "
            f"The primary criterion {'passed' if confirmation['success'] else 'failed'}; "
            "the sampled panel is secondary.",
            "",
        ]
    pilot = read("pilot_technical")
    invalidated = read("invalidated_attempt_1")
    if invalidated:
        lines.extend(
            [
                "An earlier runtime attempt was invalidated because the simulator's visual-only Amber Acorn "
                "approximation exposed the concealed Joker order. Its 16 pilot games, 384 baseline games, and "
                f"{invalidated['completed_games']['development']} partial architecture games are retained separately "
                "and excluded from every result below. Its architecture outcomes were not opened. "
                "Public-information-v2 canonicalizes the known owned roster, samples concealed order for simulation, "
                "and disables ordering and indexed Joker actions while that order is unknown. "
                "All five replacement seed panels are fresh and disjoint from the invalidated attempt, with identical "
                "panel sizes and search settings. The original 48-hour clock remains in force.",
                "",
            ]
        )
    invalidated_second = read("invalidated_attempt_2")
    if invalidated_second:
        counts = invalidated_second["completed_games"]
        lines.extend(
            [
                "A second attempt stopped during confirmation when the legacy continuation heuristic proposed "
                "card slot 16, outside the existing 16-card action vocabulary. That attempt completed 16 pilot, "
                "384 baseline, 1,024 architecture, and "
                f"{counts['confirmation']['complete']} confirmation games; its one failed game was recorded, "
                "never counted as a loss. Confirmation win outcomes were not opened. The invalidated architecture "
                "panel had 123 neural-only, 145 hand-only, 116 shop-only, and 149 combined wins out of 256; "
                "these are unconfirmed historical results, excluded from the replacement analysis below. "
                "The correction restricts heuristic proposals to addressable card slots while retaining the full "
                "hand for engine transitions and scoring, and bounds concealed Aura masks to the same vocabulary. "
                "All five panels were reserved again under a fresh V3 seed namespace, excluding both earlier "
                "reservations. Observation semantics remain public-information-v2. Search settings and selection "
                "rules did not change. A new technical-phase allowance was recorded explicitly; the original "
                "overall 48-hour and architecture deadlines were retained. All prior evidence is preserved under "
                "`runs/public_hybrid_20261002/invalidated_attempt_2/`.",
                "",
            ]
        )
    if pilot:
        lines.extend(
            [
                f"The technical pilot completed {pilot['games']} games and {pilot['decisions']:,} decisions, "
                f"with {pilot['boundary_audits']} private-state perturbation checks, "
                f"{pilot['illegal_actions']} illegal actions, and {pilot['fallbacks']} searches falling back without "
                f"a completed paired comparison. It projected {pilot['projected_study_seconds'] / 3600:.2f} "
                "hours for the full study using the combined arm's measured cost for every game. "
                "Pilot wins did not select search settings.",
                "",
            ]
        )
    baseline = read("baseline_selection")
    for label, result in (
        ("Baseline selection: 128 common seeds", baseline),
        ("Architecture comparison: 256 separate seeds", architecture),
        ("Greedy confirmation: 400 untouched seeds", read("confirmation_result")),
        ("Secondary sampled comparison: 128 additional seeds", read("sampled_result")),
    ):
        if result is None:
            continue
        lines.extend(
            [
                f"## {label}",
                "",
                "| Arm | Wins | Rate | Changed decisions | Searches | Fallbacks "
                "| Mean / p95 / max decision seconds | Worker hours |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        mode_order = {"off": 0, "hand": 1, "shop": 2, "both": 3}
        for key, stats in sorted(result["arms"].items(), key=lambda item: mode_order[item[0].rsplit("/", 1)[-1]]):
            lines.append(
                f"| {key} | {stats['wins']}/{stats['games']} | {100 * stats['wins'] / stats['games']:.2f}% "
                f"| {stats['changes']:,} | {stats['searches']:,} | {stats['fallbacks']:,} "
                f"| {stats['mean_decision_seconds']:.3f} / {stats['p95_decision_seconds']:.3f} "
                f"/ {stats['max_decision_seconds']:.3f} "
                f"| {stats['worker_seconds'] / 3600:.3f} |"
            )
        lines.append("")
        if "selected" in result:
            lines.extend([f"Selected: **{result['selected'] or 'none'}**.", ""])
        effects = result.get("paired", {})
        if "points" in effects:
            effects = {"assisted minus baseline": effects}
        if effects:
            lines.extend(
                [
                    "| Paired contrast | Gain (percentage points) | 95% game-bootstrap interval "
                    "| Gained / lost games |",
                    "| --- | ---: | --- | ---: |",
                ]
            )
            for mode, effect in effects.items():
                low, high = effect["interval95"]
                lines.append(
                    f"| {mode} | {effect['points']:+.2f} | [{low:+.2f}, {high:+.2f}] "
                    f"| {effect['gained']} / {effect['lost']} |"
                )
            lines.append("")
    counts = {}
    for stage in ("pilot", "baseline", "development", "confirmation", "sampled"):
        files = list((out / stage).rglob("*.json"))
        counts[stage] = dict(
            complete=sum(not p.name.endswith(".error.json") for p in files),
            errors=sum(p.name.endswith(".error.json") for p in files),
        )
    lines.extend(
        [
            "## Accounting and interpretation",
            "",
            "| Panel | Completed games | Recorded errors |",
            "| --- | ---: | ---: |",
        ]
    )
    for stage, count in counts.items():
        lines.append(f"| {stage} | {count['complete']} | {count['errors']} |")
    verification = read("final_verification")
    if verification:
        lines.extend(
            [
                "",
                f"Final verification accounted for all **{verification['total_games']:,} games** and independently "
                "recomputed every published summary and paired statistic. All "
                f"{verification['source_files_verified']} frozen scientific source files and all three checkpoint "
                "hashes matched. The 928 reserved seeds are unique and disjoint from 1,856 seeds reserved for "
                "invalidated attempts, as well as the retained historical exclusions. The original clock elapsed "
                f"**{verification['elapsed_hours']:.2f} hours**, including invalidated attempts and interruptions, "
                "within its 48-hour cap. Final-study game execution used "
                f"{verification['worker_hours']:.2f} aggregate worker hours; this excludes invalidated attempts. "
                "The broad preflight passed 149 tests using the compiled engine path; seven reporting/resume "
                "regression tests subsequently passed. Outcome hashes and these checks are saved in "
                "`outcome_hashes.json` and `final_verification.json`.",
                "",
                "The maximum measured planning interval was "
                f"{verification['max_observed_planning_seconds']:.3f} seconds, including timer and cleanup "
                "overhead around the three-second planning budget. Decision latency additionally includes "
                "observation construction and neural inference, as shown above.",
            ]
        )
    lines.extend(
        [
            "",
            "The fixed architecture advances an assisted arm only at a greedy gain of at least five percentage points. "
            "Confirmation additionally requires a paired 95% game-bootstrap interval strictly above zero, "
            "no information-boundary violations, and no illegal or silently dropped games. Bootstrap intervals use "
            "10,000 resamples of complete paired games with generator seed 1729. Baseline ties prefer PPO replica 0, "
            "then replica 1, then supervised; assisted ties prefer fewer assisted phases, then measured latency.",
            "",
            "Search uses at most six hand candidates or four shop plans and four shared sampled worlds. "
            "Only equally completed sample sets are compared; deadline handling can retain fewer "
            "than four complete sets. "
            "A fallback means no completed comparison, in which case the neural proposal is retained. "
            "Changed decisions count overrides of the model proposal at the same public decision state. "
            "Continuation caps are 60 hand actions and 100 shop actions. The three-second limit covers planning; "
            "reported decision latency also includes neural inference and public observation construction. "
            "Cost is aggregate worker wall time, not metered CPU time or financial expenditure.",
            "",
            "Teacher agreement and forecast error were not advancement gates. The study measures actual terminal wins; "
            "it does not claim that unchanged neural weights learned new strategy. No outcome-selected coefficient "
            "sweep or additional auxiliary-loss experiment follows from these results.",
            "",
            "See [player instructions and historical findings](README.md), "
            "[incumbents](../../incumbents/registry.json), [demonstrations](../../datasets/registry.json), "
            "and the [cleanup receipt](../../archives/experiments_20261002/receipt.json). "
            "Seed reservations, frozen source/configuration hashes, the frozen runtime archive, "
            "per-game outcomes and latency arrays, and verification logs are in "
            f"`{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}`.",
            "",
        ]
    )
    adapter = read("reporting_adapter")
    if adapter:
        lines.extend(
            [
                "All 400 confirmation pairs finished before a NumPy boolean caused JSON summary serialization "
                "to stop. A separately hashed reporting adapter converts NumPy scalars to native JSON values. "
                "The study resumed saved games without replaying them; all frozen scientific source files, "
                "gameplay settings, and seed lists remained unchanged. The interruption and adapter hash are "
                "preserved in `reporting_interruption.json` and `reporting_adapter.json`.",
                "",
            ]
        )
    frozen = read("frozen")
    if frozen:
        lines.extend([f"Frozen source identifier: `{frozen['source_id']}`.", ""])
    (ROOT / "docs/transfer/public_hybrid_20261002.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/public_hybrid_20261002")
    render(parser.parse_args().output)
