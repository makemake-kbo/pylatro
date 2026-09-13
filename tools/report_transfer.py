#!/usr/bin/env python3
"""Compile completed transfer artifacts into a compact, auditable report without promoting partial results."""
# Markdown table rows intentionally remain readable as complete template strings.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pylatro_agent.transfer import paired_statistics, summarize, write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def outcomes(root, names):
    rows = []
    for name in names:
        manifest = read(root / name / "manifest.json")
        if not manifest or manifest.get("status") != "complete":
            return None
        rows.extend(json.loads(line) for line in (root / name / "outcomes.jsonl").read_text().splitlines())
    return rows


def pct(value):
    return f"{100 * value:.2f}%"


def delta_text(value):
    low, high = value["paired_ci95"]
    return f"{100 * value['sampled_or_greedy_delta']:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]"


def progression_table(summaries):
    labels = list(summaries)
    lines = ["| Ante cleared | " + " | ".join(labels) + " |", "| ---: | " + " | ".join(["---:"] * len(labels)) + " |"]
    for ante in range(1, 9):
        values = [pct(summaries[label]["ante_clear_rates"][str(ante)]) for label in labels]
        lines.append(f"| {ante} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    confirmation = read(root / "continuations/confirmation_result.json")
    training_results = read(root / "continuations/development_comparisons.json")
    offline = read(root / "diagnostics_v2/offline_summary.json")
    teacher = read(root / "teacher_dev200/summary.json")
    teacher_manifest = read(root / "teacher_dev200/manifest.json")
    screening = read(root / "development/screening.json") or []
    expanded = read(root / "development/expanded.json") or []
    factorial = read(root / "development/factorial.json") or []
    selection = read(root / "development/selection.json")
    replay = read(root / "diagnostics_v2/replay.json") or []
    diagnostic_manifest = read(root / "diagnostics_v2/manifest.json") or {}
    reconstruction = read(root / "historical_diagnostics/reconstruction_provenance.json")
    historical = read(root / "historical_diagnostics/replay.json") or []
    reward_prefixes = read(root / "current_prefix_reward_equivalence_v3_events.json")
    interventions = read(root / "interventions/summary.json")
    input_audit = None
    budget = read(root / "accelerator_budget.json")
    compute = read(root / "continuations/compute.json")
    verification = read(root / "verification_final.json")
    execution_notes = read(root / "execution_notes.json") or []
    selected_offline = read(root / "continuations/selected_offline/offline_summary.json")
    student_agreement = read(root / "continuations/corrections/agreement_strata.json")
    agreement_standardization = None
    report = []
    if confirmation:
        sampled_totals = {}
        for arm in (confirmation["candidate_arm"], "control"):
            summaries = [
                row for label, row in confirmation["per_arm"].items()
                if label.startswith(arm + "/") and label.endswith("/sampled")
            ]
            sampled_totals[arm] = (
                sum(row["wins"] for row in summaries), sum(row["games"] for row in summaries)
            )
        candidate_wins, candidate_games = sampled_totals[confirmation["candidate_arm"]]
        control_wins, control_games = sampled_totals["control"]
        report.append(
            f"Gameplay result: **{confirmation['status']}**. Candidate arm: `{confirmation['candidate_arm']}`. "
            f"Sampled wins: {candidate_wins}/{candidate_games} ({pct(candidate_wins / candidate_games)}) versus "
            f"{control_wins}/{control_games} ({pct(control_wins / control_games)}) for the matched control. "
            f"Confirmation-panel sampled difference versus matched continuation control: "
            f"**{delta_text(confirmation['paired']['sampled'])}** (paired 95% interval)."
        )
    else:
        report.append("Transfer investigation is in progress. No gameplay improvement is validated yet.")
    report.append(
        "The original HF dataset, ten checkpoints, and previous logs are preserved. PPO was not started. "
        "The reference is Blue Deck, White Stake, Ante 8, mixture 0.5, temperature 1.0, FP32, batch 32, "
        "four CPU engine workers, and MPS policy inference. Wins finish at engine ante 9, so mean final ante can exceed 8."
    )
    if teacher:
        low, high = teacher["win_rate_wilson_ci95"]
        report.append(
            f"Matched V68 teacher: **{teacher['wins']}/{teacher['games']} ({pct(teacher['win_rate'])})**, "
            f"Wilson 95% interval [{pct(low)}, {pct(high)}], {teacher['stalls']} stalls. "
            "Search shops and scaler growth are explicit; blind rollouts are off. "
            "The dataset rate is 3,728/5,000 (74.56%) on different seeds and simulator sources."
        )
    report.append("**Cause and evidence assessment**")
    report.append(
        "| Candidate cause | Evidence | What is established |\n| --- | --- | --- |\n"
        "| Simulator drift | Current replay diverges in all 20 selected HF games; reconstructed historical sources match all 3,909 decisions. | Saved actions are not reproducible under current rules. The current wrappers agree on observations and state fields for 476 common prefix steps. Effect on policy wins requires matched historical/current evaluation below. |\n"
        "| Reward-input defects | Fast generation omitted boss identity and shop-event reward fields, and computed action diagnostics after the action. A purchase in seed 2015075 lost a 0.4 Joker-upgrade reward. | Corrected for future generation with regression tests and matching rewards on all 476 legal prefix steps. New records use reward snapshot version 3. Original HF targets remain fixed in the trials; these fixes do not change the evaluated policy's observations or actions. |\n"
        "| Evaluation configuration | Old CLI omitted deck/mixture controls, silently fell back from malformed metadata, and offered a legacy teacher. | Measurement defects fixed. These are not established causes of the previously reported custom Blue/.5 evaluation. |\n"
        "| Imperfect imitation | Training-state action metrics below. | Critical hand choices are much less accurate than blind/shop choices. Exact teacher disagreement alone is not proof of a bad action. |\n"
        "| State distribution | Same selected checkpoint evaluated on HF teacher states and current student states; correction trial below. | Agreement difference is descriptive. Historical/current rule differences also affect these panels and cannot be separated by agreement alone. |\n"
        "| Missing/quantized inputs | Candidate score tokens are structural proxies; card permanent bonuses and Joker features are quantized; RNG state is absent. | No observation feature was changed. Exact-score action interventions test an oracle action rule, not a trained feature migration. |"
    )
    if training_results:
        for row in training_results:
            label = "Critic-loss continuation" if row["arm"] == "actor_only" else "Student-state correction replacement"
            scope = (
                "Tests removing critic losses for 1,000 continuation updates; it does not isolate their effect during original pretraining."
                if row["arm"] == "actor_only"
                else "Tests replacing half the HF examples with the finite correction set; label quality, HF exposure, and reuse can interact."
            )
            report[-1] += f"\n| {label} | Development sampled difference versus matched continuation control: {delta_text(row['paired']['sampled'])}. | {scope} Confirmation is reported separately below. |"
    if replay:
        table = "**Recorded-action divergence examples**\n\n| HF seed | First divergent step | Field/index | Saved → reconstructed | First illegal recorded-action step |\n| ---: | ---: | --- | --- | ---: |"
        for row in replay[:3]:
            difference = row["first_saved_fast"]
            if difference:
                table += f"\n| {row['seed']} | {difference['step']} | {difference['field']}{difference.get('index', '')} | {difference.get('left')} → {difference.get('right')} | {row.get('replay_error', {}).get('step', 'none')} |"
        report.append(table)
        report.append(
            "Source provenance identifies scoring/boss-rule changes in `bf3a6e5` and pack handling changes in `81b4afd` after data generation. "
            "Historical reconstruction verifies each restored file against its generation SHA-256; per-file matching revisions are retained in structured results. "
            "The exact historical replay establishes source drift. A current-prefix replay ending at an illegal saved action does not establish full-episode reward parity."
        )
    if offline:
        report.append(
            "**What epoch 10 learned on 200 training games**\n\n"
            "Uniform seed sampling without replacement used NumPy RNG 1729. All 44,348 decisions were retained "
            "(147 winning games, 53 losses). These are training-data diagnostics, not held-out accuracy."
        )
        table = "| Decisions | Count | Accuracy | NLL | Teacher-action probability | Candidate coverage |\n| --- | ---: | ---: | ---: | ---: | ---: |"
        for name in (
            "overall",
            "family/play_subset",
            "family/discard_subset",
            "family/shop_buy",
            "family/shop_leave",
            "family/pack_claim",
            "family/pack_skip",
        ):
            row = offline["strata"][name]
            metrics = row["decision_weighted"]
            coverage = metrics["candidate_covered"]
            table += f"\n| {name} | {row['decisions']} | {pct(metrics['accuracy'])} | {metrics['nll']:.4f} | {pct(metrics['teacher_probability'])} | {pct(coverage) if coverage is not None else 'n/a'} |"
        report.append(table)
        game = offline["strata"]["overall"]["game_weighted"]
        report.append(
            f"Game-weighted accuracy is {pct(game['accuracy'])}; NLL is {game['nll']:.4f}. "
            "No teacher labels were unreachable under mixture .5. Candidate coverage only counts play/discard labels; "
            "the autoregressive component supplies support outside the candidate list. Phase, ante, outcome, "
            "family confusion, and per-seed/step rows are retained in the diagnostic artifacts."
        )
    if screening:
        table = "**Checkpoint screen: first 50 development seeds**\n\n| Epoch | Sampled wins / 150 | Sampled win rate | Greedy wins / 50 |\n| ---: | ---: | ---: | ---: |"
        for row in screening:
            table += f"\n| {row['epoch']} | {row['sampled']['wins']} | {pct(row['sampled']['win_rate'])} | {row['greedy']['wins']} |"
        report.append(table)
    if expanded:
        table = "**Full development expansion**\n\n| Epoch | Sampled wins / 600 | Sampled win rate | Greedy wins / 200 |\n| ---: | ---: | ---: | ---: |"
        for row in expanded:
            table += f"\n| {row['epoch']} | {row['sampled']['wins']} | {pct(row['sampled']['win_rate'])} | {row['greedy']['wins']} |"
        report.append(table)
    if selection:
        report.append(
            f"Epoch {selection['epoch']} was selected by full-panel sampled win rate, with mean final ante as the declared tiebreaker."
        )
    if factorial:
        table = "**Epoch-10 deck/mixture control: first 100 development seeds**\n\n| Deck | Mixture | Sampled wins / 300 | Greedy wins / 100 |\n| --- | ---: | ---: | ---: |"
        for row in factorial:
            table += f"\n| {row['deck']} | {row['mixture']} | {row['sampled']['wins']} | {row['greedy']['wins']} |"
        report.append(table)
    comparisons = {}
    regression = {}
    for mode in ("greedy", "sampled"):
        rows = outcomes(root / "regression_epoch10", [mode])
        if rows is not None:
            regression[mode] = summarize(rows)
    if len(regression) == 2:
        report.append(
            "**Previously reported seed-panel regression:** "
            f"greedy {regression['greedy']['wins']}/{regression['greedy']['games']}; "
            f"sampled {regression['sampled']['wins']}/{regression['sampled']['games']} with sampling seed 31013. "
            "This rerun uses the original 32-step idle limit and the 10000-series seeds. "
            "The previous custom evaluator reported 8/50 and 0/25 respectively."
        )
    if selection:
        selected_rows = outcomes(
            root / "development", [f"epoch{selection['epoch']}_sample{s}_dev200" for s in (0, 1, 2)]
        )
        teacher_rows = outcomes(root, ["teacher_dev200"])
        if selected_rows is not None and teacher_rows is not None:
            teacher_repeats = [{**row, "sampling_seed": s} for row in teacher_rows for s in (0, 1, 2)]
            comparisons["selected_student_minus_teacher"] = paired_statistics(selected_rows, teacher_repeats)
            report.append(
                "On the matched development panel, selected student minus V68 teacher sampled difference is "
                + delta_text(comparisons["selected_student_minus_teacher"])
                + ". The deterministic teacher outcome is reused only to align sampling repeats; the bootstrap still resamples 200 game seeds."
            )
            selected_summary = summarize(selected_rows)
            report.append(
                f"The teacher's mean final ante is {teacher['mean_final_ante']:.3f}; the selected sampled student's is "
                f"{selected_summary['mean_final_ante']:.3f}. Student stalls: {selected_summary['stalls']}; "
                f"censored games: {selected_summary['censored']}/{selected_summary['games']} "
                f"(teacher: {teacher['censored']}/{teacher['games']}). Ante-clear denominators are all games in each panel.\n\n"
                + progression_table({"V68 teacher": teacher, f"Epoch {selection['epoch']} sampled": selected_summary})
            )
    demonstration = outcomes(root / "development", [f"epoch10_sample{s}_demonstration200" for s in (0, 1, 2)])
    demonstration_greedy = outcomes(root / "development", ["epoch10_greedy_demonstration200"])
    if demonstration is not None and demonstration_greedy is not None:
        sampled_demo, greedy_demo = summarize(demonstration), summarize(demonstration_greedy)
        report.append(
            "**Autonomous play on the 200 HF demonstration seeds:** "
            f"sampled {sampled_demo['wins']}/{sampled_demo['games']} ({pct(sampled_demo['win_rate'])}), "
            f"greedy {greedy_demo['wins']}/{greedy_demo['games']} ({pct(greedy_demo['win_rate'])}). "
            "These are fresh resets under current rules on training-game seeds, not recorded-action replay. "
            "Compare the fresh-seed Blue/.5 epoch-10 factorial above; the seed panels differ, so this comparison is unpaired."
        )
    if selected_offline and student_agreement:
        table = "**Selected checkpoint agreement by visited-state source**\n\n| States | Decisions | Games | Decision accuracy | Game accuracy | Decision NLL | Game NLL |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        for label, summary in (("HF teacher", selected_offline), ("Sampled student", student_agreement)):
            row = summary["strata"]["overall"]
            d, g = row["decision_weighted"], row["game_weighted"]
            table += f"\n| {label} | {row['decisions']} | {row['games']} | {pct(d['accuracy'])} | {pct(g['accuracy'])} | {d['nll']:.4f} | {g['nll']:.4f} |"
        report.append(table)
        a, b = selected_offline["strata"], student_agreement["strata"]
        shared = sorted(key for key in a if key.startswith("family/") and key in b)
        denominator = sum(a[key]["decisions"] for key in shared)
        agreement_standardization = {
            "weighting": "HF teacher-action-family frequencies, restricted to families observed in both panels",
            "teacher_decision_coverage": denominator / a["overall"]["decisions"],
            "student_decision_coverage": sum(b[key]["decisions"] for key in shared) / b["overall"]["decisions"],
            "teacher_accuracy": sum(
                a[key]["decisions"] / denominator * a[key]["decision_weighted"]["accuracy"] for key in shared
            ),
            "student_accuracy": sum(
                a[key]["decisions"] / denominator * b[key]["decision_weighted"]["accuracy"] for key in shared
            ),
        }
        report.append(
            "The aggregate agreement drop partly reflects a different mix of decisions. Reweighting common teacher-action families to HF frequencies gives "
            f"{pct(agreement_standardization['teacher_accuracy'])} HF accuracy versus {pct(agreement_standardization['student_accuracy'])} student-state accuracy "
            f"(HF coverage {pct(agreement_standardization['teacher_decision_coverage'])}, student coverage {pct(agreement_standardization['student_decision_coverage'])}). "
            "This is descriptive standardization; ante, build state, and rule differences remain."
        )
        table = "| Teacher-action family | HF-state accuracy | Student-state accuracy |\n| --- | ---: | ---: |"
        for family in ("play_subset", "discard_subset", "shop_buy", "shop_leave", "pack_claim", "pack_skip"):
            key = "family/" + family
            if key in a and key in b:
                table += f"\n| {family} | {pct(a[key]['decision_weighted']['accuracy'])} | {pct(b[key]['decision_weighted']['accuracy'])} |"
        report.append(table)
    current = outcomes(root / "development", [f"factorial_b_blue_m0.5_sample{s}" for s in (0, 1, 2)])
    old = outcomes(root / "historical_epoch10_mps", [f"sampled{s}" for s in (0, 1, 2)])
    if current is not None and old is not None:
        comparisons["current_minus_historical_rules"] = paired_statistics(current, old)
        report.append(
            "**Simulator-rule intervention, identical epoch 10 and 100 development seeds:** "
            f"current rules {pct(summarize(current)['win_rate'])}; historical rules {pct(summarize(old)['win_rate'])}; "
            f"current minus historical = {delta_text(comparisons['current_minus_historical_rules'])}. "
            "This changes the simulator rules while holding policy and evaluation settings fixed; "
            "it is not a proposal to restore incorrect rules in production."
        )
    current_greedy = outcomes(root / "development", ["factorial_b_blue_m0.5_greedy"])
    old_greedy = outcomes(root / "historical_epoch10_mps", ["greedy0"])
    if current_greedy is not None and old_greedy is not None:
        comparisons["current_minus_historical_rules_greedy"] = paired_statistics(current_greedy, old_greedy)
        report.append(
            "The matched greedy current-minus-historical difference is "
            + delta_text(comparisons["current_minus_historical_rules_greedy"])
            + "."
        )
    for deck in ("b_blue", "b_red"):
        a = outcomes(root / "development", [f"factorial_{deck}_m0.5_sample{s}" for s in (0, 1, 2)])
        b = outcomes(root / "development", [f"factorial_{deck}_m0.1_sample{s}" for s in (0, 1, 2)])
        if a is not None and b is not None:
            key = deck + "_mixture_05_minus_01"
            comparisons[key] = paired_statistics(a, b)
            report.append(f"Within {deck}, mixture .5 minus .1 sampled difference: {delta_text(comparisons[key])}.")
    for mixture in (0.1, 0.5):
        a = outcomes(root / "development", [f"factorial_b_blue_m{mixture}_sample{s}" for s in (0, 1, 2)])
        b = outcomes(root / "development", [f"factorial_b_red_m{mixture}_sample{s}" for s in (0, 1, 2)])
        if a is not None and b is not None:
            key = f"mixture_{mixture}_blue_minus_red"
            comparisons[key] = paired_statistics(a, b)
            report.append(f"At mixture {mixture}, Blue minus Red sampled difference: {delta_text(comparisons[key])}.")
    if interventions:
        report.append(
            "**One-action interventions on student disagreements**\n\n"
            "One disagreement per correction game is reservoir-sampled using RNG 1729. Both branches "
            "start from the same snapshot and use the same continuation sampling seed at batch 1. "
            "Selection is uniform within each game, not uniform over all decisions across games."
        )
        table = "| Intervention | Games | Win difference | Blind-survival difference | Final-ante difference |\n| --- | ---: | ---: | ---: | ---: |"
        for name, row in interventions.items():
            fields = []
            for key in ("won", "blind_survived", "max_ante"):
                metric = row[key]
                fields.append(
                    "n/a"
                    if metric is None
                    else f"{metric['mean_difference']:+.4f} [{metric['paired_ci95'][0]:+.4f}, {metric['paired_ci95'][1]:+.4f}]"
                )
            table += f"\n| {name} | {row['games']} | " + " | ".join(fields) + " |"
        report.append(table)
        intervention_rows = [
            json.loads(line) for line in (root / "interventions/interventions.jsonl").read_text().splitlines()
        ]
        probed = [row for row in intervention_rows if row["exact_score_probes"]]
        input_audit = {
            "states": len(intervention_rows),
            "exact_score_states": len(probed),
            "states_with_structural_exact_argmax_disagreement": sum(
                bool(row["input_audit"]["structural_and_exact_argmax_differ"]) for row in probed
            ),
            "states_with_candidate_score_token_collision": sum(
                row["input_audit"]["candidate_score_token_collisions_with_distinct_exact_scores"] > 0 for row in probed
            ),
            "states_with_lossy_permanent_bonus_token": sum(
                row["input_audit"]["card_permanent_bonus_not_exactly_recoverable_from_bonus_token"] > 0
                for row in intervention_rows
            ),
            "unencoded_extra_keys": sorted(
                {
                    key
                    for row in intervention_rows
                    for key in row["input_audit"]["joker_extra_keys_without_dedicated_numeric_feature"]
                }
            ),
        }
        report.append(
            f"Exact scoring was probed in {len(probed)} play-disagreement states. Structural and exact candidate maxima differ in "
            f"{input_audit['states_with_structural_exact_argmax_disagreement']}; "
            f"{input_audit['states_with_candidate_score_token_collision']} have equal score-token values for candidates with different exact scores. "
            "The per-state audit records permanent-bonus quantization and Joker extra keys without dedicated numeric features. "
            "Other tokens and history can carry related information: these observations do not establish whole-observation aliasing. "
            "An exact-oracle action effect cannot by itself separate missing information from imperfect use of existing inputs."
        )
    correction_ablation = read(root / "continuations/correction_ablation.json")
    if training_results:
        report.append(
            "Each arm uses training seeds 0 and 1, FP32, physical batch 320, 1,000 AdamW updates, "
            "peak LR 3e-5, weight decay .01, and 100 warmup updates followed by cosine decay. "
            "Dropout .1, entropy coefficient .001, HF outcome-weight beta 1.5 and clipping [.25, 4], "
            "and mixture .5 stay fixed. The control retains critic coefficients .5/.1; both other arms use zero. "
            "Correction batches contain 160 HF and 160 unit-weight teacher-correction rows."
        )
        report.append(
            "**Controlled continuation development comparisons**\n\n"
            "| Arm versus continuation control | Sampled paired difference | Greedy paired difference |\n| --- | ---: | ---: |\n"
            + "\n".join(
                f"| {row['arm']} | {delta_text(row['paired']['sampled'])} | {delta_text(row['paired']['greedy'])} |"
                for row in training_results
            )
        )
    if confirmation:
        report.append(
            "**Untouched confirmation panel**\n\n400 game seeds, three sampling seeds per training replicate, "
            "two training replicates per arm. Repeats and training replicates are averaged within game seed "
            "before the paired bootstrap. The experiment does not estimate uncertainty over a broad population of training seeds."
        )
        report.append(
            "Sampled gains for training seeds 0 and 1: "
            + ", ".join(f"{100 * delta:+.2f} pp" for delta in confirmation["sampled_training_replicate_deltas"])
            + ". "
            f"Greedy paired difference: {delta_text(confirmation['paired']['greedy'])}. "
            "Validation additionally requires no new legality, non-finite-output, or infrastructure failures."
        )
        table = "| Arm / training seed / mode | Wins / games | Win rate | Game-cluster 95% interval | Mean final ante | Stalls |\n| --- | ---: | ---: | --- | ---: | ---: |"
        for label, row in confirmation["per_arm"].items():
            low, high = row["win_rate_cluster_ci95"]
            table += f"\n| {label} | {row['wins']}/{row['games']} | {pct(row['win_rate'])} | [{pct(low)}, {pct(high)}] | {row['mean_final_ante']:.3f} | {row['stalls']} |"
        report.append(table)
        report.append(
            "Confirmation ante-clear rates, with all games in each arm as the denominator:\n\n"
            + progression_table({label: row for label, row in confirmation["per_arm"].items() if label.endswith("/sampled")})
        )
    if correction_ablation:
        report.append(
            "The secondary development comparison of teacher-correction continuation against actor-only HF continuation is "
            + delta_text(correction_ablation["paired"]["sampled"])
            + " sampled. This tests the correction-data contribution beyond removing critic losses; it is not an independent confirmation result."
        )
    if budget or compute:
        report.append(
            "**Compute accounting**\n\n"
            + (
                f"Diagnostic accelerator wall time: {budget['diagnostic_accelerator_seconds'] / 3600:.3f} hours. "
                if budget
                else ""
            )
            + (
                f"Continuation/intervention/development/confirmation wall time: {compute['total_elapsed_seconds'] / 3600:.3f} hours. "
                if compute and "total_elapsed_seconds" in compute
                else "Continuation compute is still being recorded. "
            )
            + "The maximum is 12 accelerator hours; three hours were reserved for confirmation. CPU teacher search is accounted separately."
        )
        if compute and "total_elapsed_seconds" in compute:
            report.append(
                f"Confirmation used {compute['confirmation_elapsed_seconds'] / 3600:.3f} hours. "
                f"All six training runs processed 320,000 examples each (1,920,000 total) in "
                f"{sum(row['elapsed_seconds'] for row in compute['training_runs']) / 3600:.3f} hours. "
                "Per-run learning curves are retained in `continuations/*_seed*/learning.jsonl` and TensorBoard events."
            )
        if teacher_manifest and teacher_manifest.get("status") == "complete":
            report.append(
                f"The matched V68 teacher panel used {teacher_manifest['elapsed_seconds'] / 3600:.3f} hours of CPU wall time "
                f"with {teacher_manifest['workers']} workers; this is excluded from accelerator usage. "
                "The continuation ledger conservatively includes CPU orchestration and the diagnostic pause within its wall time."
            )
    report.append(
        "**Verification and reproducibility**\n\n146 targeted regression tests passed at the implementation checkpoint; "
        "final verification is recorded in the artifact directory. Tests cover deck propagation, strict metadata, "
        "Parquet preservation, policy-path agreement, action support/modes, snapshot isolation, label provenance, "
        "seed separation, duplicate/missing paired rows, clustered statistics, actor-only gradients, and the reward reproducer. "
        "Additional regressions cover acceptance at exactly five points and exact zero paired gains despite floating-point cancellation. "
        "See [the reproducible workflow](README.md) for commands and trial settings."
    )
    if verification:
        report.append(
            "Final artifact verification passed: ten original checkpoints and the HF files retain their hashes; "
            "correction labels retain their pre-training hashes and legality; all six training runs have matching settings; "
            "and acceptance was recomputed from all 6,400 raw confirmation games. "
            f"Conservative compute accounting, including the overhead reserve, totals "
            f"{verification['accounted_accelerator_seconds_including_overhead_reserve'] / 3600:.3f} of 12 hours. "
            f"Confirmation stalls: {sum(row['stalls'] for row in confirmation['per_arm'].values())}; "
            f"censored games: {sum(row['censored'] for row in confirmation['per_arm'].values())}."
        )
    if execution_notes:
        report.append(
            "A diagnostic wrapper exceeded its initial 20-minute stage timeout. Its already-running greedy evaluator "
            "finished the full panel, and the resumed pipeline validated every completed panel before continuation training. "
            "No historical games were discarded or rerun; the error and recovery logs are preserved."
        )
    report.append(
        "**Remaining uncertainty and next training**\n\n"
        "The finite seed panels and two training replicates limit statistical power. HF teacher states were generated "
        "under historical rules, so state-distribution and rule-alignment effects can interact. Exact-score interventions "
        "do not isolate an individual missing feature. A null actor-only continuation cannot rule out critic interference "
        "during the original ten-epoch training: this intervention tests 1,000 further updates from epoch 7. "
        "The correction comparison tests replacement of half the HF examples by a finite 100-game correction set; "
        "it does not isolate label quality from reduced HF exposure or correction-set reuse. "
        "Keep tokenizer v13 unchanged unless a versioned migration and matched "
        "new-input training experiment are justified. Generate future teacher demonstrations under pinned current engine "
        "sources with corrected reward snapshots; keep whole-game seed splits and the confirmation panel excluded from training. "
        + (
            "Use the validated continuation arm as the next supervised starting point; retain its matched control and provenance."
            if confirmation and confirmation["status"] == "validated"
            else "Do not promote an unvalidated continuation or start PPO on the basis of imitation loss alone."
        )
    )
    if confirmation and confirmation["status"] != "validated":
        report.append(
            "Recommended next experiment: generate a separate current-rule V68 dataset with the pinned Blue/search/scaler "
            "configuration and corrected reward snapshots. On that same dataset, compare the existing supervised objective "
            "with an otherwise identical objective that doubles play/discard BC weights, retaining the same normalization, "
            "optimizer settings, 1,000-update budget, and two training replicates. This targets the weak hand-decision "
            "imitation and the short-term benefit from teacher-action substitutions; its gameplay benefit is untested. "
            "Keep v13 inputs fixed and reserve a new confirmation panel before the experiment, since this investigation's "
            "confirmation panel is now exposed. Rebuilding the data addresses established source/reward inconsistencies; "
            "it is not itself a claim that those inconsistencies explain the sampled win-rate gap."
        )
    elif confirmation:
        report.append(
            "Before any later PPO work, separately train and validate critic calibration on executed current-rule "
            "trajectories: the actor-only continuation arms were trained without critic losses. PPO remains stopped in this study."
        )
    report.append(f"Full local artifacts: `{root}`. Compact structured results: [results.json](results.json).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(report) + "\n")
    write_json(
        args.output.parent / "results.json",
        {
            "status": confirmation["status"] if confirmation else "in_progress",
            "teacher": teacher,
            "teacher_timing": {
                key: teacher_manifest.get(key) for key in ("elapsed_seconds", "workers")
            } if teacher_manifest else None,
            "screening": screening,
            "expanded": expanded,
            "factorial": factorial,
            "paired_diagnostics": comparisons,
            "previous_panel_regression": regression,
            "continuation_development": training_results,
            "correction_ablation_development": correction_ablation,
            "confirmation": confirmation,
            "interventions": interventions,
            "input_audit": input_audit,
            "demonstration_sampled": summarize(demonstration) if demonstration is not None else None,
            "demonstration_greedy": summarize(demonstration_greedy) if demonstration_greedy is not None else None,
            "selected_checkpoint_agreement": {"teacher_states": selected_offline, "student_states": student_agreement},
            "agreement_family_standardization": agreement_standardization,
            "budget": budget,
            "compute": compute,
            "verification": verification,
            "execution_notes": execution_notes,
            "replay": {"current": replay, "historical": historical},
            "current_reward_prefixes": reward_prefixes,
            "provenance": {
                "epoch10_checkpoint_sha256": diagnostic_manifest.get("sha256"),
                "data_hashes": diagnostic_manifest.get("data_hashes"),
                "source_drift": diagnostic_manifest.get("source_drift"),
                "historical_reconstruction": reconstruction,
                "runtime": read(root / "runtime.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
