# Selection disagreement and critical prediction errors both remain

The [completed action-coverage fit](consequence_action_coverage_results_20260928.md) improves consequence prediction on fixed development alternatives but fails retention. Its [saved-data follow-up](consequence_action_coverage_residual_20260928.md) separates two remaining problems: the preference head sometimes overrides consequence estimates that favor the base, and the consequence estimates themselves still misrank critical actions. Neither better average prediction nor simply using the current consequence heads establishes a safe policy.

Among the 262 actions that lose a base clear, the new scorer predicts lower clear probability than base in **149 cases**, and lower score in **165**. Among the 90 gained clears, it also predicts lower clear probability in **19**, and lower score in **23**. These counts show disagreement between preference and consequence outputs, not the result of a tested confidence gate. Replacing preference with predicted clear probability would change both harmful and useful decisions and would still omit longer-term resource and growth value.

There are **five selected immediate losses where the base survives**, all on the final hand. Four are outside the state's original proposal set. Only one has predicted score and clear probability favoring the surviving base; in the other four, both consequence outputs favor the failing action. The prediction reference here is always base, including when the report separately counts ten selected losses where the teacher survives.

| Development state, seed:step | Original proposal set | Base / selected actual score | Base / selected predicted clear probability |
| --- | --- | ---: | ---: |
| 850000034:40 | Outside | 504 / 70 | 10.78% / 14.99% |
| 850000040:133 | Outside | 6,188 / 132 | 34.12% / 48.87% |
| 850000060:84 | Outside | 2,576 / 690 | 74.72% / 90.63% |
| 850000064:138 | Outside | 13,296 / 5,002 | 12.27% / 11.34% |
| 850000099:27 | Inside | 2,052 / 760 | 46.83% / 64.37% |

The persistent state 850000064:138 illustrates the output disagreement directly: the selected action has lower predicted score and clear probability, but its preference logit is −1.197 versus the base's −3.344. The realized base Pair clears with 13,296; the selected High Card scores 5,002 and loses. In state 850000060:84, by contrast, predicted clear probability is 90.63% for the failing action versus 74.72% for the surviving base. These are original-RNG realizations, not independently estimated conditional success probabilities or proof that every observed reversal is avoidable from visible inputs.

The net clear deficit remains concentrated outside the original proposal distribution:

| New scorer's choice group | States | Base clears | Selected clears | Net difference |
| --- | ---: | ---: | ---: | ---: |
| Inside original proposals | 4,936 | 2,233 | 2,224 | −9 |
| Outside original proposals | 1,072 | 466 | 303 | −163 |
| All eligible | 6,008 | 2,699 | 2,527 | −172 |

These groups are defined by the new scorer and differ from the control's groups. They cannot identify the causal effect of additional labels on a fixed subgroup; that evidence comes from the fixed-control-action prediction comparison in the main report. On its new outside choices, selected-action MSE is 1.467351 versus a same-state original-proposal average of 0.920893, and BCE is 0.411660 versus 0.357342. The selected alternatives remain harder to predict.

Restricting selection to the teacher-containing original proposal sets would yield 2,680 clears, 39 immediate losses and six losses where the teacher survives, versus base 2,699/40/7. This restricted rule uses teacher information and is not deployable; it still fails the overall clear gate. No new policy, gate or candidate is selected from the diagnosis.

The next architecture comparison can now ask a narrower question: with the expanded labels held fixed, can an explicit relational readout improve critical consequence rankings and retention beyond this pooled head? The earlier relational comparison used the original labels, so it does not answer that question. This remains a hypothesis; do not combine it with new state diversity, independently learned features, duration selection or a new selection rule. Continue to require the original prediction and retention gates before integration, and reserve independent confirmation for a final candidate. A feature-path change is not yet established as necessary.

The standard-library-only analysis reconciled all 6,113 state identities, 6,008 eligible choices, exact gained/lost and new-loss sets, all audited retention groups, and original-proposal prediction metrics. Maximum discrepancy in the previously audited metrics was 2.41e−10. It also saves every derived eligible-state row and all fatal-case predictions/outcomes. These checks reproduce prior independently audited quantities; no second independent implementation of every new statistic is claimed.

Owner session 48389 actually exited zero in `9e637c`, after a 2.304-second analysis; admission was checked and written in `88f342`. There was no new inference, training, simulator execution, autonomous evaluation or confirmation.

- [Frozen protocol](../../eval/consequence_prediction_20260927/action_coverage_residual_protocol_v1.json), SHA256 `46ac223d0bd433728f2e218993efaaab928ea7406b2a67594330bd7e458a57b8`.
- [Complete diagnosis](../../eval/consequence_prediction_20260927/action_coverage_residual_v1/result_v1.json), SHA256 `dc9e843d8851dea3cee307f13095e5ab8d9d085bc21d9e8c14be7e3da45d2757`.
- [Actual owner-exit receipt](../../eval/consequence_prediction_20260927/action_coverage_residual_owner_terminal_v1.json), SHA256 `5dae721d52acb122e81724daa9b668ae5fa260e6df07ccee414fd32d99359670`.

Long-horizon generalization and fresh-win improvement remain unproven. The failed fixed endpoint and its decision remain unchanged.
