# Accept the two admitted training-label schemas

The first inventory owner actually exited one in tool chunk `fa7261`, raising `KeyError: game_won` on the first training state before writing any eligible-state row or selecting a pilot. Preserve that helper, protocol, empty partial output, process log and failed owner. No model forward, optimizer update, simulation or new continuation outcome occurred.

The original admitted training labels contain 231,590 outcomes with an explicit `game_won` field and 14,047 audited historical outcomes without it. All 47,125 added outcomes have the field. Both schemas retain `phase`, `game_over` and `blind_cleared`. The frozen runtime's `GamePhase` encodes a terminal win as `game_won` and a terminal loss as `game_over`.

The separate v2 inventory derives terminal status from `phase`. Require `game_over` to match the loss phase, `blind_cleared` to match `shop` or `game_won`, and, wherever present, the Boolean `game_won` to agree with the winning phase. Never treat a missing win flag as a loss or as a nonterminal action. Check both schemas over all admitted labels before execution. Preserve original outcome dictionaries in the selected pilot so later replay can match every field that was originally present.

The original [inventory plan](consequence_continuation_training_inventory_20260928.md), three-action rule, four groups, first-64-game pilot eligibility, eight distinct source games, hash ordering, budgets and all scientific requirements are unchanged. This is an input-compatibility correction, not a changed sample or a scientific retry. The v2 outputs and owner are separate, and retain the v1 failure as a prerequisite.
