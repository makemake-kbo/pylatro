# Training-pair check for the fixed-base scorer

After the single fit and optimizer/checkpoint audit finish, evaluate the candidate on the same 430 mixed-survival training states and all 3,396 labeled survivor/fatal pairs. Reuse the previous survival-ranking endpoint's admitted seven-slot predictions instead of forwarding the control again. Require identical active targets and exact reproduction of its saved pair summaries; reconcile its original proposal overlap against its full training prediction artifact.

The candidate uses the fixed base conditional probability plus its preference correction around the immutable initial readout. Save raw preference, initial preference and base terms, then independently verify their composition and packed-action alignment. Compare score, clear and preference ordering for all pairs, early/later states, original/extra actions and teacher/nonteacher winners, with the existing state-balanced pair weighting. Reuse the same original-RNG outcomes; no new labels or simulation.

Keep the existing 900-second measurement and audit caps plus 30 seconds supervision grace, CPU FP32, one thread, batches of 128 and prediction overlap tolerances. This is an in-sample mechanism diagnostic, not a policy gate or a win-rate result. Held-out prediction, selected-action retention and later autonomous/confirmation standards remain unchanged.
