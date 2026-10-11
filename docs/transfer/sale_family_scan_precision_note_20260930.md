# Native scan precision clarification

The fourth paragraph of [the prepared scan plan](sale_family_scan_20260930.md) describes the native NLL and logit tolerance together as 2e-5. The inherited implementation actually retains the original, stricter **1e-5 native NLL tolerance**; raw arithmetic and logits use **2e-5**. This was verified against the admitted parent protocol before preparing or launching the new scan.

No source, fitting recipe, candidate, gate or tolerance has been changed. The future scan protocol must preserve these three original values exactly. The earlier plan remains preserved; this note corrects its description.
