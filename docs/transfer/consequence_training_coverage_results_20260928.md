# The scorer selects unlabelled consequences during training too

The complete training-support measurement finds **5,932 of 46,427 eligible policy choices (12.78%)** outside the original consequence-label proposals. Across all 47,149 teacher-play states, including base non-play states, the best legal play is outside the proposals in **6,084 (12.90%)**. This is not solely a deployment or development-state issue: the preference objective includes every legal play, but consequence supervision leaves some of the model's own highest-ranked alternatives unlabelled.

The fixed candidate rule—highest-preference legal hand outside each state's existing proposals—produces **47,125 additional actions**. The other 24 states have no remaining legal action. Candidate rank has median 2, mean 2.098 and range 1–6. It would expand the 245,637 existing labels to **292,762**, a **19.18%** increase, without adding training games or state contexts.

| Training group | States | Best legal play outside proposals | Additional-label candidates |
|---|---:|---:|---:|
| All | 47,149 | 6,084 | 47,125 |
| Early, Ante below 4 | 16,326 | 1,995 | 16,324 |
| Later | 30,823 | 4,089 | 30,801 |
| Base equals teacher | 35,430 | 3,356 | 35,411 |
| Base differs from teacher, including non-play base actions | 11,719 | 2,728 | 11,714 |
| Last hand | 904 | 133 | 898 |

The scorer's best legal play matches teacher in 35,134 states; mean full-action teacher NLL is 0.953279 and mean teacher rank is 2.955. These are training-fitting measurements, not a transfer or win-rate improvement. No outcome was inferred for an unlabelled action. The expansion does not increase the number of last-hand state contexts, so sparse decisive-context coverage remains a separate question.

The [predeclared measurement](consequence_training_coverage_20260928.md) used the original fixed epoch-20 full-support checkpoint, not the development-selected prefix. All 12,249,244 legal training state/action outputs were saved as packed FP32 predictions. Measurement took 45.961 seconds; the independent saved-data audit took 14.374 seconds. It checked every legal mask, membership, offset, state identity, original-proposal overlap, chosen action, outside candidate, selected/retained slot list and summary. Existing training proposal metrics reproduce within a maximum absolute difference of 1.37e−10. Model tensors remained unchanged, with no gradients or optimizer steps.

The [coverage audit](../../eval/consequence_prediction_20260927/training_coverage_v1/verification_v1.json) and [actual owner exit receipt](../../eval/consequence_prediction_20260927/training_coverage_owner_terminal_v1.json) bind the result. Both phase workers and owner 60303 actually exited zero before admission. No new simulation, training, autonomous games or confirmation ran in this measurement. The existing fit's final validation failure and subsequent saved-artifact recovery remain in its admission chain.

This supports a concrete training-only action-label expansion. Collection must reproduce factual trajectories and isolate every extra branch; a later single fit must keep the architecture, features, full-support preference objective, normalization and state exposure fixed. An improvement from that intervention remains to be demonstrated, and the full long-horizon goal remains unproven.
