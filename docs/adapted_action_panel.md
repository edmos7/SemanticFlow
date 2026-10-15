# Sampled source actions after adaptation

The frozen Base step600 policy and both run06 adapted checkpoints are evaluated
on the same twelve source-validation windows used in the rehearsal. Each window
uses four fixed noise seeds at 5, 10 and 20 Euler integration steps. No tuning or
integration-step selection is performed on these measurements.

The panel retains all sampled predictions and scores full32 and first8 actions
against demonstrations and the source-training mean baseline. It reports
per-dimension error, controller-coordinate gripper sign accuracy, clipping and
the decomposition of sampled error into empirical seed-mean error and variance.
An empirical seed average is a diagnostic statistic, not a deployed controller.

Every arm loads the same common checkpoint and native processing. Adapted arms
restore only the hash-verified shared adapters from the completed rehearsal.
Source reports and export receipts bind these tensors to the same initialization
and training contract. Inputs, processed layouts, targets and noise hashes must
match across arms. Parameters, buffers and global RNG must remain unchanged.

This evaluation performs no optimizer updates or simulator trials. Offline
sampled-action error can reveal controller defects but does not establish the
protocol's source-success gate or human-video transfer.
