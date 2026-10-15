# Source adaptation rehearsal

This engineering check compares standard raw-feature adaptation with SemFlow
from the original common Base checkpoint. Both arms update only shared adapters
for 60 steps, with the same source windows, optimizer and paired action noise.
Each step includes measured source action retention. Semantic weighting is
calibrated from six source-training gradient pairs before updates.

Twelve adaptation, twelve retention and twelve validation windows come from
36 distinct episodes across the six source components. Future raw and semantic
targets are cached at the common checkpoint, including its task conditioning.
Validation never participates in updates or weight calibration.

The cloud stages export and verify each checkpoint before fresh-process restore
and paired frozen/adapted validation. Restore checks optimizer state, adapter
weights, random state and a reference loss. Cross-arm gates require matching
initial weights, targets, calibration, plan and frozen validation results.

This run has no target recordings or robot rollouts. Its loss measurements do
not establish controller success or human-to-robot transfer. Target adaptation
settings and the original controller-success gate remain separate decisions.
