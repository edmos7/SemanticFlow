# Sampled source actions after human adaptation

Compare the common Base step600 checkpoint with each of the six independently
human-adapted models: standard and SemFlow for three clips. Evaluate all models
on the same12 held-out source windows across six component tasks, four paired
noise seeds and5/10/20 integration steps. This produces144 action chunks per
model,1008 total, with no optimizer updates or rollouts.

Reuse the original action panel's full32 and first8 normalized errors,
source-training mean baseline, dimension errors, gripper sign accuracy,
clipping, seed decomposition and synchronized sampling latency. Store samples
and require identical inputs, prompt layouts and initial noise; verify that
sampling leaves parameters, buffers and global RNG unchanged.

Load each task's complete adapters from the hash-verified human training export
with a matching successful fresh-process recovery receipt. Reset to the common
initialization between task models. Compare both methods separately for each
human adaptation; retain all declared integration counts without selection.

This measures source-action retention after human adaptation. It does not
measure success on the composed target tasks. Prior source controller readiness
remains unmet, so forecast-fit improvements alone do not establish transfer.
