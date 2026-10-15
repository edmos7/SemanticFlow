# Sampler clock audit

The production sampler computes each flow time independently as step/steps.
Pinned native inference repeatedly adds a float32 step size. These clocks match
at5steps; at10steps they first differ at iteration7. Native sinusoidal time
features use periods as short as0.004, amplifying time differences. This is a
numerical hypothesis explaining the failed GPU parity audit, not a demonstrated
cause of poor controller performance.

Keep production sampling unchanged. On six existing source-validation windows,
one per source task, use paired seed4242 at5/10/20steps. Compare four paths:
indexed production, accumulated clock with the original division update,
accumulated clock with the native multiply update, and actual native inference
through the same Panda maps and conditioner. Record input times, time embedding,
noisy actions and velocities at each iteration. Preserve failed rows before
raising. Matched-clock/native outputs must agree within1e-4; production outputs
must reproduce saved source-panel samples within1e-5.

Report action-error changes and numerical sensitivity across all declared
conditions. No optimizer update, human-video input, target rollout or controller
success claim. A corrected time grid warrants continuation of the blocked
integrity audit only after native GPU parity is confirmed.
