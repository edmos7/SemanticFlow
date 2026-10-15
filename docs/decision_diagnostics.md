# One diagnostic decision cycle

Evaluate the existing Base600 and human-adapted checkpoints, with one60-update
source-action-retention-only control. Freeze this plan before new diagnostic
scores. Prior adaptation results are already known; these are engineering
thresholds, not a preregistered statistical test.

1. Verify complete checkpoint delta equality. Recompute training-only state
   and action statistics directly from the pinned source files using NumPy;
   check raw action alignment, normalization, inverse conversion and clipping.
2. On six fixed source-validation windows, compare velocity calls and5/10/20
   integration steps with actual pinned native flow/inference code. Retain the
   trained Panda maps and conditioner at native module boundaries, the same
   observed prefix, float32 initial noise and synchronous execution. This is
   native calling/solver parity, not a claim that the bimanual pretrained model
   already supplies a calibrated Panda controller. Numerical tolerance1e-4. Match native's accumulated float32 clock for
   the parity comparison; record the indexed-clock difference separately.
   A separate GPU audit confirmed this clock-only explanation on all six
   source components. All downstream sampling keeps the historical indexed
   clock to reproduce saved comparison samples.
3. On the12fixed source-validation windows and four paired seeds at10steps,
   evaluate predicted, zero, static and shuffled forecasts plus bypass of the
   entire conditioner. Run these on Base and each of the three SemFlow models.
   Reproduce saved ordinary samples within1e-5 before interpreting interventions.
   Zero and bypass can be off-distribution; static and shuffle are the primary
   usefulness comparisons. No future labels enter the prediction path.
4. From Base, train shared adapters for60updates using only the identical
   source-retention schedule, noise, optimizer and clipping. Export before
   fresh-process adapter/optimizer/RNG/loss recovery. Sample the same12windows,
   four seeds and5/10/20steps. Compare to all six saved human-adapted panels.
   One retention control suffices because those six runs share this schedule.

The primary endpoint is pooled normalized first8-action MSE at10steps. Keep
full32, all source components, seeds, clipping and5/20step retention sensitivity.
Useful forecast conditioning requires predicted forecasts to improve error by
at least1% over both static and shuffled forecasts, each with improvements on
at least4of6source tasks and paired action RMS change at least0.001.
Incremental human benefit requires mean SemFlow improvement of at least1% over
retention-only, with improvement for at least2of3human-adapted models.

An implementation/parity failure stops execution and calls for repair before
interpreting adaptation. With passed integrity checks, either useful forecast
conditioning or incremental human benefit justifies a limited source-controller
repair effort. If neither passes, pause this implementation. Large action
changes alone do not meet the usefulness criterion. No outcome establishes
robot transfer; source controller readiness remains unmet. No target rollout
selection, additional human data or parameter sweep enters this run.
