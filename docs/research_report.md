# Research question and decision chain

## Question

Could human video teach reusable world knowledge that an action model can use for downstream adaptation? SemFlow tests a small version of that idea: adapt future scene representations from human RGB and task language, then measure their effect on generated robot actions. There is no human-to-robot action retargeting.

The operational test uses a pretrained EgoSteer model, frozen DINOv3 features, 16 task-conditioned scene tokens, and future horizons of 0.5, 1, and 2 seconds. Robot source demonstrations calibrate new Panda state/action maps and a scene conditioner. This is not robot-data-free, and no claim is made that upstream pretraining lacked human action labels. Human adaptation itself uses RGB and ordinary instructions only, without human states/actions, manual masks, or event labels. No target robot demonstrations enter adaptation.

## What we tested

The common initialization is source-calibrated step 600. Six component tasks supply 248 training and 52 validation episodes. Three personally recorded chest-level clips show cups placed on plates, a jar and carton placed in a box, and a bowl placed into a drawer followed by closing it.

Standard adaptation uses raw future-feature loss and source action retention. SemFlow adds semantic state and transition losses, with targets cached at the common initialization. Each task and method resets the checkpoint and trains the rank-8 shared adapter for 60 steps, seed 42, batch size 1, learning rate 5e-5. A separate 60-step retention-only run controls for continued source training.

The primary diagnostic is normalized first-eight-action MSE with ten sampling steps. Full 32-action chunks and five/twenty-step sampling are sensitivity checks. Twelve distinct source-validation windows, two per task, use four paired action-noise seeds. These are not independent adaptation seeds or a target success metric.

## Results and decisions

1. **Source control remained weak.** An earlier simulation evaluation completed 17 source trials, with zero successes, before timing out. It was not a completed six-task rollout matrix.
2. **Human adaptation executed and recovered.** Both methods completed 180 updates across three clips. Fresh-process checkpoint recoveries passed. Human forecast measurements use overlapping windows from the training clips and measure fit, not generalization.
3. **Human adaptation did not improve sampled source actions.** All six adapted models worsened error relative to the frozen initialization across the retained action horizons and sampling-step checks. The source-mean baseline also outperformed the frozen model.
4. **Continued source training explains almost all relative gains.** SemFlow improved primary error over retention-only by 0.0423%, 0.0126%, and 0.0194% for box, drawer, and cups respectively: 0.0248% on average.
5. **Predictions affected actions without demonstrating useful control.** Predicted futures improved error over static futures by about 0.85–0.90%, and over shuffled futures by 0.12–0.16%. Whole-conditioner bypass improved frozen error by about 17%, still behind the mean baseline. Bypass removes current and future scene injection and can be outside training conditions.
6. **A sampler-clock mismatch was real but small.** Accumulated float32 time reproduced upstream sampling on the audited subset. Historical indexed time was retained for paired comparisons. Average action-error changes were approximately +0.019% at ten steps and −0.016% at twenty steps on six windows and one noise seed. This does not explain the poor controller.
7. **Pause this implementation.** The final diagnostic rules required at least 1% gain over both static and shuffled futures with four of six task wins, or at least 1% mean human increment over retention-only with two of three models improving. The gain thresholds failed. These engineering rules were fixed before the final diagnostic cycle, after earlier results were known; they were not a statistical preregistration.

## Interpretation

The scene interface is a plausible bottleneck given the bypass result, but these experiments do not identify a single cause. Weak source control, new embodiment mappings, limited adaptation, and the semantic proxy confound transfer. No successful human-adapted target rollout or hardware result was established.

A continuation should first demonstrate a source controller that benefits from scene conditioning, then isolate forecast usefulness before repeating human adaptation. These results justify pausing this interface and training setup, not rejecting the broader world-knowledge hypothesis.

See [reproducibility](reproducibility.md) and [constraints](project_constraints.md).
