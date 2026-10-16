# SemanticFlow (SemFlow)

Can a world-action model (WAM) learn useful scene changes from a single human video, then use that knowledge to guide robot actions?

SemFlow explores this link between predicting the world and acting in it. We extend [EgoSteer](https://github.com/egosteer/egosteer), a pretrained world-model-enhanced action policy, so its predicted future scene features are explicitly available when generating actions.

## TL;DR

- **The idea:** a human and a robot move differently, but they can cause similar changes: a cup ends up on a plate, an object goes into a drawer, or the drawer closes. We asked whether learning to predict those changes from human video could help a robot act.
- **The model:** we started with EgoSteer-3B-Base. Its world expert predicts future DINOv3 visual features; our interface turns them into compact scene tokens and feeds them to the action expert. Robot demonstrations from six LIBERO tasks trained this interface and the Panda state/action mappings. This starting point still had weak robot control.
- **The training:** we recorded three first-person RGB videos, one per intended task, and adapted a small part of the model using video and task instructions. The human clips supplied no action labels. No robot demonstrations of the intended target tasks were used for adaptation.
- **The comparison:** ordinary adaptation learned to predict future DINOv3 features. SemFlow added losses on the scene tokens and their changes over time, treating them as a possible proxy for task meaning. Both also rehearsed source robot actions. We compared them with the unchanged model and with source rehearsal alone, first at 60 updates and then at 300.
- **What we saw:** at 300 updates, SemFlow's error against recorded source robot actions was about 3.7% higher than the unchanged model's, and very close to both adaptation controls. Earlier simulator attempts, one per target for each of the frozen and SemFlow models, completed none of the three tasks. We stopped at 300 under the experiment's continuation rules.
- **What we can conclude:** we have not shown a benefit from this setup. Action error is only a diagnostic: matching recorded actions more closely does not necessarily mean completing tasks more often. With weak starting control, one human clip per task, one training seed, and no completed multi-seed target benchmark, we cannot reliably judge whether SemFlow improves robot task performance. The broader research question remains open.

**AI assistance:** This is a solo project by Eduardo Mosca, developed with assistance from OpenAI Codex. Uses of “we” throughout the repository refer to the author working with that AI assistance, rather than a team of human researchers. The author is responsible for the work and its claims.

## Design choices: why these predictions might help

[DINOv3](https://github.com/facebookresearch/dinov3) learns visual features that support tasks such as semantic segmentation and object correspondence. That motivates our working assumption: its features may describe useful aspects of objects and scenes beyond their exact pixels. If a human and a robot cause similar scene changes, predicting those changes might offer a way to learn from human video without translating human motion into robot commands.

We compress the visual features into 16 scene tokens, guided by the task instruction. SemFlow trains both the predicted tokens and their changes across 0.5, 1, and 2 seconds. Here, “semantic flow” means change in this learned representation, not optical flow. We have not established that the tokens isolate task meaning: camera motion, appearance, and occlusion can also change them.

- **Predict features rather than pixels:** this builds on EgoSteer's existing world expert and avoids training a video generator. The tradeoff is that the model does not directly produce viewable future video.
- **Make forecasts available during control:** a learned interface lets the action expert attend to predicted scene tokens. These forecasts use observed inputs; they do not simulate candidate action plans.
- **Keep the comparison controlled:** all methods use the same starting model, adapter capacity, and forecast horizons. The feature teacher and target scene readout stay fixed during human adaptation, so the training targets do not move with the model. Source-action rehearsal is intended to limit forgetting; the retention-only control tests what rehearsal alone explains.

**What worked:** training and checkpoint recovery completed, and diagnostic interventions showed that changing the predicted futures changed generated actions. This establishes a working connection between forecasting and action generation, not useful robot control.

**What did not work, or remains unclear:** SemFlow did not improve the measured source-action agreement over the controls, and the limited simulator attempts showed no task successes. Removing the whole scene interface lowered action error, which makes the interface worth investigating, but does not identify the cause. We still lack evidence that the proposed semantic representation helps task performance or transfers from human to robot scenes.

## Human demonstrations and intended robot tasks

Each row pairs one chest-level human video with examples of the two robot skills needed for the intended task. The robot previews are **recorded LIBERO-90 dataset demonstrations**; they show the component skills, not our policy performing the combined task.

| Human input used for adaptation | LIBERO component 1 — dataset example | LIBERO component 2 — dataset example |
|---|---|---|
| **Jar + carton → box**<br>![Human box demonstration](assets/demos/human-basket_two_foods.gif) | **Alphabet soup → basket**<br>![LIBERO soup component](assets/demos/libero-basket_two_foods-component-1.gif) | **Tomato sauce → basket**<br>![LIBERO sauce component](assets/demos/libero-basket_two_foods-component-2.gif) |
| **Cream/grey cups → left/right plates**<br>![Human cup demonstration](assets/demos/human-mugs_two_plates.gif) | **White mug → left plate**<br>![LIBERO left plate component](assets/demos/libero-mugs_two_plates-component-1.gif) | **Yellow/white mug → right plate**<br>![LIBERO right plate component](assets/demos/libero-mugs_two_plates-component-2.gif) |
| **White bowl → drawer; close it**<br>![Human drawer demonstration](assets/demos/human-bowl_drawer_close.gif) | **Black bowl → bottom drawer**<br>![LIBERO bowl component](assets/demos/libero-bowl_drawer_close-component-1.gif) | **Close bottom drawer**<br>![LIBERO closing component](assets/demos/libero-bowl_drawer_close-component-2.gif) |

The intended LIBERO-10 tasks combine each row's two skills. See [task names, full human videos, and provenance](docs/demo_showcase.md). We adapted a separate model for each human clip. The action-error comparison below tests those models on the original source robot tasks; the earlier simulator attempts are linked separately.

## Results: action error and simulator attempts

The table asks a narrow question: after adaptation, how closely do generated actions match recorded actions on held-out examples from the six source tasks? All models start from the same checkpoint and use the same evaluation examples and random noise. Each row names the human clip used for adaptation, not the robot task being tested. “Retention only” means continued source robot training without human-video adaptation.

| Human adaptation mapping | Frozen | Standard 300 | SemFlow 300 | Retention only 300 | Change in error vs frozen |
|---|---:|---:|---:|---:|---:|
| Box / basket | 1.552688 | 1.610968 | 1.610588 | 1.609951 | +3.73% |
| Bowl / drawer | 1.552688 | 1.610508 | 1.609491 | 1.609951 | +3.66% |
| Cups / plates | 1.552688 | 1.609731 | 1.610291 | 1.609951 | +3.71% |

The numbers are normalized action MSE (lower means closer to the recorded actions), measured over the first eight actions and all seven action dimensions. We used 12 held-out windows, four paired noise seeds, and ten sampling steps. These noise seeds do not represent independent training runs. The shared frozen and retention-only controls are repeated for comparison.

SemFlow did not meet the experiment's thresholds for continuing to 600 updates, based on action error and forecast fit. That was a rule for deciding whether to spend more compute, not a test of robot task success. The bundled reproduction command currently covers the 60-update results; the 300-update evaluation exports are not yet included.

The earlier simulator comparisons are available here: [basket](assets/rollouts/paired/basket_two_foods-comparison.mp4), [drawer](assets/rollouts/paired/bowl_drawer_close-comparison.mp4), and [mugs](assets/rollouts/paired/mugs_two_plates-comparison.mp4). Each video compares the frozen model with the earlier SemFlow model. Neither completed the task within 400 steps. These were single attempts, not an evaluation of the 300-update models or enough trials to estimate success rates. See [rollout documentation](docs/target_rollouts.md) and [media provenance](assets/rollouts/paired/provenance.json).

<details>
<summary>Earlier 60-update results and diagnostic checks</summary>

Only a shared rank-8 adapter (about 1.6 million parameters) was trained. Source robot data contained 248 training episodes and 52 validation episodes across six tasks.

| Offline action check at 60 updates | Normalized action MSE ↓ |
|---|---:|
| Always predict the source training-mean action | 1.036449 |
| Frozen starting model | 1.552688 |
| Source rehearsal alone | 1.570889 |
| SemFlow after each human clip | 1.570225–1.570691 |

![Action adaptation and scene-conditioning diagnostics](assets/figures/action_diagnostics.png)

Removing the entire scene-conditioning interface lowered frozen action error by about 17%, to 1.285209. This suggests the interface deserves investigation, but it removes both current and predicted scene information, so it does not tell us whether the future predictions themselves helped. The mean-action baseline still had lower error; that does not make it a successful robot controller.

An earlier source-task simulator evaluation completed 17 trials with no successes before timing out. Together with the limited target attempts, this leaves us without a strong starting controller or a completed task-performance benchmark.

See the [research report](docs/research_report.md) for the initial experiment, detailed checks, and possible explanations.

</details>

## Reproduce the initial results

With Python 3.10 or later, from the repository root:

```bash
python3 tools/reproduce_results.py --results results
```

This command checks the bundled evidence and recomputes the initial 60-update diagnostics. It does not rerun training or simulation, or reproduce the later 300-update table. To regenerate the figure, install Matplotlib and NumPy, then run:

```bash
python3 tools/plot_submission.py
```

Software checks, with the optional research dependencies available:

```bash
PYTHONPATH=src:tools python3 -m unittest discover -s tests -q
```

See [reproducibility and v0 limits](docs/reproducibility.md) for evidence contents and GPU-stage documentation.

## Constraints and v0 scope

**Simulation-only: no robot hardware was available.** The application allowed two weeks, with a deadline of 16 October 2026, 23:59 BST. GPU experiments used a cloud A100 80 GB. Measured human-training peak allocation was 7.27 GiB; this is not a minimum GPU requirement. See [project information, time, and compute](docs/project_constraints.md).

This v0 includes research code, verified numerical evidence, a reproducible figure, and the experiment narrative. The three human videos and source-component previews are included. A clean-machine end-to-end training recipe remains pending. The human-video prediction losses were measured on overlapping windows from the training clips, so they do not show generalization to new human videos. The planned multi-seed target rollout matrix was not run.

Upstream EgoSteer notices are preserved under [licenses/egosteer](licenses/egosteer/NOTICE). Model and dataset access and licenses apply separately. No project-wide reuse license has been selected for v0.
