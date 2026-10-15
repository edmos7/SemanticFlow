# Bounded human adaptation

Compare standard future-feature adaptation with SemFlow on three single-view
human clips. Each task and arm starts independently from the same Base step600
checkpoint. Train only the shared rank8 adapters for60 updates, batch1,
learning rate5e-5, AdamW weight decay0.01 and gradient clipping1.0.

Use12 deterministically spaced valid anchors per clip, retaining measured
one-second causal history and0.5/1/2-second future targets. Teacher targets,
including their task context, are cached at the common initialization before
any updates. The absent head view is masked; human state/action slots are
absent. Each update includes measured source-action retention. Calibrate the
semantic weight using only the six fixed source-training gradient pairs used
in the source rehearsal. Both arms share windows, ordering and noise seeds.

Export each arm's three task checkpoints before fresh-process recovery.
Verify adapters, optimizer, RNG and reference loss; report pre/post human
training-window forecast losses and the fixed12-window held-out source action
flow loss. These human scores measure fitting to the supplied clips: there is
no independent human test set, checkpoint selection or robot rollout.
The source controller gate remains unmet, so this bounded experiment cannot
establish successful robot transfer. Do not treat lower flow loss as improved
sampled actions.
