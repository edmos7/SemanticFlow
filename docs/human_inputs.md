# Human RGB input compatibility

The optional `video` dependencies decode hash-verified single-view recordings.
`read_human_window` checks every decoded presentation timestamp against the
manifest and keeps only selected frames in memory. Six distinct observations
cover one second at nominal 5 Hz: select the latest original frame at or before
each requested history time. Future loss targets use the first frame at or after
0.5, 1 and 2 seconds. Incomplete windows fail.

Frames retain the full scene through bilinear resizing and centered black
padding to the native configured image size. No stretching or cropping is used.
This preprocessing is a declared compatibility choice, not validated transfer.

`HumanBatchProcessor` accepts only history RGB, measured times and ordinary
instruction. It sends one chest-view video to the native processor, omits state
and action slots, and masks the absent head view. Empty native storage slots
are padding, never additional observations. Future images enter a separate
loss-target path.

The upstream processor assumes constant frame timing. The human adapter instead
encodes relative presentation times as integer microsecond ticks in its
index/rate metadata interface, with frame resampling disabled. This tick rate
is not a camera FPS claim. Native Qwen then averages pairs of timestamps and
prints them to one decimal place; the GPU preflight verifies those actual prompt
tags. Forecast targets retain their original timestamp offsets independently.

`probe_human_inputs.py` verifies a reviewed manifest, source index, checkpoint
and export receipt before loading the pinned native model. Three fixed windows
test losses, gradients through frozen readouts, missing-view invariance and
unchanged parameters. No optimizer steps or rollouts occur. Local unit fixtures
cannot establish that this native GPU probe passes or that adaptation helps.
