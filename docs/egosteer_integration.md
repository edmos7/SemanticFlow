# EgoSteer feasibility boundary

The probe loads the released pretrained model through its actual Hydra policy,
checkpoint validator, Qwen batch processor, backbone stream, and world-model loss.
It is an engineering gate, not a training result. No human target clips are needed.

## Pinned sources

| Artifact | Revision |
| --- | --- |
| [Official implementation](https://github.com/egosteer/egosteer) | `5e646d9dd278280af14e5050993862bead94cc19` |
| [EgoSteer base](https://huggingface.co/EgoSteer/EgoSteer-3B-Base) | `3e299fb4e2abc17244168be36d95a2920a9c9ad3` |
| [Qwen backbone](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `89644892e4d85e24eaac8bacfd4f463576704203` |
| [DINOv3 teacher](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `ea8dc2863c51be0a264bab82070e3e8836b02d51` |

The base checkpoint is approximately 5.58 GB and excludes the frozen teacher.
Native construction also downloads Qwen initialization weights (approximately
4.26 GB) and DINOv3 (approximately 1.21 GB). Allow at least 15 GB runtime disk
for these assets, in addition to the CUDA environment. Peak host/GPU RAM must
be measured; adapter size alone does not determine activation memory.

DINOv3 has a manual Hugging Face access gate. Colab authentication does not grant
this access. `access` checks teacher access first and downloads no weights; an access
failure writes a JSON report and exits nonzero. Use an approved account through
the runtime's normal Hugging Face authentication or pass an already authorized
complete teacher directory. Do not copy secrets into code or launch bundles.

## Remote setup and commands

`tools/run_egosteer_cloud.py` provides a bounded dependency gate in a fresh,
isolated Python 3.12 environment. It pins Torch 2.10.0 / torchvision 0.25.0 with
CUDA 13, Transformers 5.6.2, and an official FlashAttention 2.8.3 wheel whose
SHA-256 is verified before installation. Native imports and BF16 attention
forward/backward must pass before model downloads. The notebook stack stays
unchanged. Pass fresh `--work-dir` and `--output-dir` paths; the total budget is
at most 600 seconds. Use the resulting `venv/bin/python` for model stages below.

Use a clean checkout of the pinned official source on the cloud runtime. Its
recommended stack is PyTorch `2.10.0` / torchvision `0.25.0` with CUDA 12.8 and
Transformers `5.6.2`. First inspect the hosted runtime's existing versions and
test compatibility without replacing its CUDA stack. The probe records actual
versions and does not require a matching version string. For a separate
environment reproducing the reference, the additional dependencies are:

```bash
python -m pip install transformers==5.6.2 hydra-core omegaconf huggingface_hub safetensors pillow numpy psutil einops packaging ninja
MAX_JOBS=2 python -m pip install flash-attn --no-build-isolation
```

The native implementation imports its FlashAttention vision patch unconditionally.
Changing the vision YAML field to SDPA does not avoid that requirement. The probe
uses native FlashAttention for vision and SDPA for text/experts; it does not use
an attention shim or `torch.compile`. Do not install the entire upstream training
requirements file for this gate: the probe does not need rendering, hand models,
WebDataset, xformers, or their dataset dependencies.

From the uploaded SemFlow bundle, run bounded stages:

```bash
python tools/probe_egosteer.py --stage environment --output /content/semflow-runs/egosteer-environment.json
python tools/probe_egosteer.py --stage access --output /content/semflow-runs/egosteer-access.json
python tools/probe_egosteer.py --stage model-load --upstream-root /content/egosteer --cache-dir /content/hf-cache --output /content/semflow-runs/egosteer-load.json
python tools/probe_egosteer.py --stage backward --upstream-root /content/egosteer --cache-dir /content/hf-cache --output /content/semflow-runs/egosteer-backward.json
```

`forward` is available between load and backward for isolating failures. All
stages write a final JSON success/failure record. Download reports and the small
`.audit.pt` checkpoint to durable storage before releasing the GPU. No stage
allocates a cloud runtime itself or starts an unbounded training process.

## What the probe establishes

The released architecture is preserved: six RGB history frames, a 384-square
DINO teacher, one future frame, and a 32-step, 48-dimensional human action
interface. Native checkpoint loading tolerates absent `frozen_teacher.*` only;
all world-model parameter groups must exist. The teacher loads real weights.
Memory-mapped, weights-only checkpoint reading reduces host RAM duplication.

The synthetic moving-square clip supplies RGB and ordinary text, with no state,
action, or camera-motion labels. Future-action and future-motion conditioning
are disabled explicitly. A second camera is masked padding required by the
native two-view world-stream interface, not additional observed data. Native
processing retains the complete prompt without truncation.

The backward stage freezes everything except the first shared text layer's
2048-element input-normalization weight. It requires a finite, nonzero gradient
from the native raw-feature prediction loss, a parameter update, and an optimizer
checkpoint round trip. This is a narrow gradient audit; it is not LoRA and does
not establish full adapter-training memory or recovery after a runtime restart.
The optional native action-stream check reports velocity shape and finiteness
from generated noise; it is not normalized Panda control or a simulator rollout.

## Native SemFlow interface

Load the native checkpoint first, then call `inject_shared_lora(native)` and
construct `NativeSemFlowPolicy(native)`. The shared language q/v projections use
rank 8, unit residual scaling and zero-initialized output factors. Standard and
SemFlow adaptation use precisely the same parameter set. Restore the entire
common source checkpoint separately for every arm/task/seed; adapter-only audit
files do not contain the new source-calibrated interfaces.

`encode_observations` accepts a complete causal prompt, six observation times,
current raw RGB and observed-view validity. It rejects future frames, labels,
actions and answer tokens. Human state stays absent; actual Panda state uses
six source-normalized 9D observations (eef position, xyzw quaternion, gripper).
The 0.5/1/2-second readout reuses the native expert independently at each horizon;
learned time offsets preserve its unmodified one-second queries. Full-scene
DINO patches are pooled into 16 task-conditioned queries per horizon.

`predict_velocity` conditions native action tokens on those forecasts before
the flow expert. Learned 7D input/output maps retain the pretrained native
action modules. `sample_actions` produces 32 normalized actions from explicit
paired noise; execute eight only after source normalization and simulator
controller conversion are established. It does not issue robot commands.

`enable_shared_adapter_training(wrapper, policy=native)` freezes/evaluates the
complete controller, then enables only the shared adapters. Prediction-side
pooling and readouts retain gradients; teacher targets belong in separate
loss-only calls under `no_grad`. `tools/probe_native_integration.py` checks the
real pretrained path, both loss gradients, adapter serialization and runtime
forecast interventions using synthetic RGB. Its new interfaces are untrained.

Source normalization, controller/pooling calibration, source-control success,
restart recovery and human-video adaptation/evaluation remain. A successful
synthetic integration audit does not establish manipulation or transfer.
