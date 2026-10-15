# Qualitative before/after target rollouts

The frozen source-calibrated Base600 model and the corresponding human-adapted SemFlow model each attempted the three intended LIBERO-10 compositions. Every attempt reached the 400-step limit without simulator success. These six qualitative attempts do not establish a benchmark success rate or satisfy the original source-control gates.

Each pair used seed 10000, benchmark initial-state index 0, the same action-noise sequence, the historical indexed ten-step sampler, eight actions per replan, and the same controller conversion. No optimizer, checkpoint selection, outcome-based seed selection, or new adaptation was performed. Exported reports verified matching task, initial-state and BDDL identities, reproducible resets, and unchanged model parameters/buffers.

## Runner and isolation

`tools/rollout_human_targets.py` validates the existing source request and human-training/recovery artifact chains, then runs `--condition semflow` or `--condition frozen`. Frozen comparisons additionally require the exported adapted report and receipt through `--after-report` and `--after-receipt`. See `--help` for the full input contract.

The simulator worker remains source-only by default. Explicit target mode rejects demonstration replay and arbitrary reset-state inputs. Benchmark states initialize the simulator only; policy inputs contain measured RGB, Panda proprioception, and ordinary task language. The native interpreter verifies the pinned benchmark files and converts initial states to NumPy for the separate simulator environment without Torch.

The cloud orchestrator exposes `target-preflight`, `target-rollouts` and `target-frozen` stages. Preflight prepares the pinned environments and runs software tests without robot execution. A matching, hash-verified host export receipt is required before inference. The rollout process is bounded to 2,400 seconds, each trial to 600 seconds, and the enclosing subprocess to 2,420 seconds. MP4 export is explicitly enabled for these stages. Failures and partial attempts are retained when cleanup can run; forcibly killed processes may leave undecodable videos.

This is a historical experiment interface, not a clean-machine end-to-end reproduction recipe. Checkpoints and original export receipts are not distributed with the repository.

## Presentation and verification

`tools/build_rollout_comparison.py` takes `--before-run`, `--after-run` and `--output` directories. It checks completed reports, matching experimental conditions, video hashes and exact 20 Hz timestamps before creating labeled side-by-side MP4/GIF previews, contact sheets and sanitized provenance. Different video durations require explicit terminal handling; the builder never silently stretches time.

Full MP4s contain 401 frames at 20 Hz, with frozen Base600 on the left and adapted SemFlow on the right. Each model panel contains the fixed agent camera and wrist camera. GIF previews sample approximately six frames per second with rounded timing. All exported and presentation videos were decoded and hash-verified; contact sheets were visually inspected. The README links the full comparison and separate original videos. Source-dataset previews remain distinct from these learned-policy executions.

See [media provenance](../assets/rollouts/paired/provenance.json) for checkpoint, adapter, task and video hashes, and [the research report](research_report.md) for the offline negative results and their limits.
