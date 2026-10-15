# Source ingestion

`configs/source_data.json` pins the six approved LIBERO-90 HDF5 files to the
[upstream mirror](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90):
3,550,136,643 bytes total. The lock includes file sizes, SHA-256 hashes, and
source BDDL hashes from LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
No suite archive, target demonstration, model, or Hub cache is downloaded.

Use Python with the optional `data` dependencies from `pyproject.toml`.
Set `SEMFLOW_DATA_ROOT` to the existing project directory on the mounted
External HD. Missing, system-root, and read-only storage fail before downloads.
Data subpaths reject symlinks and nested mounts. Nothing falls back to `/tmp`
or the system disk.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m semflow.cli acquire-sources
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m semflow.cli ingest-sources
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m semflow.cli verify-sources
```

Raw files stay at `$SEMFLOW_DATA_ROOT/datasets/libero90-source-v1/raw/`.
The adjacent `index.json` contains the episode manifest, decoded-content
provenance, fixed split assignments, and normalization statistics. Verification
rehashes the pinned files and recomputes the entire index, including statistics.
Existing verified files are reused; existing index differences fail.
Interrupted or corrupt `.partial` files remain for inspection and are never
silently overwritten. A complete, correctly hashed download can be finalized
on retry. An incomplete partial requires an explicit operator cleanup before
retrying; do not delete valid source files.

Ingestion requires exactly the six allowlisted, hash-verified files, matching
task names in all three metadata paths and the exact component instruction,
Panda delta OSC_POSE at 20 Hz, and finite aligned arrays. The pinned mirror omits
embedded BDDL text; its absence is recorded. If present, that text must match
the separately pinned upstream BDDL hash. Whole-file pins bind the actual
demonstration content to the six reviewed source files.
The content digest covers every decoded numeric episode dataset in sorted
field order with shape/type framing and canonical little-endian numeric bytes.
It excludes filenames, episode names, task labels, and storage compression.
Duplicate content under another name or task is rejected. Simulator states
remain in the original files for provenance hashing; the training reader never
returns them, rewards, dones, or privileged future states.

Whole-episode hashes select the existing seed-42, 20% validation split.
Every source must have both splits; a missing split fails without reseeding.
The source-only completeness check does not require human clips. They remain
an empty list, with no fabricated state, actions, or camera motion.

The pinned [LIBERO conversion script](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/scripts/create_dataset.py)
records observations after each action. The reader pairs `obs[:-1]` with
`actions[1:]`, drops the final observation without a next action, and timestamps
aligned frames at 20 Hz. Stored axis-angle orientation becomes quaternion
`xyzw`, with nonnegative `w`. Panda state has nine values: EEF position,
quaternion, and two gripper positions. RGB uses the stored convention and is
flipped vertically only for OpenGL images.

Normalization uses frame-weighted population mean/std from **source training
episodes only**, after alignment. Scales below `1e-6` become `1`; z-scores are
reversible and unclipped. The index names every contributing episode hash and
stores per-dimension observed bounds. These bounds are descriptive statistics,
not simulator safety limits. Normalized actions still require controller
conversion and safety validation before execution.

`read_episode` requires an index verified against its source files and returns
only upright RGB, normalized measured Panda state/actions, timestamps,
instruction, split, and content hash. Causal history/chunk batching belongs to
the following source-calibration milestone.

The in-memory tests validate contracts and exclusion boundaries. They do not
verify real-file compatibility, source control, semantic grounding, or transfer.
Run real acquisition, ingestion, and verification before advancing calibration.
