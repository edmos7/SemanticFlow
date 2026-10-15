#!/usr/bin/env python3
"""Bounded real-source ingestion, independent recomputation, and reader audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import signal
import time
import traceback

from semflow.config import load_config, load_json
from semflow.source_acquisition import external_path, json_bytes
from semflow.source_data import ingest_sources, read_episode
from semflow.storage import resolve_data_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--lock", type=Path, default=Path("configs/source_data.json"))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_name):
        parser.error("run name must contain only letters, digits, underscores or hyphens")
    root = resolve_data_root()
    relative = f"runs/{args.run_name}"
    output = external_path(root, relative)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "stages": {}, "reader_audits": [],
              "python": platform.python_version(), "limitations": [
                  "No policy training or simulator rollout.",
                  "No manipulation success, semantic grounding or transfer result."]}
    started = time.monotonic()

    def save():
        resolve_data_root(str(root))
        report["seconds"] = time.monotonic() - started
        path = external_path(root, relative + "/verification.json")
        temporary = external_path(root, relative + "/verification.json.partial")
        temporary.write_bytes(json_bytes(report))
        temporary.replace(path)

    def deadline(signum, frame):
        raise TimeoutError("source ingestion audit exceeded 900 seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        import h5py
        import numpy as np

        report["packages"] = {"numpy": np.__version__, "h5py": h5py.__version__}
        config, tasks = load_config(args.config)
        lock = load_json(args.lock)
        report["source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path("src/semflow/source_acquisition.py"), Path("src/semflow/source_data.py"),
                         Path("src/semflow/storage.py"), Path(__file__), args.config, args.lock]}
        save()
        for stage, verify_only in (("ingest", False), ("independent_verification", True)):
            print(stage, flush=True)
            report["stages"][stage] = ingest_sources(lock, tasks, config, verify_only=verify_only)
            save()
        index_path = Path(report["stages"]["ingest"]["index"])
        index = load_json(index_path)
        report["normalization_frames"] = index["normalization"]["state"]["count"]
        report["total_aligned_frames"] = sum(p["aligned_frames"] for p in index["provenance"].values())
        for task in sorted(index["coverage"]):
            for split in ("train", "validation"):
                record = next(e for e in index["manifest"]["episodes"] if e["task"] == task and e["split"] == split)
                observation = read_episode(root, index, record["id"])
                n = len(observation["timestamps_seconds"])
                if (observation["panda_state"].shape != (n, 9) or observation["actions"].shape != (n, 7)
                        or not np.isfinite(observation["panda_state"]).all()
                        or not np.isfinite(observation["actions"]).all()):
                    raise ValueError("real normalized Panda reader failed shape/finite checks")
                report["reader_audits"].append({"id": record["id"], "split": split,
                    "sha256": record["sha256"], "frames": n,
                    "state_shape": list(observation["panda_state"].shape),
                    "action_shape": list(observation["actions"].shape),
                    "rgb_shapes": {k: list(v.shape) for k, v in observation["rgb"].items()}})
                save()
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        signal.alarm(0)
        save()
    path = output / "verification.json"
    print(json.dumps({"status": report["status"], "report": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "seconds": report["seconds"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
