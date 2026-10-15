"""Small local CLI; unsupported training operations are intentionally absent."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

from . import __version__
from .config import ContractError, load_config, load_json, validate_task_manifest
from .data import validate_episode_manifest
from .storage import storage_status


def recording_guide(tasks: dict) -> str:
    lines = [
        "# SemFlow egocentric recording guide", "",
        "Record one continuous RGB clip per task, approximately 20–45 seconds long.",
        "Use a head- or chest-mounted camera with both the workspace and relevant objects visible.",
        "Show the initial scene, complete manipulation, and final state. Keep the final state visible for a few seconds.",
        "Match object roles and outcomes; an identical simulator room is unnecessary.",
        "Avoid editing, speed changes, cuts, manual labels, or hand tracking. Preserve the original file and timestamps.",
        "Record an ordinary task instruction alongside each clip. Do not select recordings using robot evaluation scores.", "",
    ]
    for target in tasks["targets"]:
        lines.extend([f"## {target['id']}", "", target["instruction"], "", target["human_analogue"], ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="semflow")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="inspect configuration, optional imports, and mounted storage")
    doctor.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    doctor.add_argument("--data-root")
    doctor.add_argument("--json", action="store_true")
    validate = commands.add_parser("validate-config", help="check the locked pilot contract")
    validate.add_argument("config", type=Path, nargs="?", default=Path("configs/pilot.json"))
    validate.add_argument("--training-ready", action="store_true")
    inspect = commands.add_parser("inspect-manifest", help="check episode splits and human-only modalities")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    inspect.add_argument("--require-complete", action="store_true")
    guide = commands.add_parser("recording-guide", help="print the three-task human recording guide")
    guide.add_argument("--tasks", type=Path, default=Path("configs/tasks.json"))
    guide.add_argument("--output", type=Path)
    for name in ("acquire-sources", "ingest-sources", "verify-sources"):
        source = commands.add_parser(name, help="acquire or audit only the six pinned source tasks")
        source.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
        source.add_argument("--lock", type=Path, default=Path("configs/source_data.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = {"version": __version__, "python": platform.python_version(), "errors": [], "optional_modules": {name: importlib.util.find_spec(name) is not None for name in ("torch", "libero", "h5py")}}
            try:
                config, tasks = load_config(args.config)
                report["config"] = {"valid": True, "targets": len(tasks["targets"]), "checkpoint_pinned": config["checkpoint_revision"] is not None}
            except (OSError, ValueError) as error:
                report["errors"].append(str(error))
            try:
                report["storage"] = storage_status(args.data_root)
            except (OSError, ValueError) as error:
                report["errors"].append(str(error))
            report["ok"] = not report["errors"]
            print(json.dumps(report, indent=2) if args.json else "\n".join(f"{key}: {value}" for key, value in report.items()))
            return 0 if report["ok"] else 1
        if args.command == "validate-config":
            config, tasks = load_config(args.config, training_ready=args.training_ready)
            print(json.dumps({"valid": True, "experiment": config["experiment"], "targets": len(tasks["targets"]), "training_ready_checked": args.training_ready}, indent=2))
        elif args.command == "inspect-manifest":
            config, tasks = load_config(args.config)
            counts = validate_episode_manifest(load_json(args.manifest), tasks, require_complete=args.require_complete, split_seed=config["split_seed"], validation_fraction=config["source_validation_fraction"])
            print(json.dumps({"valid": True, "complete_required": args.require_complete, **counts}, indent=2))
        elif args.command in ("acquire-sources", "ingest-sources", "verify-sources"):
            from .source_acquisition import acquire_sources
            from .source_data import ingest_sources

            config, tasks = load_config(args.config)
            lock = load_json(args.lock)
            if args.command == "acquire-sources":
                result = {"files": acquire_sources(lock, tasks)}
            else:
                result = ingest_sources(lock, tasks, config, verify_only=args.command == "verify-sources")
            print(json.dumps(result, indent=2))
        elif args.command == "recording-guide":
            content = recording_guide(validate_task_manifest(load_json(args.tasks)))
            if args.output:
                args.output.write_text(content, encoding="utf-8")
            else:
                print(content, end="")
        return 0
    except (OSError, ValueError) as error:
        print(f"semflow: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
