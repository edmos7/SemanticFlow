#!/usr/bin/env python3
"""Kernel-safe bounded source calibration, recovery and source-only evaluation.

Requires a reviewed bundle, a hash-pinned source index, and a separate temporary
HF token file. Only the six public source HDF5 files are read into cloud RAM.
The caller must export/hash-verify the archive and stop the authenticated runtime.
"""

import hashlib
import argparse
import json
import os
from pathlib import Path
import signal
import selectors
import shutil
import subprocess
import sys
import tarfile
import time


BALANCED_STAGES = ("balanced-panel-0",) + tuple(
    stage + "-" + str(i) for i in range(1, 4) for stage in ("balanced-train", "balanced-panel"))
COMPARISON_STAGES = ("comparison-preflight",) + tuple(
    "comparison-" + arm + "-" + mode for arm in ("base", "realman") for mode in ("initial", "train", "final"))

TARGET_STAGES = ("target-preflight", "target-rollouts", "target-frozen")

CLOCK_STAGES = ("clock-preflight", "clock-audit")

DECISION_STAGES = ("decision-preflight", "decision-audit", "decision-retention-train", "decision-retention-evaluate", "decision-semflow")

HUMAN_ACTION_STAGES = ("humanaction-preflight", "humanaction-frozen", "humanaction-standard", "humanaction-semflow")

ACTION_STAGES = ("action-preflight", "action-frozen", "action-standard", "action-semflow")

HUMAN_ADAPT_STAGES = ("humanadapt-preflight",) + tuple("humanadapt-" + arm + "-" + mode
    for arm in ("standard", "semflow") for mode in ("train", "evaluate"))

REHEARSAL_STAGES = ("rehearsal-preflight",) + tuple("rehearsal-" + arm + "-" + mode
    for arm in ("standard", "semflow") for mode in ("train", "evaluate"))

CONTINUATION_STAGES = ("continuation-preflight", "continuation-recover-120", "continuation-train-600",
    "continuation-panel-600", "continuation-train-1200", "continuation-panel-1200",
    "continuation-train-2400", "continuation-panel-2400")


def verified_comparison(output, stage, bundle_sha256):
    """Check an older durable receipt without mistaking the current cumulative archive for it."""
    if stage not in COMPARISON_STAGES[1:] + CONTINUATION_STAGES[1:] + REHEARSAL_STAGES[1:] + ACTION_STAGES[1:] + HUMAN_ADAPT_STAGES[1:] + HUMAN_ACTION_STAGES[1:] + DECISION_STAGES[1:] + CLOCK_STAGES[1:]:
        raise ValueError("unknown comparison result")
    outer_path = output / ("source-calibration-" + stage + ".json")
    outer = json.loads(outer_path.read_text())
    receipt = json.loads((output.parent / ("semflow-export-receipt-" + stage + ".json")).read_text())
    inner_path = output / "calibration" / (stage + ".json")
    expected = dict(stage=stage, bundle_sha256=bundle_sha256,
        report_sha256=hashlib.sha256(outer_path.read_bytes()).hexdigest(),
        comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
    if (outer["status"] != "passed" or outer["bundle_sha256"] != bundle_sha256
            or any(receipt.get(k) != v for k,v in expected.items())
            or outer.get("comparison_report_sha256") != expected["comparison_report_sha256"]):
        raise RuntimeError("comparison result differs from verified external export")
    inner = json.loads(inner_path.read_text())
    if inner["status"] != "passed":
        raise RuntimeError("comparison inner report failed")
    if "checkpoint" in inner:
        checkpoint = inner_path.with_suffix(".pt")
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if (actual != inner["checkpoint"]["sha256"] or receipt.get("checkpoint_sha256") != actual
                or checkpoint.stat().st_size != inner["checkpoint"]["bytes"]):
            raise RuntimeError("comparison checkpoint differs from verified external export")
    return inner


def balanced_previous(stage):
    if stage not in BALANCED_STAGES:
        raise ValueError("unknown balanced source stage")
    position = BALANCED_STAGES.index(stage)
    return BALANCED_STAGES[position - 1] if position else "preflight"


def balanced_checkpoint(output, stage):
    """Checkpoint to be exported with a completed stage, not its input."""
    if stage not in BALANCED_STAGES:
        raise ValueError("unknown balanced source stage")
    round_id = int(stage.rsplit("-", 1)[1])
    return output / ("calibration/balanced-train-%d/source.pt" % round_id) if round_id else None


def conditioning_checkpoint(checkpoint, receipt_path):
    """Only the already exported final balanced checkpoint may enter this probe."""
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = json.loads(receipt_path.read_text())
    if (digest != "0836b6006d2acce44e980ca1c2a8885e73f2dcac1ada2d8a01c0d764280e35fa"
            or checkpoint.stat().st_size != 49833988 or receipt.get("stage") != "balanced-train-3"
            or receipt.get("checkpoint_sha256") != digest):
        raise RuntimeError("conditioning checkpoint differs from the durable step2400 export")
    return digest


def require_export(output, stage, bundle_sha256, status):
    """Require the host's verified export receipt before risking another stage."""
    report_path = output / ("source-calibration-" + stage + ".json")
    prior = json.loads(report_path.read_text())
    receipt = json.loads((output.parent / ("semflow-export-receipt-" + stage + ".json")).read_text())
    if prior["status"] != status or prior["bundle_sha256"] != bundle_sha256:
        raise RuntimeError("prior stage must pass for this exact bundle")
    expected = {"stage": stage, "bundle_sha256": bundle_sha256,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "archive_sha256": hashlib.sha256((output.parent / "semflow-results.tar.gz").read_bytes()).hexdigest()}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("verified external export receipt differs from current stage artifacts")
    if stage in ("train", "restart"):
        checkpoint = output / "calibration/common-source.pt"
        if receipt.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError("checkpoint differs from the verified external export")
    if stage == "overfit":
        checkpoint = output / "calibration/overfit/fixed-window-step600.pt"
        if receipt.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError("overfit checkpoint differs from the verified external export")
    if stage in BALANCED_STAGES:
        checkpoint = balanced_checkpoint(output, stage)
        if checkpoint is not None and receipt.get("checkpoint_sha256") != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise RuntimeError("balanced checkpoint differs from verified external export")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "restart", "evaluate", "diagnose",
                                           "sampler-audit", "overfit", "overfit-audit",
                                           "conditioning-preflight", "conditioning-audit",
                                           "realman-preflight", "realman-audit",
                                           "human-preflight", "human-audit") + BALANCED_STAGES + COMPARISON_STAGES + CONTINUATION_STAGES + REHEARSAL_STAGES + ACTION_STAGES + HUMAN_ADAPT_STAGES + HUMAN_ACTION_STAGES + DECISION_STAGES + CLOCK_STAGES + TARGET_STAGES, required=True)
    args = parser.parse_args()
    root, output = Path("/content/semflow"), Path("/content/semflow-results")
    token = Path("/content/semflow-hf-token")
    bundle = Path("/content/semflow-source.tar.gz")
    root.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    report = {"status": "running", "stage": args.stage, "stages": [], "no_target_data": True,
              "storage": "Ephemeral runtime only; verified direct export to mounted SEMFLOW_DATA_ROOT required."}
    if args.stage in ("human-preflight", "human-audit") + HUMAN_ADAPT_STAGES:
        report.update(no_target_data=False, no_target_robot_data=True, human_rgb_only=True)
    if args.stage in TARGET_STAGES:
        report.update(no_target_data=False, no_target_robot_data=True, qualitative_target_execution=True)
    report_path = output / ("source-calibration-" + args.stage + ".json")
    started = time.monotonic()
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", HF_TOKEN_PATH=str(token),
        HF_HOME="/content/semflow-model-cache", CUBLAS_WORKSPACE_CONFIG=":4096:8")

    def run(name, command, timeout):
        record = {"name": name, "status": "running"}
        report["stages"].append(record)
        begin = time.monotonic()
        print("Starting " + name, flush=True)
        report_path.write_text(json.dumps(report, indent=2))
        with (output / (name + ".log")).open("wb") as log:
            process = None
            try:
                process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, start_new_session=True)
                with selectors.DefaultSelector() as events:
                    events.register(process.stdout, selectors.EVENT_READ)
                    while events.get_map():
                        remaining = timeout - (time.monotonic() - begin)
                        if remaining <= 0 or not events.select(timeout=remaining):
                            raise subprocess.TimeoutExpired(command, timeout)
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            events.unregister(process.stdout)
                            break
                        log.write(chunk)
                        log.flush()
                        print(chunk.decode("utf-8", errors="replace"), end="", flush=True)
                code = process.wait(timeout=max(.01, timeout - (time.monotonic() - begin)))
                record.update(status="passed" if code == 0 else "failed", returncode=code)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record["status"] = "timeout"
            except OSError as error:
                record.update(status="failed", error=str(error))
                raise
            finally:
                record["seconds"] = time.monotonic() - begin
                report_path.write_text(json.dumps(report, indent=2))
        if record["status"] != "passed":
            raise RuntimeError(name + " " + record["status"])
        print("Completed " + name, flush=True)

    try:
        with tarfile.open(bundle) as archive:
            archive.extractall(root, filter="data")
        manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
        for name, expected in manifest["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("source bundle member hash mismatch")
        report["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        if not token.is_file():
            raise RuntimeError("separate temporary authorized HF token file required")
        native_work = Path("/content/semflow-native-dependencies")
        simulator_work = Path("/content/semflow-simulator")
        native_python = native_work / "venv/bin/python"
        uv = shutil.which("uv") or str(native_work / "bootstrap/bin/uv")
        if args.stage in ("preflight", "conditioning-preflight", "realman-preflight", "comparison-preflight", "continuation-preflight", "human-preflight", "rehearsal-preflight", "action-preflight", "humanadapt-preflight", "humanaction-preflight", "decision-preflight", "clock-preflight", "target-preflight"):
            prior_setup = output / "native-dependencies/egosteer_cloud_run.json"
            if prior_setup.exists():
                if json.loads(prior_setup.read_text())["status"] != "passed":
                    raise RuntimeError("existing native setup failed; use a fresh runtime")
                run("native-setup-recheck", [str(native_python), str(root / "tools/probe_native_dependencies.py"),
                    "--upstream-root", str(native_work / "egosteer"),
                    "--output", str(output / "native-dependencies/recheck.json"), "--timeout-seconds", "90"], 100)
            else:
                run("native-setup", [sys.executable, str(root / "tools/run_egosteer_cloud.py")], 610)
            run("data-dependency", [str(uv), "pip", "install", "--python", str(native_python), "h5py==3.15.1"], 90)
            if args.stage in ("human-preflight", "rehearsal-preflight", "action-preflight", "humanadapt-preflight", "humanaction-preflight", "decision-preflight", "clock-preflight", "target-preflight"):
                run("video-dependency", [str(uv), "pip", "install", "--python", str(native_python),
                    "av==15.1.0", "Pillow==12.1.1"], 90)
            run("unit-tests", [str(native_python), str(root / "tools/run_cloud_tests.py")], 180)
            if args.stage in ("conditioning-preflight", "realman-preflight", "comparison-preflight", "continuation-preflight", "human-preflight", "rehearsal-preflight", "action-preflight", "humanadapt-preflight", "humanaction-preflight", "decision-preflight", "clock-preflight"):
                # This diagnostic has no simulator setup, smoke rollout or training.
                report.update(status="passed", optimizer_steps=0, rollouts=0)
                return
            run("simulator-setup", [sys.executable, str(root / "tools/run_libero_cloud.py"), "--install-only"], 310)
            if args.stage == "target-preflight":
                report.update(status="passed", optimizer_steps=0, rollouts=0)
                return
        elif args.stage in ("target-rollouts", "target-frozen"):
            require_export(output, "target-preflight", report["bundle_sha256"], "passed")
        elif args.stage in HUMAN_ADAPT_STAGES:
            previous = HUMAN_ADAPT_STAGES[HUMAN_ADAPT_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "humanadapt-preflight":
                verified_comparison(output, previous, report["bundle_sha256"])
        elif args.stage in CLOCK_STAGES:
            require_export(output, "clock-preflight", report["bundle_sha256"], "passed")
        elif args.stage in DECISION_STAGES:
            previous = DECISION_STAGES[DECISION_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "decision-preflight":
                verified_comparison(output, previous, report["bundle_sha256"])
        elif args.stage in HUMAN_ACTION_STAGES:
            previous = HUMAN_ACTION_STAGES[HUMAN_ACTION_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "humanaction-preflight":
                verified_comparison(output, previous, report["bundle_sha256"])
        elif args.stage in ACTION_STAGES:
            previous = ACTION_STAGES[ACTION_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "action-preflight":
                verified_comparison(output, previous, report["bundle_sha256"])
        elif args.stage in REHEARSAL_STAGES:
            previous = REHEARSAL_STAGES[REHEARSAL_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "rehearsal-preflight":
                verified_comparison(output, previous, report["bundle_sha256"])
        elif args.stage in BALANCED_STAGES:
            require_export(output, balanced_previous(args.stage), report["bundle_sha256"], "passed")
        elif args.stage in CONTINUATION_STAGES:
            previous = CONTINUATION_STAGES[CONTINUATION_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "continuation-preflight":
                prior = verified_comparison(output, previous, report["bundle_sha256"])
                if "stop_decision" in prior and not prior["stop_decision"]["continue_training"]:
                    raise RuntimeError("source-only stopping rule forbids next stage")
        elif args.stage in COMPARISON_STAGES:
            previous = COMPARISON_STAGES[COMPARISON_STAGES.index(args.stage)-1]
            require_export(output, previous, report["bundle_sha256"], "passed")
            if previous != "comparison-preflight":
                verified_comparison(output, previous, report["bundle_sha256"])
        else:
            prior_stage, prior_status = {"train": ("preflight", "passed"),
                "restart": ("train", "training_complete_evaluation_pending"),
                "evaluate": ("restart", "passed"), "diagnose": ("preflight", "passed"),
                "sampler-audit": ("preflight", "passed"), "overfit": ("sampler-audit", "passed"),
                "overfit-audit": ("overfit", "passed"),
                "conditioning-audit": ("conditioning-preflight", "passed"),
                "realman-audit": ("realman-preflight", "passed"),
                "human-audit": ("human-preflight", "passed")}[args.stage]
            require_export(output, prior_stage, report["bundle_sha256"], prior_status)
        calibration = output / "calibration"
        calibration.mkdir(exist_ok=True)
        if args.stage in ("target-rollouts", "target-frozen"):
            inner_path = calibration / args.stage / "report.json"
            command = [str(native_python), str(root / "tools/rollout_human_targets.py"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--panel-request", "/content/semflow-human-action-request.json", "--input-root", "/content",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json",
                "--upstream-root", str(native_work / "egosteer"),
                "--simulator-python", str(simulator_work / "venv/bin/python"),
                "--libero-checkout", str(simulator_work / "LIBERO"), "--output", str(inner_path)]
            if args.stage == "target-frozen":
                command += ["--condition", "frozen", "--after-report", "/content/semflow-after-report.json",
                            "--after-receipt", "/content/semflow-after-receipt.json"]
            run(args.stage, command, 2420)
            inner = json.loads(inner_path.read_text())
            report.update(status="passed" if inner["status"] == "completed" else "incomplete",
                optimizer_steps=0, rollouts=len(inner["rollouts"]),
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in CLOCK_STAGES:
            inner_path = calibration / "clock-audit.json"
            command = [str(native_python), str(root / "tools/probe_sampler_clock.py"),
                "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--clock-request", "/content/semflow-clock-request.json",
                "--previous-report", "/content/previous-frozen.json",
                "--previous-receipt", "/content/previous-frozen-receipt.json",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json", "--output", str(inner_path)]
            run(args.stage, command, 620)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=0, rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in DECISION_STAGES:
            mode = args.stage.removeprefix("decision-")
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/decision_diagnostics.py"),
                "--mode", mode, "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--panel-request", "/content/semflow-human-action-request.json",
                "--diagnostic-request", "/content/semflow-decision-request.json", "--input-root", "/content",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json", "--output", str(inner_path)]
            if mode == "retention-evaluate":
                prior = verified_comparison(output, "decision-retention-train", report["bundle_sha256"])
                command += ["--arm-checkpoint", str(calibration / "decision-retention-train.pt"),
                            "--arm-checkpoint-sha256", prior["checkpoint"]["sha256"]]
            if mode == "semflow":
                for stage in ("decision-audit", "decision-retention-evaluate"):
                    verified_comparison(output, stage, report["bundle_sha256"])
                command += ["--audit-report", str(calibration / "decision-audit.json"),
                            "--retention-report", str(calibration / "decision-retention-evaluate.json")]
            run(args.stage, command, 920)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=inner["optimizer_steps"], rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in HUMAN_ACTION_STAGES:
            condition = args.stage.removeprefix("humanaction-")
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/probe_human_actions.py"),
                "--condition", condition, "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--panel-request", "/content/semflow-human-action-request.json", "--input-root", "/content",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json",
                "--output", str(inner_path)]
            if condition != "frozen":
                verified_comparison(output, "humanaction-frozen", report["bundle_sha256"])
                command += ["--frozen-report", str(calibration / "humanaction-frozen.json")]
                if condition == "semflow":
                    verified_comparison(output, "humanaction-standard", report["bundle_sha256"])
                    command += ["--standard-report", str(calibration / "humanaction-standard.json")]
            run(args.stage, command, 920)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=0, rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in ACTION_STAGES:
            condition = args.stage.removeprefix("action-")
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/probe_adapted_actions.py"),
                "--condition", condition, "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--panel-request", "/content/semflow-action-request.json", "--input-root", "/content",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json",
                "--output", str(inner_path)]
            if condition != "frozen":
                verified_comparison(output, "action-frozen", report["bundle_sha256"])
                command += ["--frozen-report", str(calibration / "action-frozen.json")]
                if condition == "semflow":
                    verified_comparison(output, "action-standard", report["bundle_sha256"])
                    command += ["--standard-report", str(calibration / "action-standard.json")]
            run(args.stage, command, 920)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=0, rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in HUMAN_ADAPT_STAGES:
            _, condition, mode = args.stage.split("-")
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/adapt_human.py"),
                "--mode", mode, "--condition", condition,
                "--human-request", "/content/semflow-human-request.json",
                "--manifest", "/content/semflow-human-manifest.json",
                "--data-root", "/content/semflow-human-data",
                "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json",
                "--output", str(inner_path)]
            if condition == "semflow":
                verified_comparison(output, "humanadapt-standard-train", report["bundle_sha256"])
                command += ["--reference-report", str(calibration / "humanadapt-standard-train.json")]
            if mode == "evaluate":
                prior_stage = "humanadapt-" + condition + "-train"
                prior = verified_comparison(output, prior_stage, report["bundle_sha256"])
                command += ["--arm-checkpoint", str(calibration / (prior_stage + ".pt")),
                            "--arm-checkpoint-sha256", prior["checkpoint"]["sha256"]]
            run(args.stage, command, 920)
            inner = json.loads(inner_path.read_text())
            if condition == "semflow" and mode == "evaluate":
                standard = verified_comparison(output, "humanadapt-standard-evaluate", report["bundle_sha256"])
                if any(inner["tasks"][t]["frozen"] != standard["tasks"][t]["frozen"] for t in inner["tasks"]):
                    raise RuntimeError("paired frozen validation baseline differs between arms")
            report.update(status=inner["status"], optimizer_steps=inner["optimizer_steps"], rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in REHEARSAL_STAGES:
            _, condition, mode = args.stage.split("-")
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/rehearse_adaptation.py"),
                "--mode", mode, "--condition", condition,
                "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-rehearsal-request.json",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json",
                "--output", str(inner_path)]
            if condition == "semflow":
                verified_comparison(output, "rehearsal-standard-train", report["bundle_sha256"])
                command += ["--reference-report", str(calibration / "rehearsal-standard-train.json")]
            if mode == "evaluate":
                prior_stage = "rehearsal-" + condition + "-train"
                prior = verified_comparison(output, prior_stage, report["bundle_sha256"])
                command += ["--arm-checkpoint", str(calibration / (prior_stage + ".pt")),
                            "--arm-checkpoint-sha256", prior["checkpoint"]["sha256"]]
            run(args.stage, command, 920)
            inner = json.loads(inner_path.read_text())
            if condition == "semflow" and mode == "evaluate":
                standard = verified_comparison(output, "rehearsal-standard-evaluate", report["bundle_sha256"])
                if inner["frozen_validation"] != standard["frozen_validation"]:
                    raise RuntimeError("paired frozen validation baseline differs between arms")
            report.update(status=inner["status"], optimizer_steps=inner["optimizer_steps"], rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage == "human-audit":
            inner_path = calibration / "human-audit.json"
            run("human-audit", [str(native_python), str(root / "tools/probe_human_inputs.py"),
                "--upstream-root", str(native_work / "egosteer"),
                "--request", "/content/semflow-human-request.json",
                "--manifest", "/content/semflow-human-manifest.json",
                "--index", "/content/semflow-source-index.json",
                "--checkpoint", "/content/semflow-diagnostic-checkpoint.pt",
                "--checkpoint-receipt", "/content/semflow-checkpoint-receipt.json",
                "--data-root", "/content/semflow-human-data", "--output", str(inner_path)], 920)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=0, rollouts=0,
                human_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            return
        if args.stage in CONTINUATION_STAGES:
            _, mode, number = args.stage.split("-")
            step = int(number)
            start = {600:120,1200:600,2400:1200}[step] if mode == "train" else step
            if start == 120:
                checkpoint = Path("/content/semflow-continuation-input.pt")
                checkpoint_sha = "da71ac6ca3a5f18ca4d8f891d30ac007a6208a41cfb01e2ad0a71456ae90babb"
            else:
                source_stage = "continuation-train-" + str(start)
                source = verified_comparison(output, source_stage, report["bundle_sha256"])
                checkpoint = calibration / (source_stage + ".pt")
                checkpoint_sha = source["checkpoint"]["sha256"]
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/continue_realman_source.py"),
                "--stage", args.stage, "--upstream-root", str(native_work / "egosteer"),
                "--index", "/content/semflow-source-index.json", "--output", str(inner_path),
                "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_sha]
            if mode == "panel":
                previous_panel = Path("/content/semflow-continuation-panel120.json")
                if step != 600:
                    prior_step = {1200:600,2400:1200}[step]
                    prior_stage = "continuation-panel-" + str(prior_step)
                    verified_comparison(output, prior_stage, report["bundle_sha256"])
                    previous_panel = calibration / (prior_stage + ".json")
                command += ["--initial-panel", "/content/semflow-continuation-panel120.json",
                            "--previous-panel", str(previous_panel)]
            run(args.stage, command, 3950 if mode == "train" else 1100)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=inner["optimizer_steps"], rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            if "checkpoint" in inner:
                report["checkpoint_sha256"] = inner["checkpoint"]["sha256"]
            if "stop_decision" in inner:
                report["stop_decision"] = inner["stop_decision"]
            return
        if args.stage in COMPARISON_STAGES:
            _, arm, mode = args.stage.split("-")
            inner_path = calibration / (args.stage + ".json")
            command = [str(native_python), str(root / "tools/compare_source_initializations.py"),
                "--arm", arm, "--mode", mode, "--upstream-root", str(native_work / "egosteer"),
                "--index", "/content/semflow-source-index.json", "--output", str(inner_path)]
            if mode != "initial":
                source_stage = "comparison-" + arm + ("-initial" if mode == "train" else "-train")
                source = verified_comparison(output, source_stage, report["bundle_sha256"])
                command += ["--checkpoint", str(calibration / (source_stage + ".pt")),
                            "--checkpoint-sha256", source["checkpoint"]["sha256"]]
            if arm == "realman":
                verified_comparison(output, "comparison-base-initial", report["bundle_sha256"])
                command += ["--reference-initial", str(calibration / "comparison-base-initial.json")]
                if mode == "train":
                    verified_comparison(output, "comparison-base-train", report["bundle_sha256"])
                    command += ["--reference-train", str(calibration / "comparison-base-train.json")]
            run(args.stage, command, 1100)
            inner = json.loads(inner_path.read_text())
            report.update(status=inner["status"], optimizer_steps=inner["optimizer_steps"], rollouts=0,
                comparison_report_sha256=hashlib.sha256(inner_path.read_bytes()).hexdigest())
            if "checkpoint" in inner:
                report["checkpoint_sha256"] = inner["checkpoint"]["sha256"]
            if args.stage == COMPARISON_STAGES[-1]:
                for completed in COMPARISON_STAGES[1:-1]:
                    verified_comparison(output, completed, report["bundle_sha256"])
                # Use the native environment (Torch/NumPy); keep orchestration dependency-free.
                comparison = calibration / "comparison-summary.json"
                run("comparison-summary", [str(native_python), str(root / "tools/compare_source_initializations.py"),
                    "--summarize", str(calibration), "--output", str(comparison)], 60)
            return
        if args.stage == "realman-audit":
            audit_output = calibration / "realman-audit.json"
            run("realman-audit", [str(native_python), str(root / "tools/probe_realman.py"),
                "--upstream-root", str(native_work / "egosteer"),
                "--index", "/content/semflow-source-index.json", "--output", str(audit_output)], 900)
            report.update(status=json.loads(audit_output.read_text())["status"], optimizer_steps=0, rollouts=0)
            return
        command = [str(native_python), str(root / "tools/train_source_calibration.py"),
            "--upstream-root", str(native_work / "egosteer"),
            "--cache-dir", "/content/semflow-model-cache/hub",
            "--index", "/content/semflow-source-index.json", "--index-sha256",
            "45f87e0eb9b2ab90edb2e81d0e5945da26e8184c9285476f6b7f55895a879e38",
            "--simulator-python", str(simulator_work / "venv/bin/python"),
            "--libero-checkout", str(simulator_work / "LIBERO")]
        mode = "smoke" if args.stage == "preflight" else args.stage
        round_id = None
        if args.stage in BALANCED_STAGES:
            mode, round_text = args.stage.rsplit("-", 1)
            round_id = int(round_text)
            command += ["--balanced-round", round_text]
        original_input = mode in ("diagnose", "sampler-audit", "overfit") or args.stage in ("balanced-panel-0", "balanced-train-1")
        if original_input:
            receipt = json.loads(Path("/content/semflow-checkpoint-receipt.json").read_text())
            checkpoint = Path("/content/semflow-diagnostic-checkpoint.pt")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if receipt.get("stage") != "train" or receipt.get("checkpoint_sha256") != digest:
                raise RuntimeError("diagnostic checkpoint differs from durable training receipt")
            report["checkpoint_sha256"] = digest
            command += ["--checkpoint", str(checkpoint), "--checkpoint-sha256", digest]
        if mode == "overfit-audit":
            checkpoint = calibration / "overfit/fixed-window-step600.pt"
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            command += ["--checkpoint", str(checkpoint), "--checkpoint-sha256", digest]
        if mode == "conditioning-audit":
            checkpoint = Path("/content/semflow-diagnostic-checkpoint.pt")
            digest = conditioning_checkpoint(checkpoint, Path("/content/semflow-checkpoint-receipt.json"))
            report["checkpoint_sha256"] = digest
            command += ["--checkpoint", str(checkpoint), "--checkpoint-sha256", digest,
                        "--reference-panel", "/content/semflow-reference-panel.json"]
        if args.stage in BALANCED_STAGES and not original_input:
            prior_round = round_id - 1 if mode == "balanced-train" else round_id
            checkpoint = calibration / ("balanced-train-%d/source.pt" % prior_round)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report["checkpoint_sha256"] = digest
            command += ["--checkpoint", str(checkpoint), "--checkpoint-sha256", digest]
        limit = {"smoke": 1200, "train": 3000, "restart": 300, "evaluate": 1920, "diagnose": 900,
                 "sampler-audit": 1100, "overfit": 1500, "overfit-audit": 1100,
                 "balanced-train": 2100, "balanced-panel": 900, "conditioning-audit": 900}[mode]
        label = args.stage if args.stage in BALANCED_STAGES else mode
        run(label, command + ["--mode", mode, "--output", str(calibration / (label + ".json"))], limit)
        report["status"] = json.loads((calibration / (label + ".json")).read_text())["status"]
    except Exception as error:
        report.update(status="failed", error=str(error).split("\n")[0][:600])
    finally:
        token.unlink(missing_ok=True)
        report["temporary_token_removed"] = not token.exists()
        report["seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        exported = subprocess.run([sys.executable, str(root / "tools/export_cloud_results.py"),
            *(["--include-videos"] if args.stage in TARGET_STAGES else [])],
            capture_output=True, text=True, timeout=120, check=True)
        print(exported.stdout, flush=True)
    print(json.dumps(report), flush=True)
    if report["status"] == "failed":
        raise RuntimeError("source calibration pipeline failed; inspect exported evidence")


if __name__ == "__main__":
    main()
