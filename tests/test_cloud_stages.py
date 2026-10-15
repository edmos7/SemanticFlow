"""A completed cloud stage cannot substitute for a verified durable export."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("cloud_stages",
    Path(__file__).resolve().parents[1] / "tools/run_source_calibration_cloud.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExportGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "results"
        (self.output / "calibration").mkdir(parents=True)
        self.bundle = "a" * 64
        report = self.output / "source-calibration-train.json"
        report.write_text(json.dumps({"status": "training_complete_evaluation_pending",
                                      "bundle_sha256": self.bundle}))
        archive = self.root / "semflow-results.tar.gz"
        archive.write_bytes(b"exported archive fixture")
        self.checkpoint = self.output / "calibration/common-source.pt"
        self.checkpoint.write_bytes(b"trained checkpoint fixture")
        self.receipt = self.root / "semflow-export-receipt-train.json"
        self.receipt.write_text(json.dumps({"stage": "train", "bundle_sha256": self.bundle,
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()}))

    def verify(self):
        runner.require_export(self.output, "train", self.bundle, "training_complete_evaluation_pending")

    def test_completed_stage_without_export_receipt_is_rejected(self):
        self.receipt.unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()

    def test_verified_export_allows_continuation(self):
        self.verify()

    def test_checkpoint_changed_after_export_is_rejected(self):
        self.checkpoint.write_bytes(b"unexported replacement")
        with self.assertRaisesRegex(RuntimeError, "checkpoint differs"):
            self.verify()

    def test_receipt_from_another_archive_is_rejected(self):
        (self.root / "semflow-results.tar.gz").write_bytes(b"different archive")
        with self.assertRaisesRegex(RuntimeError, "receipt differs"):
            self.verify()

    def test_overfit_checkpoint_requires_its_own_verified_export(self):
        report = self.output / "source-calibration-overfit.json"
        report.write_text(json.dumps({"status": "passed", "bundle_sha256": self.bundle}))
        checkpoint = self.output / "calibration/overfit/fixed-window-step600.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"fixed window checkpoint fixture")
        receipt = json.loads(self.receipt.read_text())
        receipt.update(stage="overfit", report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        (self.root / "semflow-export-receipt-overfit.json").write_text(json.dumps(receipt))
        runner.require_export(self.output, "overfit", self.bundle, "passed")
        checkpoint.write_bytes(b"not exported")
        with self.assertRaisesRegex(RuntimeError, "overfit checkpoint differs"):
            runner.require_export(self.output, "overfit", self.bundle, "passed")

    def test_balanced_stage_chain_requires_every_export_and_checkpoint(self):
        prior = "preflight"
        for stage in runner.BALANCED_STAGES:
            self.assertEqual(runner.balanced_previous(stage), prior)
            prior = stage
        for stage in ("balanced-train-1", "balanced-panel-1"):
            checkpoint = runner.balanced_checkpoint(self.output, stage)
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(b"balanced fixture")
            report = self.output / ("source-calibration-" + stage + ".json")
            report.write_text(json.dumps({"status": "passed", "bundle_sha256": self.bundle}))
            receipt = json.loads(self.receipt.read_text())
            receipt.update(stage=stage, report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            (self.root / ("semflow-export-receipt-" + stage + ".json")).write_text(json.dumps(receipt))
            runner.require_export(self.output, stage, self.bundle, "passed")
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "balanced checkpoint differs"):
                runner.require_export(self.output, stage, self.bundle, "passed")

    def test_conditioning_checkpoint_rejects_wrong_stage_even_with_matching_hash(self):
        checkpoint = SimpleNamespace(read_bytes=lambda: b"fixture", stat=lambda: SimpleNamespace(st_size=49833988))
        digest = "0836b6006d2acce44e980ca1c2a8885e73f2dcac1ada2d8a01c0d764280e35fa"
        receipt = SimpleNamespace(read_text=lambda: json.dumps({"stage": "train", "checkpoint_sha256": digest}))
        with patch.object(runner.hashlib, "sha256", return_value=SimpleNamespace(hexdigest=lambda: digest)):
            with self.assertRaisesRegex(RuntimeError, "step2400"):
                runner.conditioning_checkpoint(checkpoint, receipt)
            receipt.read_text = lambda: json.dumps({"stage": "balanced-train-3", "checkpoint_sha256": digest})
            self.assertEqual(runner.conditioning_checkpoint(checkpoint, receipt), digest)

    def test_conditioning_preflight_exports_before_audit_is_allowed(self):
        stage = "conditioning-preflight"
        report = self.output / ("source-calibration-" + stage + ".json")
        report.write_text(json.dumps({"status": "passed", "bundle_sha256": self.bundle}))
        with self.assertRaises(FileNotFoundError):
            runner.require_export(self.output, stage, self.bundle, "passed")
        receipt = json.loads(self.receipt.read_text())
        receipt.update(stage=stage, report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
        (self.root / ("semflow-export-receipt-" + stage + ".json")).write_text(json.dumps(receipt))
        runner.require_export(self.output, stage, self.bundle, "passed")

    def test_realman_preflight_requires_its_own_export(self):
        stage = "realman-preflight"
        report = self.output / ("source-calibration-" + stage + ".json")
        report.write_text(json.dumps({"status": "passed", "bundle_sha256": self.bundle}))
        with self.assertRaises(FileNotFoundError):
            runner.require_export(self.output, stage, self.bundle, "passed")
        receipt = json.loads(self.receipt.read_text())
        receipt.update(stage=stage, report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
        destination = self.root / ("semflow-export-receipt-" + stage + ".json")
        destination.write_text(json.dumps(receipt))
        runner.require_export(self.output, stage, self.bundle, "passed")
        receipt["bundle_sha256"] = "unreviewed"
        destination.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, "receipt differs"):
            runner.require_export(self.output, stage, self.bundle, "passed")
