"""Matched source learning: randomness, seed decomposition, recovery and export gates."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import compare_source_initializations as comparison
from probe_source_actions import metrics
from run_source_calibration_cloud import COMPARISON_STAGES, verified_comparison

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from semflow.models.source_calibration import calibration_losses


def panels():
    rows=[]
    stats={"action":{"mean":[0]*7,"scale":[1]*7}}
    for split in ("train","validation"):
        for task in range(6):
            for window in range(2):
                target=np.ones((32,7))
                row=dict(task=str(task),split=split,anchor=window,episode_sha256=split+str(task)+str(window),
                         window_sha256=split+str(task)+str(window),normalized_target=target.tolist(),samples=[])
                for i,seed in enumerate(comparison.SEEDS):
                    result=metrics(np.full((32,7),i),target,stats)
                    row["samples"].append(dict(result,seed=seed,initial_noise_sha256=str(seed),integration_steps=10))
                rows.append(row)
    return rows


def reports():
    training=[dict(step=i,task=str(i%6),episode_sha256=str(i),split="train",anchor=20,
        window_sha256=str(i),flow_inputs_sha256=str(i),processed_layout={"ids":[1,20]}) for i in range(1,121)]
    panel=panels()
    return {(arm,mode):dict(status="passed",initialization={"sha":"same"},comparison_contract={"same":True},
        training=copy.deepcopy(training),panel=copy.deepcopy(panel),panel_layouts=[{"ids":[1,20]}]*24)
        for arm in ("base","realman") for mode in ("initial","train","final")}


class ComparisonTests(unittest.TestCase):
    def test_seed_decomposition_and_task_dimension_reporting(self):
        result=comparison.decomposition(panels())
        for split in result.values():
            for task in split.values():
                for horizon in task.values():
                    self.assertEqual(horizon["sampled_mse"],1.5)
                    self.assertEqual(horizon["mean_prediction_mse"],.25)
                    self.assertEqual(horizon["seed_variance"],1.25)
                    self.assertEqual(horizon["train_mean_baseline_mse"],1)
                    self.assertEqual(horizon["per_dimension_mse"],[1.5]*7)

    def test_rejects_incomplete_nonfinite_or_unpaired_panels(self):
        for mutation in (lambda rows:rows.pop(),
                         lambda rows:rows[0]["samples"][0].update(seed=-1),
                         lambda rows:rows[0]["samples"][0]["normalized_prediction"][0].__setitem__(0,float("nan"))):
            rows=panels(); mutation(rows)
            with self.assertRaises(ValueError): comparison.decomposition(rows)

    def test_comparison_rejects_confounds_and_accepts_exact_pairing(self):
        reference=reports()
        result=comparison.compare_reports(reference)
        self.assertTrue(result["matched"])
        self.assertEqual(result["paired"]["validation"]["first_eight"]["realman_minus_base_final"],0)
        for mutate in (
            lambda r:r[("realman","initial")]["initialization"].update(sha="different"),
            lambda r:r[("realman","train")]["training"][0].update(flow_inputs_sha256="different"),
            lambda r:r[("realman","final")]["panel"][0].update(anchor=99),
            lambda r:r[("realman","final")]["panel_layouts"].append({}),
            lambda r:r[("base","train")].update(status="failed"),
            lambda r:r.pop(("base","initial"))):
            changed=copy.deepcopy(reference); mutate(changed)
            with self.assertRaises(ValueError): comparison.compare_reports(changed)

    def test_export_receipt_pins_old_result_and_checkpoint(self):
        import hashlib
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); output=root/"results"; calibration=output/"calibration"; calibration.mkdir(parents=True)
            stage="comparison-base-initial"
            checkpoint=calibration/(stage+".pt"); checkpoint.write_bytes(b"checkpoint")
            inner=calibration/(stage+".json")
            inner.write_text(json.dumps(dict(status="passed",checkpoint=dict(bytes=10,sha256=sha(checkpoint)))))
            outer=output/("source-calibration-"+stage+".json")
            outer.write_text(json.dumps(dict(status="passed",bundle_sha256="bundle",comparison_report_sha256=sha(inner))))
            receipt=root/("semflow-export-receipt-"+stage+".json")
            receipt.write_text(json.dumps(dict(stage=stage,bundle_sha256="bundle",report_sha256=sha(outer),
                comparison_report_sha256=sha(inner),checkpoint_sha256=sha(checkpoint))))
            self.assertEqual(verified_comparison(output,stage,"bundle")["status"],"passed")
            checkpoint.write_bytes(b"tampered!!")
            with self.assertRaisesRegex(RuntimeError,"checkpoint"):
                verified_comparison(output,stage,"bundle")
            checkpoint.write_bytes(b"checkpoint"); inner.write_text(inner.read_text()+" ")
            with self.assertRaisesRegex(RuntimeError,"result"):
                verified_comparison(output,stage,"bundle")
        self.assertEqual(len(COMPARISON_STAGES),7)


@unittest.skipUnless(TORCH,"PyTorch is an optional cloud training dependency")
class ComparisonTorchTests(unittest.TestCase):
    def test_paired_flow_ignores_and_preserves_global_rng(self):
        actions=torch.zeros(1,32,7)
        before=torch.get_rng_state().clone()
        a=comparison.paired_flow(actions,1)
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        torch.randn(9)
        b=comparison.paired_flow(actions,1)
        for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertFalse(torch.equal(a[0],comparison.paired_flow(actions,2)[0]))
        for step in (0,121,True):
            with self.assertRaises(ValueError): comparison.paired_flow(actions,step)

    def test_explicit_flow_inputs_reach_objective_without_rng_draw(self):
        actions=torch.zeros(1,32,7); future=torch.zeros(1,2,3,2,2,3,dtype=torch.uint8)
        noise,time=comparison.paired_flow(actions,1)
        class Stop(Exception): pass
        def predict(observation,noisy,times):
            torch.testing.assert_close(noisy,(1-time)*noise)
            torch.testing.assert_close(times,time[:,:,0].expand(-1,32))
            raise Stop()
        before=torch.get_rng_state().clone()
        with self.assertRaises(Stop):
            calibration_losses(SimpleNamespace(predict_velocity=predict),None,None,future,actions,
                               weights={},flow_inputs=(noise,time))
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        for bad in ((noise,time+2),(noise[:,:,:6],time),(noise.double(),time),
                    (noise.requires_grad_(),time)):
            with self.assertRaisesRegex(ValueError,"paired flow"):
                calibration_losses(None,None,None,future,actions,weights={},flow_inputs=bad)

    def test_processor_strips_targets_and_fingerprints_layout(self):
        seen=[]
        def processor(windows,device):
            seen.extend(windows)
            return {"batch":{"input_ids":torch.ones(1,4,dtype=torch.long)}}
        wrapper=comparison.ObservedProcessor(processor)
        wrapper([{"loss_targets":{"privileged":True},"instruction":"source"}],"cpu")
        self.assertEqual(seen,[{"instruction":"source"}])
        self.assertEqual(wrapper.layouts[0]["input_ids"],[1,4])
        self.assertEqual(len(wrapper.layouts[0]["input_ids_sha256"]),64)

    def test_roundtrip_rejects_tensor_rng_and_optimizer_changes(self):
        a={"rng":[torch.tensor([1,2])],"optimizer":{"step":1}}
        comparison.assert_nested_equal(a,copy.deepcopy(a))
        for b in ({"rng":[torch.tensor([1,3])],"optimizer":{"step":1}},
                  {"rng":[torch.tensor([1,2])],"optimizer":{"step":0}}):
            with self.assertRaisesRegex(ValueError,"roundtrip"): comparison.assert_nested_equal(a,b)
