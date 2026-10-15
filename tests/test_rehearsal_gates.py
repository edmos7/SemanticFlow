"""Cross-arm and durable-export gates for the bounded source rehearsal."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path: sys.path.insert(0,str(TOOLS))
from rehearse_adaptation import validate_matched_reference
from run_source_calibration_cloud import REHEARSAL_STAGES, verified_comparison

class RehearsalGateTests(unittest.TestCase):
    def test_matched_reference_rejects_each_changed_contract_field(self):
        report={k:k for k in ('request','plan','initial_adapter_sha256','teacher_cache_sha256','semantic_calibration','trainable_parameters')}
        reference=dict(report,status='passed',condition='standard',mode='train')
        validate_matched_reference(report,reference)
        for key in reference:
            changed=copy.deepcopy(reference);changed[key]='different'
            with self.assertRaises(ValueError): validate_matched_reference(report,changed)

    def test_receipt_covers_inner_report_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'results';(output/'calibration').mkdir(parents=True)
            stage=REHEARSAL_STAGES[1];bundle='pinned-bundle'
            inner_path=output/'calibration'/(stage+'.json');checkpoint=inner_path.with_suffix('.pt')
            checkpoint.write_bytes(b'checkpoint')
            sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            inner={'status':'passed','checkpoint':{'sha256':sha(checkpoint),'bytes':checkpoint.stat().st_size}}
            inner_path.write_text(json.dumps(inner))
            outer_path=output/('source-calibration-'+stage+'.json')
            outer_path.write_text(json.dumps({'status':'passed','bundle_sha256':bundle,'comparison_report_sha256':sha(inner_path)}))
            receipt={'stage':stage,'bundle_sha256':bundle,'report_sha256':sha(outer_path),
                     'comparison_report_sha256':sha(inner_path),'checkpoint_sha256':sha(checkpoint)}
            (root/('semflow-export-receipt-'+stage+'.json')).write_text(json.dumps(receipt))
            self.assertEqual(verified_comparison(output,stage,bundle),inner)
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'checkpoint'): verified_comparison(output,stage,bundle)
            checkpoint.write_bytes(b'checkpoint');inner_path.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'export'): verified_comparison(output,stage,bundle)
