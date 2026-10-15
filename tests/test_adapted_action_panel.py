"""Paired sampling, fixed-state and evaluation coverage contracts."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import probe_adapted_actions as panel
from test_balanced_source import fixture
from semflow.source_batching import source_window
TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from torch import nn

@unittest.skipUnless(TORCH,'optional torch required')
class ActionPanelTests(unittest.TestCase):
    def make_report(self,condition='frozen',mutate=False):
        source=fixture()
        records=[{'record':r,'anchor':20} for task in source.tasks
                 for r in [r for r in source.index['manifest']['episodes'] if r['task']==task and r['split']=='validation'][:2]]
        windows=[source_window(source.episode(r['record']),r['anchor']) for r in records]
        test=self
        class Policy(nn.Module):
            def __init__(self):
                super().__init__();self.value=nn.Parameter(torch.tensor(.2));self.register_buffer('cache',torch.tensor(0.))
            def encode_observations(self,**kwargs):return None
            def sample_actions(self,observation,noise,*,num_steps):
                if mutate: self.cache.add_(1)
                test.assertEqual(noise.dtype,torch.float32)
                return noise*self.value+num_steps*.001
        def processor(windows,device):
            test.assertNotIn('loss_targets',windows[0]);return {}
        report=dict(condition=condition,request={},panel_config=panel.PANEL_CONFIG,
                    initial_adapter_sha256='common',processed_layouts=[])
        panel.sample_panel(Policy(),processor,windows,records,source.index['normalization'],report,lambda:None,device='cpu')
        report['status']='passed'
        return report

    def test_sampling_pairing_decomposition_and_no_state_changes(self):
        rng=torch.get_rng_state().clone()
        reports={arm:self.make_report(arm) for arm in panel.ARMS}
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        result=panel.compare_panels(reports)
        self.assertTrue(result['paired'])
        for steps in ('5','10','20'):
            self.assertEqual(result['deltas'][steps]['full']['semflow_minus_frozen'],0)
            full=result['summaries']['frozen'][steps]['all']['full']
            self.assertAlmostEqual(full['sampled_mse'],full['mean_prediction_mse']+full['seed_variance'])
        self.assertEqual(len(reports['frozen']['panel']),12)
        self.assertEqual(len(reports['frozen']['panel'][0]['samples']),12)

    def test_incomplete_noise_or_window_pairing_rejected(self):
        report=self.make_report()
        incomplete=copy.deepcopy(report);incomplete['panel'][0]['samples'].pop()
        with self.assertRaises(ValueError):panel.summarize(incomplete['panel'])
        reports={arm:dict(copy.deepcopy(report),condition=arm) for arm in panel.ARMS}
        reports['semflow']['panel'][0]['samples'][0]['initial_noise_sha256']='different'
        with self.assertRaisesRegex(ValueError,'differ'):panel.compare_panels(reports)
        reports={arm:dict(copy.deepcopy(report),condition=arm) for arm in panel.ARMS}
        reports['standard']['panel'][0]['normalized_target'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'differ'):panel.compare_panels(reports)

    def test_mutated_model_buffer_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'state'):self.make_report(mutate=True)
