"""Continuation boundaries, exact restart, frozen panels and source-only stops."""
import copy
import importlib.util
import random
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from continue_realman_source import (ContinuationSchedule, stage_bounds, STAGES, stop_decision,
                                    validate_saved, CHECKPOINT_SHA256)
from compare_source_initializations import paired_flow, assert_nested_equal
from train_balanced_source import BalancedSchedule
from test_balanced_source import fixture
from test_source_initializations import panels
from run_source_calibration_cloud import CONTINUATION_STAGES
TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from semflow.models.source_calibration import checkpoint_payload, restore_checkpoint

class ContinuationTests(unittest.TestCase):
    def test_schedule_preserves_prefix_and_resume_across_old_limit(self):
        source=fixture(); old=BalancedSchedule(source); new=ContinuationSchedule(source)
        for _ in range(120):
            self.assertEqual(old.next_record(),new.next_record());old.updates+=1;new.updates+=1
        clone=ContinuationSchedule(source);clone.load_state_dict(old.state_dict())
        counts=dict.fromkeys(new.tasks,0)
        for _ in range(2280):
            self.assertEqual(new.next_record(),clone.next_record())
            counts[new.next_record()[0]['task']]+=1;new.updates+=1;clone.updates+=1
        self.assertEqual(set(counts.values()),{380})
        with self.assertRaises(ValueError):new.next_record()
        with self.assertRaises(ValueError):old.load_state_dict(new.state_dict())
        for updates in (True,-1,2401):
            with self.assertRaises(ValueError):clone.load_state_dict(dict(new.state_dict(),updates=updates))

    def test_only_reviewed_stages_and_matched_input(self):
        self.assertEqual(STAGES,CONTINUATION_STAGES)
        self.assertEqual([stage_bounds(s) for s in STAGES[1:]],
            [('recover',120,120),('train',120,600),('panel',600,600),('train',600,1200),
             ('panel',1200,1200),('train',1200,2400),('panel',2400,2400)])
        with self.assertRaises(ValueError):stage_bounds('continuation-train-3000')
        source=fixture();schedule=ContinuationSchedule(source);schedule.updates=120
        saved=dict(step=120,sampler=schedule.state_dict())
        validate_saved(saved,CHECKPOINT_SHA256,source,120)
        with self.assertRaises(ValueError):validate_saved(saved,'wrong',source,120)
        with self.assertRaises(ValueError):validate_saved(dict(saved,step=0),CHECKPOINT_SHA256,source,120)

    def test_stops_stall_regression_budget_and_changed_panel(self):
        baseline=dict(panel=panels(),panel_layouts=[{}]*24)
        self.assertFalse(stop_decision(baseline,baseline,baseline,600)['continue_training'])
        improved=copy.deepcopy(baseline)
        for row in improved['panel']:
            target=np.asarray(row['normalized_target'])
            for sample in row['samples']:
                sample['normalized_prediction']=(target+.8*(np.asarray(sample['normalized_prediction'])-target)).tolist()
        self.assertTrue(stop_decision(improved,baseline,baseline,600)['continue_training'])
        self.assertEqual(stop_decision(improved,baseline,baseline,2400)['reason'],'budget_complete')
        self.assertEqual(stop_decision(baseline,improved,improved,1200)['reason'],'validation_regression')
        improved['panel'][0]['anchor']=999
        with self.assertRaises(ValueError):stop_decision(improved,baseline,baseline,600)

@unittest.skipUnless(TORCH,'PyTorch required')
class RecoveryTests(unittest.TestCase):
    def test_flow_continues_absolute_seed_and_preserves_global_rng(self):
        x=torch.zeros(1,32,7);before=torch.get_rng_state().clone()
        for a,b in zip(paired_flow(x,120),paired_flow(x,120,max_updates=2400)):
            self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(paired_flow(x,120)[0],paired_flow(x,121,max_updates=2400)[0]))
        paired_flow(x,2400,max_updates=2400)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with self.assertRaises(ValueError):paired_flow(x,2401,max_updates=2400)

    def test_real_checkpoint_api_restarts_optimizer_rng_and_schedule_exactly(self):
        source=fixture()
        def setup():
            policy=torch.nn.Linear(2,2); reconstruction=torch.nn.Linear(2,2)
            opt=torch.optim.AdamW([{'params':list(policy.parameters()),'lr':.0003},
                                  {'params':list(reconstruction.parameters()),'lr':.00005}],foreach=False)
            return policy,reconstruction,opt,ContinuationSchedule(source)
        def step(parts):
            p,r,o,s=parts;s.updates+=1;o.zero_grad()
            loss=r(p(torch.randn(2,2))).square().sum()*(random.random()+np.random.rand())
            loss.backward();o.step()
        # Isolate CUDA so this exercises the same CPU state API on any test host.
        with patch('torch.cuda.is_available',return_value=False):
            original=setup()
            for _ in range(120):step(original)
            saved=checkpoint_payload(*original,step=120,contract={'test':True})
            for _ in range(6):step(original)
            expected=checkpoint_payload(*original,step=126,contract={'test':True})
            resumed=setup()
            restore_checkpoint(saved,*resumed,contract={'test':True})
            assert_nested_equal(saved,checkpoint_payload(*resumed,step=120,contract={'test':True}))
            for _ in range(6):step(resumed)
            assert_nested_equal(expected,checkpoint_payload(*resumed,step=126,contract={'test':True}))

if __name__=='__main__':unittest.main()
