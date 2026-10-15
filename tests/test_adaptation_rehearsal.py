"""Fixed-target, paired-retention and source-only planning contracts."""

import copy
import hashlib
import importlib.util
import unittest

from semflow.adaptation_plan import source_plan, calibrated_semantic_weight

TORCH=importlib.util.find_spec('torch') is not None
if TORCH:
    import torch
    from semflow.models.native_policy import NativeSemFlowPolicy
    from semflow.models.losses import SourceFeatureStatistics
    from semflow.models.adaptation import fixed_forecast_targets,forecast_objective,source_action_retention
    from test_native_policy import FakeNativePolicy, TinyTeacher

    class AlignedTeacher(TinyTeacher):
        def forward(self, images):
            return super().forward(images).repeat_interleave(4, dim=2)


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.index={'manifest':{'episodes':[],'human_clips':[]},'coverage':{},'provenance':{}}
        for task in range(6):
            self.index['coverage'][str(task)]={}
            for i in range(6):
                h=hashlib.sha256(f'{task}/{i}'.encode()).hexdigest()
                self.index['manifest']['episodes'].append({'task':str(task),'split':'train' if i<4 else 'validation','sha256':h})
                self.index['provenance'][h]={'aligned_frames':100+i}

    def test_disjoint_complete_deterministic_plan(self):
        first=source_plan(self.index)
        self.index['manifest']['episodes'].reverse()
        self.assertEqual(source_plan(self.index),first)
        self.assertEqual([len(v) for v in first.values()],[12,12,12])
        groups=[{r['record']['sha256'] for r in rows} for rows in first.values()]
        self.assertEqual(len(set.union(*groups)),36)
        self.assertTrue(all(r['record']['split']=='train' for r in first['adaptation']+first['retention']))

    def test_human_data_and_missing_task_coverage_rejected(self):
        self.index['manifest']['human_clips']=[{}]
        with self.assertRaisesRegex(ValueError,'human'): source_plan(self.index)
        self.index['manifest']['human_clips']=[]
        del self.index['coverage']['5']
        with self.assertRaises(ValueError): source_plan(self.index)

    def test_calibration_rejects_nonfinite_or_incomplete_pairs(self):
        pairs=[{'raw':.2,'semantic':2.} for _ in range(6)]
        self.assertAlmostEqual(calibrated_semantic_weight(pairs)['weight'],.1)
        with self.assertRaises(ValueError): calibrated_semantic_weight(pairs[:5])
        pairs[0]['semantic']=float('nan')
        with self.assertRaises(ValueError): calibrated_semantic_weight(pairs)


@unittest.skipUnless(TORCH,'optional torch required')
class FixedTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(99)
        self.native=FakeNativePolicy()
        self.native.frozen_teacher=AlignedTeacher()
        self.model=NativeSemFlowPolicy(self.native,scene_dim=8).requires_grad_(False).eval()
        self.pixels=torch.randn(1,6,3,requires_grad=True)
        self.inputs={'batch':{'input_ids':torch.tensor([[1,2,3,4]]),'attention_mask':torch.ones(1,4,dtype=torch.long),
                             'answer_start_idx':torch.tensor([4]),'pixel_values':self.pixels},
                     'current_rgb':torch.randint(0,256,(1,2,4,4,3),dtype=torch.uint8),
                     'view_valid':torch.ones(1,2,dtype=torch.bool),
                     'observation_times':torch.tensor([[-1.,-.8,-.6,-.4,-.2,0.]]),'forecast_origin':torch.zeros(1)}
        self.observation=self.model.encode_observations(**self.inputs)
        self.future=torch.randint(0,256,(1,2,3,4,4,3),dtype=torch.uint8)
        self.targets=fixed_forecast_targets(self.model,self.observation,self.future)
        self.stats=SourceFeatureStatistics(torch.zeros(8),torch.ones(8),'source-train-fixture')

    def test_teacher_stays_fixed_when_task_context_changes(self):
        before={k:v.clone() for k,v in self.targets.items()}
        with torch.no_grad(): self.native.pixel_projection.weight.add_(.5)
        changed=self.model.encode_observations(**self.inputs)
        alternate=fixed_forecast_targets(self.model,changed,self.future)
        self.assertGreater((alternate['scene']-before['scene']).abs().max().item(),0)
        for k,v in before.items(): torch.testing.assert_close(self.targets[k],v,rtol=0,atol=0)
        self.assertTrue(all(not v.requires_grad for v in self.targets.values()))

    def test_matched_raw_objective_and_prediction_gradients(self):
        ordinary,parts=forecast_objective(self.observation,self.targets,self.stats,condition='standard',semantic_weight=.2)
        semantic,_=forecast_objective(self.observation,self.targets,self.stats,condition='semflow',semantic_weight=.2)
        torch.testing.assert_close(ordinary,.1*parts['raw'])
        torch.testing.assert_close(semantic-ordinary,.2*parts['semantic'])
        grad=torch.autograd.grad(semantic,self.pixels)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_grad_bearing_targets_and_view_mismatch_rejected(self):
        changed={**self.targets,'scene':self.targets['scene'].clone().requires_grad_()}
        with self.assertRaisesRegex(ValueError,'detached'):
            forecast_objective(self.observation,changed,self.stats,condition='semflow',semantic_weight=.1)
        changed={**self.targets,'view_valid':torch.tensor([[True,False]])}
        with self.assertRaisesRegex(ValueError,'availability'):
            forecast_objective(self.observation,changed,self.stats,condition='semflow',semantic_weight=.1)

    def test_retention_noise_is_paired_without_global_rng_draw(self):
        actions=torch.randn(1,32,7)
        before=torch.get_rng_state().clone()
        a=source_action_retention(self.model,self.observation,actions,seed=2)
        b=source_action_retention(self.model,self.observation,actions,seed=2)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertNotEqual(a.item(),source_action_retention(self.model,self.observation,actions,seed=3).item())


if __name__=='__main__': unittest.main()
