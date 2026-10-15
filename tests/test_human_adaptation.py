"""Human plan, task reset and missing-modality loss boundaries."""
import copy
from pathlib import Path
import sys
import unittest
import torch
from semflow.human_adaptation_plan import human_plan, HUMAN_CONFIG
from semflow.human_batching import human_indices
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from adapt_human import reset_adapters, validate_pair
import test_adaptation_rehearsal as rehearsal_fixture
from semflow.models.adaptation import fixed_forecast_targets,forecast_objective


class HumanPlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'human_clips':[{'target_id':str(i),'sha256':str(i)*64,
            'timestamps_seconds':[j/30 for j in range(300)]} for i in range(3)]}

    def test_plan_spans_clip_with_causal_support_and_distinct_anchors(self):
        first=human_plan(self.manifest)
        self.manifest['human_clips'].reverse()
        self.assertEqual(first,human_plan(self.manifest))
        for clip in self.manifest['human_clips']:
            anchors=first[clip['target_id']]['anchors']
            self.assertEqual(len(set(anchors)),12)
            self.assertLess(anchors[0],35);self.assertGreater(anchors[-1],230)
            for a in anchors:
                past,future=human_indices(clip['timestamps_seconds'],a)
                self.assertLessEqual(max(past),a);self.assertGreater(min(future),a)
        self.assertEqual(HUMAN_CONFIG['human_evaluation'],'training-window fit only')

    def test_missing_future_coverage_and_duplicate_tasks_fail(self):
        self.manifest['human_clips'][0]['timestamps_seconds']=[j/30 for j in range(60)]
        with self.assertRaisesRegex(ValueError,'insufficient'):human_plan(self.manifest)
        self.manifest['human_clips'][0]=self.manifest['human_clips'][1]
        with self.assertRaisesRegex(ValueError,'distinct'):human_plan(self.manifest)

    def test_task_reset_restores_exact_initial_values_and_clears_gradients(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
        initial={'adapter':parameter.detach().clone()}
        optimizer=torch.optim.AdamW([parameter],lr=.1)
        parameter.sum().backward();optimizer.step()
        self.assertFalse(torch.equal(parameter,initial['adapter']))
        reset_adapters([('adapter',parameter)],initial)
        torch.testing.assert_close(parameter,initial['adapter'],rtol=0,atol=0)
        self.assertIsNone(parameter.grad)
        with self.assertRaisesRegex(ValueError,'scope'):reset_adapters([('adapter',parameter)],{})
        with self.assertRaisesRegex(ValueError,'invalid'):reset_adapters([('adapter',parameter)],{'adapter':torch.tensor([float('nan'),1.])})

    def test_matching_rejects_changed_labels_initialization_or_budget(self):
        keys=('request','human_request','plan','source_plan','initial_adapter_sha256',
              'teacher_cache_sha256','semantic_calibration','trainable_parameters')
        report={k:k for k in keys}
        reference={**report,'status':'passed','condition':'standard','mode':'train'}
        validate_pair(report,reference)
        for key in keys:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'differs'):
                validate_pair({**report,key:'changed'},reference)


class HumanLossTests(unittest.TestCase):
    def test_absent_head_targets_do_not_change_loss_or_gradient(self):
        fixture=rehearsal_fixture.FixedTargetTests();fixture.setUp()
        inputs={**fixture.inputs,'view_valid':torch.tensor([[False,True]])}
        observation=fixture.model.encode_observations(**inputs)
        labels=fixed_forecast_targets(fixture.model,observation,fixture.future)
        changed=copy.deepcopy(labels);changed['raw'][:,0]=1e6
        a,_=forecast_objective(observation,labels,fixture.stats,condition='semflow',semantic_weight=.1)
        b,_=forecast_objective(observation,changed,fixture.stats,condition='semflow',semantic_weight=.1)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a,fixture.pixels,retain_graph=True)[0]
        gb=torch.autograd.grad(b,fixture.pixels)[0]
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)
        self.assertGreater(ga.norm().item(),0)



class HumanRunnerTests(unittest.TestCase):
    def test_task_isolation_and_fresh_process_recovery(self):
        # Exercise the real bounded orchestration/serialization with a tiny CPU
        # model. Native vision/loss accuracy is covered by the GPU run, not this fixture.
        from contextlib import ExitStack, nullcontext
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import numpy as np
        import adapt_human
        import semflow.models.adaptation as objectives
        import semflow.source_batching as batching
        import semflow.human_batching as human_batching
        import train_source_calibration as training

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.adapter=torch.nn.Parameter(torch.tensor([.2]))
                self.register_buffer('frozen',torch.tensor([7.]))
            def encode_observations(self, x):return self.adapter*x
        def load(*a,**k):
            p=Policy()
            processor=lambda rows,device:{'x':torch.tensor([rows[0]['x']])}
            class HumanProcessor:
                image_size=(4,4)
                def __call__(self,rows,device):return processor(rows,device)
            return p,processor,[('adapter',p.adapter)],None,HumanProcessor()
        source={g:[{'record':{'task':str(i),'split':'validation' if g=='validation' else 'train'},'anchor':20}
                   for i in range(12)] for g in ('adaptation','retention','validation')}
        plan={str(i):{'anchors':list(range(12))} for i in range(1,4)}
        manifest={'human_clips':[{'target_id':t} for t in plan]}
        def source_window(record,anchor):
            return {'x':1.,'provenance':{'split':record['split']},'loss_targets':{
                'future_rgb':np.zeros((2,3,4,4,3),dtype=np.uint8),'actions':np.zeros((32,7),dtype=np.float64)}}
        def human_window(root,clip,anchor,**kw):
            return {'observations':{'x':float(clip['target_id'])},'provenance':{'anchor':anchor},
                    'loss_targets':{'future_rgb':np.zeros((3,4,4,3),dtype=np.uint8)}}
        def forecast(observation,targets,statistics,*,condition,semantic_weight):
            raw=(observation-targets['raw']-1).square().mean();semantic=2*raw
            return .1*raw+(semantic_weight*semantic if condition=='semflow' else 0),{
                'raw':raw,'semantic':semantic,'state':raw,'transition':raw}
        source_memory=SimpleNamespace(handles={},close=lambda:None)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            for target,name,value in (
                (adapt_human,'validate_inputs',lambda args:({}, {'source_lock':{}}, {},source,{},manifest,plan)),
                (adapt_human,'load_common',load),
                (training,'MemorySources',lambda lock:source_memory),
                (batching,'SourceWindowSampler',lambda *a,**k:SimpleNamespace(episode=lambda record:record)),
                (batching,'source_window',source_window),
                (human_batching,'read_human_window',human_window),
                (objectives,'fixed_forecast_targets',lambda p,o,r:{'raw':o.detach().clone()}),
                (objectives,'forecast_objective',forecast),
                (objectives,'source_action_retention',lambda p,o,a,seed:.01*(o+.1).square().mean()),
                (torch.cuda,'is_available',lambda:True),
                (torch.cuda,'is_current_stream_capturing',lambda:False),
                (torch.cuda,'manual_seed_all',lambda *a:None),
                (torch.cuda,'get_rng_state_all',lambda:[]),
                (torch.cuda,'set_rng_state_all',lambda state:None),
                (torch.Tensor,'cuda',lambda self:self),
                (torch,'autocast',lambda *a,**k:nullcontext()),
                (adapt_human.native_probe,'memory',lambda:{})):
                stack.enter_context(patch.object(target,name,value))
            import contextlib,io
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            root=Path(folder)
            frozen=[]
            for condition in ('standard','semflow'):
                args=SimpleNamespace(condition=condition,mode='train',data_root=root,output=root/(condition+'.json'),
                    reference_report=root/'standard.json' if condition=='semflow' else None)
                report={};adapt_human.run(args,report)
                self.assertEqual(report['optimizer_steps'],180)
                self.assertEqual(report['status'],'passed')
                self.assertTrue(all(r['initial_adapter_sha256']==report['initial_adapter_sha256'] for r in report['tasks'].values()))
                args.output.write_text(json.dumps(report))
                args.arm_checkpoint=args.output.with_suffix('.pt');args.arm_checkpoint_sha256=adapt_human.sha(args.arm_checkpoint)
                args.mode='evaluate';args.output=root/(condition+'-evaluate.json')
                evaluated={};adapt_human.run(args,evaluated)
                self.assertEqual(evaluated['optimizer_steps'],0)
                self.assertTrue(all(r['checkpoint_roundtrip_equal'] for r in evaluated['tasks'].values()))
                frozen.append({t:r['frozen'] for t,r in evaluated['tasks'].items()})
            self.assertEqual(frozen[0],frozen[1])

if __name__=='__main__':unittest.main()
