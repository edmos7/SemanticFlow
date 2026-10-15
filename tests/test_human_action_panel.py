"""Task-specific restoration and paired action evaluation after human training."""
import copy
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import probe_human_actions as human
import test_adapted_action_panel as action_fixture
from train_source_calibration import tensor_digest
from semflow.human_adaptation_plan import HUMAN_CONFIG


class HumanActionPanelTests(unittest.TestCase):
    def fixture(self):
        targets=['a','b','c'];tasks={'targets':[{'id':t} for t in targets]}
        reports={};evaluations={}
        for arm in ('standard','semflow'):
            common={'request':{},'human_request':{'config':HUMAN_CONFIG},'plan':dict.fromkeys(targets,{}),
                    'source_plan':{},'initial_adapter_sha256':'common','teacher_cache_sha256':'teacher',
                    'semantic_calibration':{},'trainable_parameters':{},'frozen_state_unchanged':True,
                    'teacher_targets_unchanged':True,'status':'passed','condition':arm}
            reports[arm]={**common,'mode':'train','optimizer_steps':180,'tasks':{
                t:{'optimizer_steps':60,'initial_adapter_sha256':'common'} for t in targets}}
            evaluations[arm]={**common,'mode':'evaluate','optimizer_steps':0,'tasks':{
                t:{'checkpoint_roundtrip_equal':True,'recovered_training_steps':60,'frozen':{'value':1}} for t in targets}}
        return tasks,reports,evaluations

    def test_training_and_fresh_recovery_required_for_every_task(self):
        tasks,reports,evaluations=self.fixture()
        self.assertEqual(human.validate_training_pair({}, {},tasks,reports,evaluations),['a','b','c'])
        bad=copy.deepcopy(evaluations);bad['semflow']['tasks']['b']['checkpoint_roundtrip_equal']=False
        with self.assertRaisesRegex(ValueError,'every task'):human.validate_training_pair({}, {},tasks,reports,bad)
        bad=copy.deepcopy(reports);bad['standard']['tasks']['c']['initial_adapter_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'every task'):human.validate_training_pair({}, {},tasks,bad,evaluations)
        bad=copy.deepcopy(evaluations);bad['semflow']['tasks']['c']['frozen']={'value':2}
        with self.assertRaisesRegex(ValueError,'different frozen'):human.validate_training_pair({}, {},tasks,reports,bad)

    def test_each_task_loads_exact_adapters_and_swapped_payload_fails(self):
        values={t:{'adapter':torch.tensor([float(i)])} for i,t in enumerate(('a','b','c'))}
        training={k:{} for k in ('request','human_request','semantic_calibration')}
        training.update(condition='standard',initial_adapter_sha256='base',teacher_cache_sha256='fixed',
                        tasks={t:{'adapter_sha256':tensor_digest(v)} for t,v in values.items()})
        saved={'schema_version':1,'contract':{k:v for k,v in training.items() if k!='tasks'},
               'tasks':{t:{'step':60,'adapters':v} for t,v in values.items()}}
        parameter=torch.nn.Parameter(torch.tensor([99.]))
        for t in ('c','a','b'):
            human.restore_task([('adapter',parameter)],saved,training,t)
            torch.testing.assert_close(parameter,values[t]['adapter'],rtol=0,atol=0)
        bad=copy.deepcopy(saved);bad['tasks']['a']=bad['tasks']['b']
        with self.assertRaisesRegex(ValueError,'identity'):human.restore_task([('adapter',parameter)],bad,training,'a')
        bad=copy.deepcopy(saved);bad['contract']['teacher_cache_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'contract'):human.restore_task([('adapter',parameter)],bad,training,'a')

    def test_all_six_models_pair_with_one_frozen_panel(self):
        fixture=action_fixture.ActionPanelTests()
        import contextlib,io
        with contextlib.redirect_stdout(io.StringIO()):base=fixture.make_report()
        frozen={'status':'passed','condition':'frozen','panel_request':{},'panels':{'frozen':base}}
        targets=['a','b','c'];arms={}
        for arm in ('standard','semflow'):
            arms[arm]={'status':'passed','condition':arm,'panel_request':{},'panels':{
                t:{**copy.deepcopy(base),'condition':arm,'adaptation_target':t} for t in targets}}
        result=human.compare_human_panels(frozen,arms['standard'],arms['semflow'],targets)
        self.assertEqual(set(result['by_human_adaptation']),set(targets));self.assertTrue(result['paired'])
        bad=copy.deepcopy(arms['semflow']);bad['panels']['a']['adaptation_target']='b'
        with self.assertRaisesRegex(ValueError,'target pairing'):human.compare_human_panels(frozen,arms['standard'],bad,targets)
        bad=copy.deepcopy(arms['semflow']);bad['panels']['a']['panel'][0]['samples'][0]['initial_noise_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'noise'):human.compare_human_panels(frozen,arms['standard'],bad,targets)
        bad=copy.deepcopy(arms['standard']);del bad['panels']['c']
        with self.assertRaisesRegex(ValueError,'incomplete'):human.compare_human_panels(frozen,bad,arms['semflow'],targets)

if __name__=='__main__':unittest.main()
