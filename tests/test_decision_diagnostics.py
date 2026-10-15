"""Diagnostic decision rules, forecast pairing and controller normalization."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import decision_diagnostics as diagnostic
from semflow.source_data import normalize

class DiagnosticTests(unittest.TestCase):
    def test_normalization_detects_shifted_raw_actions(self):
        raw=np.arange(60*7,dtype=np.float64).reshape(60,7)/1000
        stats={'mean':[.2]*7,'scale':[.3]*7}
        window={'provenance':{'anchor':20},'loss_targets':{'actions':normalize(raw[21:53],stats)}}
        handle={'data/demo':{'actions':raw}}
        result=diagnostic.normalization_audit(window,{'episode_key':'data/demo'},handle,{'normalization':{'action':stats}})
        self.assertLess(result['roundtrip_max_abs'],1e-12)
        window['loss_targets']['actions']=normalize(raw[20:52],stats)
        with self.assertRaisesRegex(ValueError,'alignment'):diagnostic.normalization_audit(window,{'episode_key':'data/demo'},handle,{'normalization':{'action':stats}})

    def test_bypass_hook_is_removed_even_after_failure(self):
        from types import SimpleNamespace
        conditioner=torch.nn.Linear(7,7)
        x=torch.ones(1,2,7);before=conditioner(x).detach().clone()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with diagnostic.bypass_conditioner(SimpleNamespace(conditioner=conditioner)):
                torch.testing.assert_close(conditioner(x),x)
                raise RuntimeError('fixture')
        torch.testing.assert_close(conditioner(x),before)
        self.assertFalse(conditioner._forward_hooks)

    def fixture(self):
        panel={'summary':{task:{v:{'first_eight':{'sampled_mse':1.,'paired_action_rms':0.}} for v in diagnostic.DIAGNOSTIC_CONFIG['interventions']}
                           for task in ['all']+[str(i) for i in range(6)]}}
        audit={'status':'passed','ablations':copy.deepcopy(panel)}
        retention={'status':'passed','panel_summary':{'10':{'all':{'first_eight':{'sampled_mse':1.}}}}}
        semflow={'status':'passed','ablations':{t:copy.deepcopy(panel) for t in ('a','b','c')}}
        previous={arm:{'panels':{t:copy.deepcopy(retention) for t in ('a','b','c')}} for arm in ('standard','semflow')}
        return audit,retention,semflow,previous,['a','b','c']

    def test_pause_when_effects_have_no_useful_action_gain(self):
        values=self.fixture()
        decision=diagnostic.decide(*values)
        self.assertEqual(decision['recommendation'],'pause_current_approach')
        for panel in values[2]['ablations'].values():
            for task in panel['summary'].values():
                task['static']['first_eight']['paired_action_rms']=10.
        self.assertEqual(diagnostic.decide(*values)['recommendation'],'pause_current_approach')

    def test_continue_requires_increment_or_both_useful_forecast_comparisons(self):
        values=self.fixture();values[3]['semflow']['panels']['a']['panel_summary']['10']['all']['first_eight']['sampled_mse']=.97
        values[3]['semflow']['panels']['b']['panel_summary']['10']['all']['first_eight']['sampled_mse']=.97
        self.assertTrue(diagnostic.decide(*values)['human_increment_passed'])
        values=self.fixture()
        for variant in ('static','shuffle'):
            for task in values[0]['ablations']['summary'].values():
                task[variant]['first_eight'].update(sampled_mse=1.02,paired_action_rms=.01)
        self.assertTrue(diagnostic.decide(*values)['useful_forecast_passed'])
        values[0]['ablations']['summary']['all']['shuffle']['first_eight']['sampled_mse']=1.
        self.assertFalse(diagnostic.decide(*values)['useful_forecast_passed'])

    def test_incomplete_evidence_cannot_produce_decision(self):
        values=self.fixture();values[1]['status']='failed'
        with self.assertRaisesRegex(ValueError,'complete'):diagnostic.decide(*values)

if __name__=='__main__':unittest.main()
