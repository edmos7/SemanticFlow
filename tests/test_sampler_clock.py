"""Clock traces and intervention isolation; no GPU/model claim from fixtures."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from probe_sampler_clock import accumulated_sample,compare_traces,capture_flow

class ClockTests(unittest.TestCase):
    def test_accumulated_float32_clock_and_input_noise_preserved(self):
        class Policy:
            def __init__(self):self.times=[]
            def predict_velocity(self,o,a,t,**kw):self.times.append(t.clone());return torch.ones_like(a)
        p=Policy();noise=torch.zeros(1,32,7);out=accumulated_sample(p,None,noise,10)
        torch.testing.assert_close(out,torch.ones_like(out));self.assertFalse(noise.any())
        t=torch.zeros(1)
        for observed in p.times:
            torch.testing.assert_close(observed,t[:,None].expand(1,32),rtol=0,atol=0);t=t+.1
        self.assertNotEqual(p.times[7][0,0].item(),torch.tensor(.7).item())

    def test_trace_comparison_records_failed_magnitude_before_gate(self):
        a={k:[torch.zeros(1)] for k in ('times','embedding','noisy','velocity')}
        b={k:[v[0].clone()] for k,v in a.items()};b['velocity'][0]+=1.
        result=compare_traces(a,b,1)
        self.assertEqual(result[0]['velocity_max_abs'],1.)
        with self.assertRaisesRegex(ValueError,'incomplete'):compare_traces(a,b,2)

    def test_capture_hooks_are_removed_after_exception(self):
        from types import SimpleNamespace
        p=SimpleNamespace(native=SimpleNamespace(time_embedding=torch.nn.Linear(1,1)),
                          panda_action_in=torch.nn.Linear(7,7),panda_action_out=torch.nn.Linear(7,7))
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with capture_flow(p) as trace:
                p.native.time_embedding(torch.zeros(1,1));p.panda_action_in(torch.ones(1,7));p.panda_action_out(torch.ones(1,7))
                self.assertEqual({k:len(v) for k,v in trace.items()},dict.fromkeys(trace,1))
                raise RuntimeError('fixture')
        self.assertFalse(p.native.time_embedding._forward_hooks);self.assertFalse(p.native.time_embedding._forward_pre_hooks)
        self.assertFalse(p.panda_action_in._forward_pre_hooks);self.assertFalse(p.panda_action_out._forward_hooks)

if __name__=='__main__':unittest.main()
