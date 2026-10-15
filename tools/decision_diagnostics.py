#!/usr/bin/env python3
"""Bounded implementation audit, forecast ablations and retention-only control."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import time
from types import FunctionType,SimpleNamespace
import numpy as np
import probe_egosteer as native_probe
from rehearse_adaptation import sha,load_common
from probe_human_actions import validate_inputs as validate_human_inputs,restore_task
from adapt_human import reset_adapters
from semflow.adaptation_plan import CONFIG
from probe_adapted_actions import sample_panel,panel_identity,PANEL_CONFIG
from probe_source_actions import metrics,window_digest,SEEDS
from compare_source_initializations import ObservedProcessor,assert_nested_equal
from semflow.models.sampling import accumulated_sample

DIAGNOSTIC_CONFIG={'integration_steps':10,'interventions':['none','zero','static','shuffle','bypass'],
    'audit_windows':6,'retention_updates':60,'normalization_atol':1e-10,'native_atol':1e-4,
    'reproduction_atol':1e-5,'primary_horizon':8,'minimum_relative_gain':.01,
    'minimum_action_rms':.001,'minimum_source_tasks':4,'minimum_human_models':2,
    'sampler_audit_clock':'native_accumulated','evaluation_clock':'historical_indexed'}


def validate_inputs(args):
    values=validate_human_inputs(args)
    spec=json.loads(args.diagnostic_request.read_text())
    if (set(spec)!={'config','panel_request_sha256','previous_panels'} or spec['config']!=DIAGNOSTIC_CONFIG
            or spec['panel_request_sha256']!=sha(args.panel_request)
            or set(spec['previous_panels'])!={'frozen','standard','semflow'}):
        raise ValueError('diagnostic declaration differs')
    previous={}
    for arm,pins in spec['previous_panels'].items():
        p=args.input_root/('previous-'+arm+'.json');receipt=args.input_root/('previous-'+arm+'-receipt.json')
        if set(pins)!={'report','receipt'} or sha(p)!=pins['report'] or sha(receipt)!=pins['receipt']:
            raise ValueError('previous panel digest differs')
        d=json.loads(p.read_text());r=json.loads(receipt.read_text())
        if (d['status']!='passed' or d['condition']!=arm or d['panel_request']!=values[4]
                or r['stage']!='humanaction-'+arm or r['comparison_report_sha256']!=pins['report']):
            raise ValueError('previous panel contract differs')
        previous[arm]=d
    return (*values,spec,previous)


@contextmanager
def bypass_conditioner(policy):
    hook=policy.conditioner.register_forward_hook(lambda module,arguments,output:arguments[0])
    try:yield
    finally:hook.remove()


@contextmanager
def native_panda_stream(policy,observation):
    """Use upstream flow implementation with the same trained Panda interfaces.

    These hooks expose the existing adapters/conditioner at native module
    boundaries; they do not replace the native expert, positions, time or solver.
    """
    import torch
    def condition(module,args,action):
        return policy.conditioner(action,observation.current_scene.to(action.dtype),
            observation.scene_forecasts.to(action.dtype),
            current_valid=torch.ones(observation.current_scene.shape[:2],dtype=torch.bool,device=action.device),
            future_valid=torch.ones(observation.scene_forecasts.shape[:2],dtype=torch.bool,device=action.device))
    hooks=[policy.native.action_encoder.register_forward_pre_hook(lambda m,a:(policy.panda_action_in(a[0]),)),
           policy.native.action_encoder.register_forward_hook(condition),
           policy.native.action_decoder.register_forward_hook(lambda m,a,o:policy.panda_action_out(o))]
    try:yield
    finally:
        for hook in reversed(hooks):hook.remove()


def native_sample(policy,observation,noise,steps,infer_function):
    """Execute pinned upstream sampler bytecode with supplied paired noise."""
    import torch
    class PairedTorch:
        def __getattr__(self,name):return getattr(torch,name)
        def randn(self,*shape,device,dtype):
            if tuple(shape)!=tuple(noise.shape) or noise.device!=device or noise.dtype!=dtype:
                raise ValueError('upstream initial noise contract differs')
            return noise.clone()
    infer=FunctionType(infer_function.__code__,{**infer_function.__globals__,'torch':PairedTorch()},
                       infer_function.__name__,infer_function.__defaults__,infer_function.__closure__)
    proxy=SimpleNamespace(action_dim=7,num_action_tokens=32,num_inference_steps=steps,
        build_slot_embeddings=lambda batch:{},forward_backbone_stream=lambda *a,**kw:observation.backbone,
        forward_flow_stream=policy.native.forward_flow_stream)
    batch={**observation.batch,'states':torch.zeros((len(noise),6,9),device=noise.device,dtype=noise.dtype),
           'actions_valid_mask':torch.ones_like(noise,dtype=torch.bool)}
    # states supplies only native sampler dtype here; prefix is already encoded
    # from actual measured states. No synthetic state reaches the backbone.
    with native_panda_stream(policy,observation):return infer(proxy,batch)


def normalization_audit(window,record,handle,index):
    from semflow.source_data import normalize
    from semflow.source_batching import controller_actions
    a=window['provenance']['anchor'];episode=handle[record['episode_key']]
    raw=np.asarray(episode['actions'][a+1:a+33],dtype=np.float64)
    normalized=window['loss_targets']['actions'];stats=index['normalization']['action']
    independent=(raw-np.asarray(stats['mean']))/np.asarray(stats['scale'])
    restored=normalize(normalized,stats,inverse=True)
    commands,clipped=controller_actions(normalized,index['normalization'],-np.ones(7),np.ones(7))
    errors={'normalized_max_abs':float(np.max(np.abs(independent-normalized))),
            'roundtrip_max_abs':float(np.max(np.abs(restored-raw))),
            'controller_max_abs':float(np.max(np.abs(commands-np.clip(raw,-1,1))))}
    if max(errors.values())>DIAGNOSTIC_CONFIG['normalization_atol'] or clipped:
        raise ValueError('raw HDF5 action alignment/normalization/controller conversion differs')
    return {**window['provenance'],**errors,'raw_action_slice':[a+1,a+33],'clipped_target_values':clipped}


def refit_normalization(index,handles):
    from semflow.source_data import aligned_panda
    states,actions=[],[]
    for record in index['manifest']['episodes']:
        if record['split']!='train':continue
        state,action=aligned_panda(handles[record['task']][record['episode_key']])
        states.append(state);actions.append(action)
    errors={}
    for name,parts in (('state',states),('action',actions)):
        x=np.concatenate(parts);std=x.std(0);scale=np.where(std<1e-6,1.,std)
        stored=index['normalization'][name]
        errors[name]={key:float(np.max(np.abs(value-np.asarray(stored[key])))) for key,value in
            (('mean',x.mean(0)),('scale',scale),('minimum',x.min(0)),('maximum',x.max(0)))}
        if max(errors[name].values())>DIAGNOSTIC_CONFIG['normalization_atol']:
            raise ValueError('independent NumPy source-training normalization refit differs')
    return {'fit_split':'train','episodes':len(states),'max_absolute_errors':errors}


def ablation_summary(rows):
    if len(rows)!=12 or any(set(r['samples'])!=set(DIAGNOSTIC_CONFIG['interventions']) for r in rows):
        raise ValueError('incomplete ablation windows')
    for r in rows:
        for samples in r['samples'].values():
            if [s['seed'] for s in samples]!=list(SEEDS):raise ValueError('ablation seed pairing differs')
    result={}
    for group in ['all']+sorted({r['task'] for r in rows}):
        selected=[r for r in rows if group=='all' or r['task']==group]
        y=np.asarray([r['normalized_target'] for r in selected])
        normal=np.asarray([[s['normalized_prediction'] for s in r['samples']['none']] for r in selected])
        result[group]={}
        for variant in DIAGNOSTIC_CONFIG['interventions']:
            x=np.asarray([[s['normalized_prediction'] for s in r['samples'][variant]] for r in selected])
            if x.shape!=(len(selected),4,32,7) or not np.isfinite(x).all():raise ValueError('invalid ablation samples')
            result[group][variant]={}
            for label,n in (('full',32),('first_eight',8)):
                error=float(np.square(x[:,:,:n]-y[:,None,:n]).mean())
                effect=float(np.square(x[:,:,:n]-normal[:,:,:n]).mean())**.5
                result[group][variant][label]={'sampled_mse':error,'paired_action_rms':effect}
    return result


def run_ablations(policy,processor,windows,records,normalization,reference,report,persist):
    import torch
    from train_source_calibration import tensor_digest
    policy.requires_grad_(False).eval();before=tensor_digest(policy.state_dict())
    rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
    report['rows']=[];observed=ObservedProcessor(processor)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,(window,record) in enumerate(zip(windows,records,strict=True)):
            prior=reference['panel'][i]
            if window_digest(window)!=prior['window_sha256']:raise ValueError('old panel window differs')
            encoded=policy.encode_observations(**observed([{k:v for k,v in window.items() if k!='loss_targets'}],'cuda'))
            row={'task':record['record']['task'],**window['provenance'],'window_sha256':window_digest(window),
                 'normalized_target':window['loss_targets']['actions'].tolist(),'samples':{}}
            for variant in DIAGNOSTIC_CONFIG['interventions']:
                row['samples'][variant]=[]
                for seed in SEEDS:
                    noise=torch.randn((1,32,7),device='cuda',dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(seed))
                    noise_sha=tensor_digest({'noise':noise})
                    if variant=='bypass':
                        with bypass_conditioner(policy):pred=policy.sample_actions(encoded,noise,num_steps=10)
                    else:pred=policy.sample_actions(encoded,noise,num_steps=10,forecast_intervention=variant)
                    array=pred[0].float().cpu().numpy()
                    if tensor_digest({'noise':noise})!=noise_sha:raise ValueError('initial noise changed')
                    sample={'seed':seed,'initial_noise_sha256':noise_sha,**metrics(array,window['loss_targets']['actions'],normalization)}
                    if variant=='none':
                        original=next(s for s in prior['samples'] if s['integration_steps']==10 and s['seed']==seed)
                        difference=float(np.abs(array-np.asarray(original['normalized_prediction'])).max())
                        if noise_sha!=original['initial_noise_sha256'] or difference>DIAGNOSTIC_CONFIG['reproduction_atol']:
                            raise ValueError('baseline does not reproduce previous action panel')
                        sample['previous_panel_max_abs']=difference
                    row['samples'][variant].append(sample)
            report['rows'].append(row);persist()
            print(json.dumps({'event':'ablation_progress','model':report['model'],'windows':i+1,'of':12}),flush=True)
    if observed.layouts!=reference['processed_layouts']:raise ValueError('previous native input layouts differ')
    if tensor_digest(policy.state_dict())!=before:raise ValueError('ablation mutated model')
    assert_nested_equal(rng,{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
    report.update(status='passed',summary=ablation_summary(report['rows']),parameters_buffers_rng_unchanged=True)


def decide(audit,retention,semflow,previous,targets):
    if audit['status']!='passed' or retention['status']!='passed' or semflow['status']!='passed':
        raise ValueError('complete diagnostic stages required')
    horizon='first_eight';control=retention['panel_summary']['10']['all'][horizon]['sampled_mse']
    gains={arm:{} for arm in ('standard','semflow')}
    for arm in gains:
        for target in targets:
            value=previous[arm]['panels'][target]['panel_summary']['10']['all'][horizon]['sampled_mse']
            gains[arm][target]=(control-value)/control
    all_models={'frozen':audit['ablations'],**semflow['ablations']};useful={}
    for name,panel in all_models.items():
        summary=panel['summary'];base=summary['all']['none'][horizon]['sampled_mse'];checks={}
        for intervention in ('static','shuffle'):
            alt=summary['all'][intervention][horizon]
            relative=(alt['sampled_mse']-base)/alt['sampled_mse']
            task_wins=sum(s[intervention][horizon]['sampled_mse']>s['none'][horizon]['sampled_mse'] for task,s in summary.items() if task!='all')
            checks[intervention]={'relative_gain_of_predicted':relative,'task_wins':task_wins,
                'action_rms':alt['paired_action_rms'],'passed':relative>=DIAGNOSTIC_CONFIG['minimum_relative_gain']
                and task_wins>=DIAGNOSTIC_CONFIG['minimum_source_tasks'] and alt['paired_action_rms']>=DIAGNOSTIC_CONFIG['minimum_action_rms']}
        useful[name]={'comparisons':checks,'passed':all(c['passed'] for c in checks.values())}
    human_gain=float(np.mean(list(gains['semflow'].values())))
    human_pass=human_gain>=.01 and sum(v>0 for v in gains['semflow'].values())>=2
    forecast_pass=any(v['passed'] for v in useful.values())
    return {'recommendation':'continue_with_source_controller_repair' if forecast_pass or human_pass else 'pause_current_approach',
        'implementation_audit_passed':True,'incremental_human_gain_vs_retention':gains,
        'mean_semflow_relative_gain':human_gain,'human_increment_passed':human_pass,
        'useful_forecast_checks':useful,'useful_forecast_passed':forecast_pass,
        'limitations':'Engineering decision thresholds fixed before this diagnostic run, after prior outcomes were known; one adaptation seed, fixed source panel, no target-transfer test. A parity failure blocks interpretation and calls for repair.'}


def run(args,report):
    import torch
    from train_source_calibration import MemorySources,tensor_digest,save_checkpoint
    from semflow.source_batching import SourceWindowSampler,source_window
    from semflow.models.adaptation import source_action_retention
    request,index,tasks,plan,panel_spec,training,targets,spec,previous=validate_inputs(args)
    report.update(mode=args.mode,diagnostic_request=spec,request=request,optimizer_steps=0,rollouts=0)
    policy,processor,entries,_=load_common(args,index,request,report)
    common=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state=policy.state_dict()
    if tensor_digest(common['model_delta'])!=tensor_digest({n:state[n] for n in common['model_delta']}):
        raise ValueError('complete loaded source checkpoint delta differs')
    report['complete_checkpoint_delta_equal']=True
    del common,state
    initial={n:p.detach().cpu().clone() for n,p in entries};initial_sha=tensor_digest(initial)
    if initial_sha!=training['standard']['initial_adapter_sha256']:raise ValueError('common source adapter hash differs')
    report['initial_adapter_sha256']=initial_sha
    frozen_versions={n:t._version for n,t in (*policy.named_parameters(),*policy.named_buffers()) if n not in initial}
    sources=MemorySources(index['source_lock'])
    try:
        sampler=SourceWindowSampler(index,tasks,sources.handles.__getitem__,cache_size=4)
        windows={g:[source_window(sampler.episode(r['record']),r['anchor']) for r in rows] for g,rows in plan.items()}
        def encode(w):return policy.encode_observations(**processor([{k:v for k,v in w.items() if k!='loss_targets'}],'cuda'))
        def persist():native_probe.write_report(report,args.output)
        contract={'request':request,'diagnostic_request':spec,'initial_adapter_sha256':initial_sha,'config':CONFIG}
        if args.mode=='audit':
            from src.policy.egosteer_inference import infer_flow_action
            from src.policy.egosteer_loss import psi_t
            report['upstream_files']={n:sha(args.upstream_root/'src/policy'/n) for n in ('egosteer.py','egosteer_inference.py','egosteer_loss.py')}
            report['normalization_refit']=refit_normalization(index,sources.handles)
            report['flow_convention']={'source_sigma_min':0.,'native_sigma_min':getattr(policy.native,'flow_sig_min',None),
                'direction':'noise at t0 to action at t1','panda_coordinates':'source-training population z-score; learned native input/output maps',
                'prefix':'cached observed-only prefix from measured source states; no RTC prefix'}
            policy.requires_grad_(False).eval();before=tensor_digest(policy.state_dict())
            rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
            report['normalization']=[];report['native_parity']=[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i in range(0,12,2):
                    window=windows['validation'][i];record=plan['validation'][i]['record']
                    report['normalization'].append(normalization_audit(window,record,sources.handles[record['task']],index))
                    obs=encode(window);noise=torch.randn((1,32,7),device='cuda',generator=torch.Generator(device='cuda').manual_seed(SEEDS[0]))
                    action=torch.from_numpy(window['loss_targets']['actions'])[None].float().cuda()
                    for t in (0.,.5,.9):
                        times=torch.full((1,32),t,device='cuda')
                        expected=(1-t)*noise+t*action
                        actual=psi_t(noise,action,times,flow_sig_min=0.)
                        if not torch.allclose(expected,actual,rtol=0,atol=1e-6):raise ValueError('native noise/target interpolation differs')
                        ours=policy.predict_velocity(obs,expected,times)
                        with native_panda_stream(policy,obs):
                            native=policy.native.forward_flow_stream(batch={**obs.batch,'actions_valid_mask':torch.ones_like(action,dtype=torch.bool)},
                                backbone_output=obs.backbone,flow_inputs={'noisy_actions':expected,'time_for_model':times},num_parallel_chunks=1)['pred_v']
                        error=float((ours-native).abs().max())
                        report['native_parity'].append({'window':i,'time':t,'velocity_max_abs':error})
                        if error>DIAGNOSTIC_CONFIG['native_atol']:
                            persist();raise ValueError('native flow stream parity failed')
                    for steps in (5,10,20):
                        ours=accumulated_sample(policy,obs,noise,steps)
                        native=native_sample(policy,obs,noise,steps,infer_flow_action)
                        error=float((ours-native).abs().max())
                        indexed=policy.sample_actions(obs,noise,num_steps=steps)
                        report['native_parity'].append({'window':i,'steps':steps,'sample_max_abs':error,
                            'audit_clock':'native_accumulated',
                            'historical_indexed_max_abs':float((indexed-native).abs().max())})
                        if error>DIAGNOSTIC_CONFIG['native_atol']:
                            persist();raise ValueError('native sampler parity failed')
                    persist()
            if before!=tensor_digest(policy.state_dict()):raise ValueError('native audit changed model')
            assert_nested_equal(rng,{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
            report['native_parity_passed']=True
            report['ablations']={'model':'frozen'}
            run_ablations(policy,processor,windows['validation'],plan['validation'],index['normalization'],
                previous['frozen']['panels']['frozen'],report['ablations'],persist)
        elif args.mode in ('retention-train','retention-evaluate'):
            parameters=[p for _,p in entries]
            optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'],foreach=False)
            torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
            def retention_loss(step):
                r=(step+step//12)%12;window=windows['retention'][r]
                if window['provenance']['split']!='train':raise ValueError('retention validation leakage')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    return source_action_retention(policy,encode(window),torch.from_numpy(window['loss_targets']['actions'])[None].cuda(),seed=CONFIG['flow_seed']+step)
            def reference():
                with torch.no_grad():return retention_loss(0).item()
            if args.mode=='retention-train':
                report['training']=[]
                for step in range(60):
                    optimizer.zero_grad(set_to_none=True);loss=retention_loss(step);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(parameters,CONFIG['gradient_clip'],error_if_nonfinite=True)
                    if not norm.item()>0:raise ValueError('zero retention gradient')
                    optimizer.step();report['optimizer_steps']=step+1
                    report['training'].append({'step':step+1,'loss':loss.item(),'gradient_norm':norm.item()})
                    if step==0 or (step+1)%12==0:
                        print(json.dumps({'event':'retention_progress','step':step+1}),flush=True);persist()
                optimizer.zero_grad(set_to_none=True)
                adapters={n:p.detach().cpu().clone() for n,p in entries}
                if tensor_digest(adapters)==initial_sha or any(not torch.isfinite(v).all() for v in adapters.values()):raise ValueError('invalid retention update')
                payload={'schema_version':1,'contract':contract,'step':60,'adapters':adapters,
                    'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'reference':reference()}
                p=args.output.with_suffix('.pt')
                if p.exists():raise FileExistsError('preserve retention checkpoint')
                save_checkpoint(p,payload);report['checkpoint']={'sha256':sha(p),'bytes':p.stat().st_size,'step':60}
            else:
                if args.arm_checkpoint is None or sha(args.arm_checkpoint)!=args.arm_checkpoint_sha256:raise ValueError('retention export hash differs')
                saved=torch.load(args.arm_checkpoint,map_location='cpu',weights_only=True)
                if saved['schema_version']!=1 or saved['contract']!=contract or saved['step']!=60:raise ValueError('retention checkpoint contract differs')
                reset_adapters(entries,saved['adapters']);optimizer.load_state_dict(saved['optimizer'])
                assert_nested_equal(optimizer.state_dict(),saved['optimizer'])
                assert_nested_equal({n:p.detach().cpu() for n,p in entries},saved['adapters'])
                torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
                assert_nested_equal({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},saved['rng'])
                value=reference()
                if not np.isclose(value,saved['reference'],rtol=1e-6,atol=1e-7):raise ValueError('retention recovery loss differs')
                report.update(checkpoint_roundtrip_equal=True,recovered_training_steps=60,condition='retention',panel_config=PANEL_CONFIG)
                observed=ObservedProcessor(processor)
                sample_panel(policy,observed,windows['validation'],plan['validation'],index['normalization'],report,persist)
                report['processed_layouts']=observed.layouts
                if panel_identity(report)!=panel_identity(previous['frozen']['panels']['frozen']):raise ValueError('retention comparison pairing differs')
                report['adapter_distances']={}
                retained=saved['adapters']
                for arm in ('standard','semflow'):
                    human=torch.load(args.input_root/(arm+'.pt'),map_location='cpu',weights_only=True)
                    report['adapter_distances'][arm]={t:float(sum((human['tasks'][t]['adapters'][n].float()-v.float()).square().sum().item() for n,v in retained.items())**.5) for t in targets}
        else:
            if args.audit_report is None or args.retention_report is None:raise ValueError('exported audit and retention reports required')
            saved=torch.load(args.input_root/'semflow.pt',map_location='cpu',weights_only=True)
            report['ablations']={}
            for target in targets:
                reset_adapters(entries,initial);restore_task(entries,saved,training['semflow'],target)
                report['ablations'][target]={'model':target}
                run_ablations(policy,processor,windows['validation'],plan['validation'],index['normalization'],
                    previous['semflow']['panels'][target],report['ablations'][target],persist)
            report['status']='passed'
            report['decision']=decide(json.loads(args.audit_report.read_text()),json.loads(args.retention_report.read_text()),report,previous,targets)
        if any(t._version!=frozen_versions[n] for n,t in (*policy.named_parameters(),*policy.named_buffers()) if n in frozen_versions):raise ValueError('frozen model changed')
        if any(p.grad is not None for p in policy.parameters()):raise ValueError('leftover parameter gradients')
        report.update(status='passed',frozen_state_unchanged=True,memory=native_probe.memory())
    finally:sources.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','panel-request','diagnostic-request','input-root','index','checkpoint','checkpoint-receipt','upstream-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('audit','retention-train','retention-evaluate','semflow'),required=True)
    for name in ('arm-checkpoint','audit-report','retention-report'):p.add_argument('--'+name,type=Path)
    p.add_argument('--arm-checkpoint-sha256')
    p.add_argument('--pilot',type=Path,default=Path('configs/pilot.json'));p.add_argument('--calibration',type=Path,default=Path('configs/source_calibration.json'))
    p.add_argument('--cache-dir',type=Path,default=Path('/content/semflow-model-cache/hub'));p.add_argument('--teacher-path',type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve diagnostic output')
    report={'status':'running','optimizer_steps':0,'rollouts':0};start=time.monotonic()
    def timeout(s,f):raise TimeoutError('diagnostic stage exceeded900s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:run(args,report)
    except Exception as error:report.update(status='failed',error=str(error).split('\n')[0][:600]);raise
    finally:signal.alarm(0);report['seconds']=time.monotonic()-start;native_probe.write_report(report,args.output)
if __name__=='__main__':main()
