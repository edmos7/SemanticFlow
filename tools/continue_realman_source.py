#!/usr/bin/env python3
"""Bounded source-only continuation of the immutable matched RealMan screen."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time

from compare_source_initializations import (execute, planned_records, paired_flow, decomposition,
                                            CONFIG, INIT_SEED, FLOW_SEED)
from probe_realman import file_sha
from train_balanced_source import BalancedSchedule, digest, panel_records

CHECKPOINT_SHA256 = "da71ac6ca3a5f18ca4d8f891d30ac007a6208a41cfb01e2ad0a71456ae90babb"
CHECKPOINT_BYTES = 49831470
PANEL_SHA256 = "241add05bc6bbf423a34e4eaebbae805fa2f6db4d2607ac299ab8a4f7a66d186"
STAGES = ("continuation-preflight", "continuation-recover-120", "continuation-train-600",
          "continuation-panel-600", "continuation-train-1200", "continuation-panel-1200",
          "continuation-train-2400", "continuation-panel-2400")
STOP_RULES = dict(min_relative_improvement=.01, max_initial_regression=1.25,
    rule="Stop if neither validation sampled nor four-seed mean MSE improves >=1% on either horizon versus previous panel; stop if either validation sampled horizon exceeds 1.25x step120. Always stop at2400. Failures/nonfinite/provenance/export/time bounds stop immediately.")

class ContinuationSchedule(BalancedSchedule):
    max_updates = 2400


def continuation_plan(source):
    schedule = ContinuationSchedule(source)
    rows = []
    for step in range(1, 2401):
        record, anchor = schedule.next_record()
        rows.append(dict(step=step, task=record["task"], episode_sha256=record["sha256"],
                         split=record["split"], anchor=anchor))
        schedule.updates += 1
    if rows[:120] != planned_records(source) or set(Counter(r["task"] for r in rows).values()) != {400}:
        raise ValueError("continuation changed prefix or task balance")
    return rows


def declaration(source):
    return dict(kind="realman-source-continuation-v1", parent_sha256=CHECKPOINT_SHA256,
        start=120, end=2400, panels=[600,1200,2400], schedule_sha256=digest(continuation_plan(source)),
        schedule_seed=424242, flow_seed=FLOW_SEED, init_seed=INIT_SEED,
        stop_rules=STOP_RULES, objective=CONFIG, image_size=[480,640])


def validate_saved(saved, sha, source, step):
    if step not in (120,600,1200,2400) or saved["step"] != step or saved["sampler"]["updates"] != step:
        raise ValueError("continuation checkpoint position differs")
    if step == 120:
        if sha != CHECKPOINT_SHA256 or "continuation" in saved:
            raise ValueError("continuation must begin at the immutable matched checkpoint")
    elif saved.get("continuation") != declaration(source):
        raise ValueError("continuation declaration differs")
    schedule = ContinuationSchedule(source)
    schedule.load_state_dict(saved["sampler"])


def panel_identity(report):
    return [dict(**{k:r[k] for k in ("task","episode_sha256","split","anchor","window_sha256","normalized_target")},
                 noise=[(s["seed"],s["initial_noise_sha256"],s["integration_steps"]) for s in r["samples"]])
            for r in report["panel"]]


def stop_decision(current, previous, initial, step):
    for other in (previous, initial):
        if panel_identity(current) != panel_identity(other) or current["panel_layouts"] != other["panel_layouts"]:
            raise ValueError("continuation panel identity/layout changed")
    scores = [decomposition(r["panel"]) for r in (current,previous,initial)]
    cur, prev, base = [s["validation"]["all"] for s in scores]
    horizons = ("full","first_eight")
    improved = any(cur[h][m] <= .99*prev[h][m] for h in horizons
                   for m in ("sampled_mse","mean_prediction_mse"))
    regression = any(cur[h]["sampled_mse"] > 1.25*base[h]["sampled_mse"] for h in horizons)
    train_improved = any(scores[0]["train"]["all"][h]["sampled_mse"] <=
                         .99*scores[1]["train"]["all"][h]["sampled_mse"] for h in horizons)
    return dict(continue_training=step < 2400 and improved and not regression,
        reason="budget_complete" if step == 2400 else "validation_regression" if regression else
               "continue" if improved else "coverage_review" if train_improved else "optimization_interface_review",
        validation_improved=improved, train_improved=train_improved,
        validation_minus_train={h:scores[0]["validation"]["all"][h]["sampled_mse"]-
            scores[0]["train"]["all"][h]["sampled_mse"] for h in horizons}, rules=STOP_RULES)


def stage_bounds(stage):
    if stage not in STAGES[1:]:
        raise ValueError("unknown continuation stage")
    _, mode, text = stage.split("-")
    end = int(text)
    start = {600:120,1200:600,2400:1200}[end] if mode == "train" else end
    return mode,start,end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES[1:],required=True)
    for name in ('upstream-root','index','output','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--previous-panel',type=Path)
    parser.add_argument('--initial-panel',type=Path)
    args=parser.parse_args()
    if (not Path('/content').is_dir() or os.environ.get('HF_HOME') != '/content/semflow-model-cache'
            or any(not p.resolve().is_relative_to('/content') for p in
                   (args.upstream_root,args.index,args.output,args.checkpoint))):
        parser.error('reviewed ephemeral cloud paths required; no host fallback')
    if args.output.exists() or args.output.with_suffix('.pt').exists():
        parser.error('preserve previous attempt')
    mode,args.start_step,args.end_step=stage_bounds(args.stage)
    args.mode='final' if mode=='panel' else mode
    args.arm='realman'; args.continuation=True
    args.reference_initial=args.reference_train=None
    args.cache_dir=Path('/content/semflow-model-cache/hub'); args.teacher_path=None
    if args.start_step==120 and (file_sha(args.checkpoint)!=CHECKPOINT_SHA256 or args.checkpoint.stat().st_size!=CHECKPOINT_BYTES):
        parser.error('immutable step120 input differs')
    report=dict(status='running',arm='realman',mode=mode,rollouts=0)
    start=time.monotonic()
    def deadline(*_): raise TimeoutError('continuation stage deadline; no retry')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(3900 if mode=='train' else 1050)
    try:
        execute(args,report)
        if mode=="panel":
            if file_sha(args.initial_panel)!=PANEL_SHA256:
                raise ValueError("step120 reference panel changed")
            report["stop_decision"]=stop_decision(report,json.loads(args.previous_panel.read_text()),
                json.loads(args.initial_panel.read_text()),args.end_step)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error).split('\n')[0][:600])
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-start
        from probe_egosteer import write_report
        write_report(report,args.output)
    print(json.dumps({k:report.get(k) for k in ('status','mode','seconds','error')}),flush=True)
    return 0 if report['status']=='passed' else 1

if __name__=='__main__': raise SystemExit(main())
