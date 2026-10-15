#!/usr/bin/env python3
"""Fixed Base/RealMan source-learning screen; no target adaptation or rollout."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import probe_egosteer as native_probe
from probe_realman import ARTIFACTS as REALMAN_ARTIFACTS, INDEX_SHA256, file_sha, verify_files
from probe_source_actions import SEEDS, optimizer_contract, window_digest
from train_balanced_source import BalancedSchedule, digest, panel_records, run_panel

UPDATES = 120
INIT_SEED = 42
FLOW_SEED = 920000
SCHEDULE_SHA256 = "be4de3dba187926518611cfcc220fa6bef00359388208202edd1cba06699f9d2"
CONFIG = dict(batch_size=1, learning_rate=.0003, adapter_learning_rate=.00005,
              weight_decay=.01, gradient_clip=1., loss_weights=dict(
                  action=1., raw_future=.1, reconstruction=.1, noncollapse=.1))


def planned_records(source):
    schedule = BalancedSchedule(source)
    rows = []
    for step in range(1, UPDATES + 1):
        record, anchor = schedule.next_record()
        rows.append(dict(step=step, task=record["task"], episode_sha256=record["sha256"],
                         split=record["split"], anchor=anchor))
        schedule.updates += 1
    if (digest(rows) != SCHEDULE_SHA256 or set(Counter(r["task"] for r in rows).values()) != {20}
            or len({r["episode_sha256"] for r in rows}) != UPDATES):
        raise ValueError("comparison schedule differs from the reviewed120-window plan")
    return rows


def paired_flow(actions, step, *, max_updates=UPDATES):
    import torch
    if type(step) is not int or not 1 <= step <= max_updates:
        raise ValueError("step outside paired training budget")
    generator = torch.Generator(device=actions.device).manual_seed(FLOW_SEED + step)
    noise = torch.randn(actions.shape, dtype=actions.dtype, device=actions.device, generator=generator)
    flow_time = torch.rand((len(actions), 1, 1), dtype=actions.dtype, device=actions.device, generator=generator)
    return noise, flow_time


def matching_initialization(actual, expected):
    if actual != expected:
        raise ValueError("fresh interface/adapter initialization differs across arms")


def decomposition(rows):
    """Finite-sample mean error plus population-denominator within-seed variance."""
    result = {}
    for split in ("train", "validation"):
        result[split] = {}
        for task in ["all"] + sorted({r["task"] for r in rows}):
            selected = [r for r in rows if r["split"] == split and (task == "all" or r["task"] == task)]
            if len(selected) != (12 if task == "all" else 2):
                raise ValueError("incomplete task/split panel")
            if any([s["seed"] for s in r["samples"]] != list(SEEDS) for r in selected):
                raise ValueError("unpaired panel seeds")
            predictions = np.asarray([[s["normalized_prediction"] for s in r["samples"]] for r in selected])
            targets = np.asarray([r["normalized_target"] for r in selected])
            if (predictions.shape != (len(selected), 4, 32, 7) or targets.shape != (len(selected), 32, 7)
                    or not np.isfinite(predictions).all() or not np.isfinite(targets).all()):
                raise ValueError("invalid panel arrays")
            result[split][task] = {}
            for label, count in (("full", 32), ("first_eight", 8)):
                x, y = predictions[:, :, :count], targets[:, :count]
                mean_error = np.square(x.mean(1) - y).mean((0, 1))
                variance = x.var(1).mean((0, 1))
                mse = np.square(x - y[:, None]).mean((0, 1, 2))
                if not np.allclose(mse, mean_error + variance, rtol=1e-10, atol=1e-10):
                    raise RuntimeError("seed decomposition failed")
                samples = [s[label] for r in selected for s in r["samples"]]
                result[split][task][label] = dict(sampled_mse=float(mse.mean()),
                    mean_prediction_mse=float(mean_error.mean()), seed_variance=float(variance.mean()),
                    per_dimension_mse=mse.tolist(), per_dimension_mean_error=mean_error.tolist(),
                    per_dimension_seed_variance=variance.tolist(), signed_mean_error=(x.mean(1)-y).mean((0,1)).tolist(),
                    train_mean_baseline_mse=float(np.square(y).mean()),
                    gripper_sign_accuracy=float(np.mean([s["gripper_sign_accuracy"] for s in samples])),
                    gripper_mean=float(np.mean([s["gripper_mean"] for s in samples])),
                    gripper_negative_fraction=float(np.mean([s["gripper_negative_fraction"] for s in samples])),
                    gripper_positive_fraction=float(np.mean([s["gripper_positive_fraction"] for s in samples])),
                    clipped_values=float(np.mean([s["clipped_values"] for s in samples])),
                    clipped_by_dimension=np.mean([s["clipped_by_dimension"] for s in samples],axis=0).tolist())
    return result


def pairing_record(row):
    return {k: row[k] for k in ("step", "task", "episode_sha256", "split", "anchor",
                                "window_sha256", "flow_inputs_sha256", "processed_layout")}


def compare_reports(reports):
    """Reject confounded results before reporting paired outcomes, never select a seed."""
    expected = {(arm, mode) for arm in ("base", "realman") for mode in ("initial", "train", "final")}
    if set(reports) != expected or any(r["status"] != "passed" for r in reports.values()):
        raise ValueError("comparison requires six passed arm reports")
    reference = reports[("base", "initial")]
    for report in reports.values():
        matching_initialization(report["initialization"], reference["initialization"])
        if report["comparison_contract"] != reference["comparison_contract"]:
            raise ValueError("comparison contracts differ")
    traces = [[pairing_record(r) for r in reports[(arm, "train")]["training"]] for arm in ("base", "realman")]
    if len(traces[0]) != UPDATES or traces[0] != traces[1]:
        raise ValueError("training windows/noise/time/layout were not paired")
    panel_identity = None
    summaries = {}
    for arm, mode in sorted(expected - {(a, "train") for a in ("base", "realman")}):
        report = reports[(arm, mode)]
        identity = [{k:r[k] for k in ("task", "episode_sha256", "split", "anchor", "window_sha256", "normalized_target")}
                    | {"noise": [(s["seed"], s["initial_noise_sha256"], s["integration_steps"]) for s in r["samples"]]}
                    for r in report["panel"]]
        if panel_identity is not None and identity != panel_identity:
            raise ValueError("panel windows/targets/noise differ")
        panel_identity = identity
        if report["panel_layouts"] != reference["panel_layouts"]:
            raise ValueError("panel processed layouts differ")
        summaries[arm + "_" + mode] = decomposition(report["panel"])
    paired = {}
    for split in ("train", "validation"):
        paired[split] = {}
        for horizon in ("full", "first_eight"):
            scores = {k:v[split]["all"][horizon]["sampled_mse"] for k,v in summaries.items()}
            paired[split][horizon] = dict(realman_minus_base_final=scores["realman_final"]-scores["base_final"],
                base_change=scores["base_final"]-scores["base_initial"],
                realman_change=scores["realman_final"]-scores["realman_initial"])
    gaps = {name:{horizon:value["validation"]["all"][horizon]["sampled_mse"]-
                  value["train"]["all"][horizon]["sampled_mse"] for horizon in ("full","first_eight")}
            for name,value in summaries.items()}
    return dict(summaries=summaries, paired=paired, validation_minus_train=gaps, matched=True,
                interpretation="One training seed; early source-learning screen, not convergence or controller success.")


class ObservedProcessor:
    """Strip labels from panel helpers too; fingerprint actual processing layouts."""
    def __init__(self, processor):
        self.processor, self.layouts = processor, []

    def __call__(self, windows, device):
        import torch
        result = self.processor([{k:v for k,v in w.items() if k != "loss_targets"} for w in windows], device)
        batch = result["batch"]
        layout = {k: list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}
        for key in ("video_grid_thw", "image_grid_thw", "input_ids", "attention_mask"):
            if torch.is_tensor(batch.get(key)):
                from train_source_calibration import tensor_digest
                layout[key + "_sha256"] = tensor_digest({key: batch[key]})
        self.layouts.append(layout)
        return result


def execute(args, report):
    import torch
    from semflow.config import load_config
    from semflow.models import NativeSemFlowPolicy, inject_shared_lora
    from semflow.models.source_calibration import (SpatialReconstruction, enable_source_training,
        calibration_losses, trainable_delta, checkpoint_payload, restore_checkpoint)
    from semflow.native_batching import PandaBatchProcessor
    from semflow.source_batching import SourceWindowSampler
    from train_source_calibration import MemorySources, tensor_digest, save_checkpoint

    if file_sha(args.index) != INDEX_SHA256 or not torch.cuda.is_available():
        raise ValueError("comparison requires the pinned index and reviewed CUDA runtime")
    index = json.loads(args.index.read_text())
    _, tasks = load_config(Path(__file__).resolve().parents[1] / "configs/pilot.json")
    source = SourceWindowSampler(index, tasks, lambda _:None, seed=INIT_SEED)
    continuation = getattr(args, "continuation", False)
    plan = planned_records(source)
    if continuation:
        from continue_realman_source import ContinuationSchedule, continuation_plan, declaration, validate_saved
        plan = continuation_plan(source)
    common = dict(kind="source-initialization-screen-v1", source_index_sha256=INDEX_SHA256,
        schedule_sha256=SCHEDULE_SHA256, updates=UPDATES, image_size=[480,640],
        init_seed=INIT_SEED, flow_seed=FLOW_SEED, configuration=CONFIG,
        panel=panel_records(source), sampling_seeds=list(SEEDS), flow_steps=10)
    artifacts = native_probe.ARTIFACTS if args.arm == "base" else REALMAN_ARTIFACTS
    native_probe.prepare_native_imports(args)
    native_probe.access_preflight(report, args, artifacts=artifacts)
    paths = native_probe.download_models(report, args, artifacts=artifacts)
    weights_hash = file_sha(paths["model_bf16.pt"])
    if weights_hash != report["access"]["egosteer"]["etag"]:
        raise ValueError("native weight checksum differs")
    if args.arm == "realman":
        verify_files(paths)
    elif file_sha(paths["config.yaml"]) != "cdf634e00f95f0b19d7b291eae7f15f6d14e0a6d1838ffe1be860fc995d98c53":
        raise ValueError("Base config checksum differs")
    native, cfg = native_probe.load_model(report, args, paths, synchronous=True)
    # This is common processing, not a change in native parameter dimensions.
    cfg.data.target_image_size = [480,640]
    cfg.data.target_pixels = 480*640
    random.seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    torch.manual_seed(INIT_SEED)
    torch.cuda.manual_seed_all(INIT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    inject_shared_lora(native)
    policy = NativeSemFlowPolicy(native).cuda()
    reconstruction = SpatialReconstruction(policy.scene.dim, native.world_model_expert.hidden_size).cuda()
    parameters = enable_source_training(policy, reconstruction)
    adapters = [p for n,p in policy.named_parameters() if p.requires_grad and ".lora_" in n]
    ids = {id(p) for p in adapters}
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in parameters if id(p) not in ids], lr=CONFIG["learning_rate"]),
        dict(params=adapters, lr=CONFIG["adapter_learning_rate"])], weight_decay=CONFIG["weight_decay"], foreach=False)
    scope = optimizer_contract(policy, reconstruction, optimizer, CONFIG, 0)
    initialization = dict(tensors_sha256=tensor_digest(dict(trainable_delta(policy), **{
        "reconstruction."+n:v for n,v in reconstruction.state_dict().items()})), optimizer_scope=scope)
    schedule = ContinuationSchedule(source) if continuation else BalancedSchedule(source)
    contract = json.loads(json.dumps(dict(common=common, artifacts=artifacts,
        upstream=native_probe.UPSTREAM_REVISION, native_weights_sha256=weights_hash)))
    report.update(comparison_contract=common, initialization=initialization, optimizer_steps=0,
                  training=[], rollouts=0, environment=native_probe.environment())
    persist = lambda: native_probe.write_report(report,args.output)
    persist()
    if args.reference_initial:
        prior = json.loads(args.reference_initial.read_text())
        matching_initialization(initialization, prior["initialization"])
        if common != prior["comparison_contract"]:
            raise ValueError("initial comparison contract differs")
    if args.mode != "initial":
        if args.checkpoint is None or file_sha(args.checkpoint) != args.checkpoint_sha256:
            raise ValueError("verified input checkpoint required")
        saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        expected_step = args.start_step if continuation else (0 if args.mode == "train" else UPDATES)
        if continuation:
            validate_saved(saved, args.checkpoint_sha256, source, expected_step)
        if saved["step"] != expected_step or saved["comparison_initialization"] != initialization:
            raise ValueError("checkpoint initialization or step differs")
        restore_checkpoint(saved, policy, reconstruction, optimizer, schedule, contract=contract)
        optimizer_contract(policy,reconstruction,optimizer,CONFIG,expected_step)
        if schedule.updates != expected_step:
            raise ValueError("restored schedule counter differs")
        # Reconstruct the payload to check optimizer moments, RNG and all deltas.
        roundtrip = checkpoint_payload(policy,reconstruction,optimizer,schedule,step=expected_step,contract=contract)
        assert_nested_equal({k:v for k,v in saved.items() if k not in ("comparison_initialization", "continuation")}, roundtrip)
        report["checkpoint_roundtrip_equal"] = True
        report["optimizer_steps"] = expected_step
        del saved, roundtrip
    processor = ObservedProcessor(PandaBatchProcessor(native,cfg))
    memory = MemorySources(index["source_lock"])
    try:
        source.open_source = memory.handles.__getitem__
        delta_names = set(trainable_delta(policy))
        frozen = {n:v for n,v in policy.state_dict().items() if n not in delta_names}
        frozen_before = tensor_digest(frozen)
        if args.mode == "train":
            reference_trace = (json.loads(args.reference_train.read_text())["training"] if args.reference_train else None)
            begin_step = args.start_step + 1 if continuation else 1
            end_step = args.end_step if continuation else UPDATES
            for step in range(begin_step, end_step+1):
                windows = schedule.sample()
                window = windows[0]
                if window["provenance"]["split"] != "train":
                    raise ValueError("validation entered optimizer input")
                before = window_digest(window)
                observations = processor(windows,"cuda")
                future = torch.from_numpy(window["loss_targets"]["future_rgb"][None]).cuda()
                actions = torch.from_numpy(window["loss_targets"]["actions"][None]).float().cuda()
                noise, flow_time = paired_flow(actions,step,max_updates=2400 if continuation else UPDATES)
                row = dict(plan[step-1], window_sha256=before,
                    flow_inputs_sha256=tensor_digest({"noise":noise,"time":flow_time}),
                    processed_layout=processor.layouts[-1])
                if any(row[k] != window["provenance"][k] for k in ("episode_sha256","split","anchor")):
                    raise ValueError("actual training window differs from plan")
                if reference_trace is not None and pairing_record(row) != pairing_record(reference_trace[step-1]):
                    raise ValueError("training pairing differs before optimizer update")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    encoded = policy.encode_observations(**observations)
                    loss, terms = calibration_losses(policy,reconstruction,encoded,future,actions,
                        weights=CONFIG["loss_weights"],flow_inputs=(noise,flow_time))
                loss.backward()
                if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                    raise RuntimeError("frozen base received gradients")
                torch.nn.utils.clip_grad_norm_(parameters,CONFIG["gradient_clip"],error_if_nonfinite=True)
                optimizer.step()
                optimizer_contract(policy,reconstruction,optimizer,CONFIG,step)
                if before != window_digest(window):
                    raise RuntimeError("training mutated source input")
                row.update(total=float(loss.detach()), **{k:float(v.detach()) for k,v in terms.items()})
                report["training"].append(row)
                report["optimizer_steps"] = step
                del encoded,loss,terms
                if step % 20 == 0:
                    print(json.dumps({"arm":args.arm,"step":step,"total":row["total"]}),flush=True)
                    persist()
        elif args.mode != "recover":
            run_panel(policy,processor,source,persist,report)
            report["panel_layouts"] = processor.layouts
            report["decomposition"] = decomposition(report["panel"])
            optimizer_contract(policy,reconstruction,optimizer,CONFIG,args.start_step if continuation else (0 if args.mode == "initial" else UPDATES))
        if tensor_digest(frozen) != frozen_before:
            raise RuntimeError("frozen native weights/buffers changed")
        if args.mode in ("initial","train"):
            payload = checkpoint_payload(policy,reconstruction,optimizer,schedule,
                step=report["optimizer_steps"],contract=contract)
            payload["comparison_initialization"] = initialization
            if continuation:
                payload["continuation"] = declaration(source)
            destination = args.output.with_suffix(".pt")
            if destination.exists():
                raise FileExistsError("checkpoint exists; preserve previous attempt")
            save_checkpoint(destination,payload)
            report["checkpoint"] = dict(file=destination.name,bytes=destination.stat().st_size,sha256=file_sha(destination))
        if args.checkpoint and file_sha(args.checkpoint) != args.checkpoint_sha256:
            raise RuntimeError("input checkpoint changed")
        if continuation:
            report["continuation"] = declaration(source)
            report["schedule_state"] = schedule.state_dict()
        report.update(status="passed",memory=native_probe.memory(), frozen_native_unchanged=True)
    finally:
        memory.close()


def assert_nested_equal(left,right):
    import torch
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left.cpu(),right.cpu()):
            raise ValueError("checkpoint tensor roundtrip differs")
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():
            raise ValueError("checkpoint mapping roundtrip differs")
        for key in left:
            assert_nested_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):
            raise ValueError("checkpoint sequence roundtrip differs")
        for a,b in zip(left,right):
            assert_nested_equal(a,b)
    elif left != right:
        raise ValueError("checkpoint value roundtrip differs")


def main():
    if "--summarize" in sys.argv:
        parser = argparse.ArgumentParser(description="Summarize the six verified comparison reports")
        parser.add_argument("--summarize", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        reports = {(arm,mode):json.loads((args.summarize / ("comparison-"+arm+"-"+mode+".json")).read_text())
                   for arm in ("base","realman") for mode in ("initial","train","final")}
        with args.output.open("x") as stream:
            json.dump(compare_reports(reports),stream,indent=2)
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=("base","realman"),required=True)
    parser.add_argument("--mode",choices=("initial","train","final"),required=True)
    for name in ("upstream-root","index","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("checkpoint","reference-initial","reference-train"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--checkpoint-sha256")
    args=parser.parse_args()
    paths=[args.upstream_root,args.index,args.output,args.checkpoint,args.reference_initial,args.reference_train]
    if (not Path("/content").is_dir() or any(not p.resolve().is_relative_to("/content") for p in paths if p)
            or os.environ.get("HF_HOME")!="/content/semflow-model-cache"):
        parser.error("requires reviewed ephemeral cloud paths; no host disk fallback")
    if args.output.exists():
        parser.error("report already exists; preserve previous attempt")
    if args.mode == "initial" and (args.checkpoint or args.checkpoint_sha256):
        parser.error("initial panels must use fresh interfaces, not a checkpoint")
    if args.arm=="realman" and (not args.reference_initial or (args.mode=="train" and not args.reference_train)):
        parser.error("RealMan requires verified Base pairing references")
    args.cache_dir,args.teacher_path=Path("/content/semflow-model-cache/hub"),None
    report=dict(status="running",arm=args.arm,mode=args.mode,rollouts=0,interpretation="Source-only early-learning screen")
    start=time.monotonic()
    def deadline(_signum,_frame):
        raise TimeoutError("comparison stage exceeded1050 seconds")
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(1050)
    try:
        execute(args,report)
    except Exception as error:
        report.update(status="failed",error_type=type(error).__name__,error=str(error).split("\n")[0][:600])
    finally:
        signal.alarm(0)
        report["seconds"]=time.monotonic()-start
        native_probe.write_report(report,args.output)
    print(json.dumps({k:report.get(k) for k in ("status","arm","mode","seconds","error")}),flush=True)
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
