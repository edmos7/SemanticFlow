"""Prompt construction for measured dual-view Panda observations only."""

import numpy as np
import torch

from .models.native_policy import OBSERVATION_KEYS


class PandaBatchProcessor:
    def __init__(self, native, cfg):
        from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor
        height, width = map(int, cfg.data.target_image_size)
        self.processor = Qwen3VLBatchProcessor(
            model_name_or_path=str(cfg.pretrained.model_name_or_path), processor=native.backbone.processor,
            processor_call_kwargs={"min_pixels": height * width, "max_pixels": height * width,
                "text_kwargs": {"padding": "longest", "truncation": False, "return_tensors": "pt"}})

    def __call__(self, windows, device):
        samples, messages = [], []
        for window in windows:
            video = window["history_rgb"]
            if video.dtype != np.uint8 or video.shape != (2, 6, 128, 128, 3):
                raise ValueError("expected six raw RGB observations from both Panda cameras")
            # head/chest are upstream tensor slot names only. Prompt text names
            # the actual cameras; missing calibration/intrinsics are not invented.
            samples.append({"vision_type": "video", "active_views": ["head", "chest"],
                            "images": torch.from_numpy(video[0]), "chest_images": torch.from_numpy(video[1]),
                            "video_fps": torch.tensor(5.)})
            instruction = window["instruction"]
            if "<" in instruction or ">" in instruction:
                raise ValueError("instruction cannot contain model control tokens")
            messages.append([{"role": "user", "content": [
                {"type": "video"}, {"type": "video"}, {"type": "text", "text":
                 f"Task: {instruction}. The first video is from the workspace camera; "
                 "the second video is from the wrist camera. States: " + "<state>" * 6 +
                 ". Predict the next action sequence."}]}])
        batch = self.processor.encode_messages(messages, samples, add_generation_prompt=True)
        batch["answer_start_idx"] = self.processor.find_answer_start_idx(batch["input_ids"])
        if set(batch) - OBSERVATION_KEYS:
            raise ValueError("native processor produced unreviewed observation fields")
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        return {"batch": batch,
                "current_rgb": torch.from_numpy(np.stack([w["current_rgb"] for w in windows])).to(device),
                "view_valid": torch.ones((len(windows), 2), dtype=torch.bool, device=device),
                "observation_times": torch.from_numpy(np.stack([w["observation_times"] for w in windows])).to(device),
                "forecast_origin": torch.zeros(len(windows), device=device),
                "panda_states": torch.from_numpy(np.stack([w["panda_states"] for w in windows])).float().to(device)}
