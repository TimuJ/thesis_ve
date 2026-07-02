"""Shared server-side FlashVSR runner for the RoPE probe (gate + sweeps).

Everything here assumes: cwd == ~/repos/FlashVSR/examples/WanVSR (stock
weights paths are cwd-relative), conda env `flashvsr`, and the stock repo
untouched (`pristine-2026-07-02` tag) — the probe only ever hooks at runtime.

Input convention matches the benchmark run: 320x180 LR frames reflect-padded
to 320x192, bicubic x4 -> model I/O at 1280x768.
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

PAD_LR = 6
TH, TW = 768, 1280


def load_infer_module():
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location(
        "infer_tiny", "infer_flashvsr_v1.1_tiny.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["infer_tiny"] = m
    spec.loader.exec_module(m)
    return m


def prepare(path, n_frames):
    """First n_frames of a 320x180 video -> (1,C,F,H,W) bf16 CUDA tensor.

    Dup-pads the tail so that F % 8 == 1 exactly (stock 8n+1 rule with the
    +4 pad folded in); output frame count == F, of which n_frames are real.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    assert len(frames) == n_frames, f"only {len(frames)} frames in {path}"
    k = (5 - n_frames) % 8
    frames = frames + [frames[-1]] * (k + 4)
    ts = []
    for fr in frames:
        p = cv2.copyMakeBorder(fr, PAD_LR, PAD_LR, 0, 0, cv2.BORDER_REFLECT_101)
        up = cv2.resize(p, (TW, TH), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(up).to(torch.float32).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        ts.append(t.to(torch.bfloat16))
    vid = torch.stack(ts, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return vid.to("cuda"), vid.shape[2]


def run_once(pipe, LQ, F):
    """One stock-parameter inference; returns fp32 CPU tensor (1,C,F,H,W)."""
    torch.cuda.empty_cache()
    out = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
        seed=0, LQ_video=LQ, num_frames=F, height=TH, width=TW,
        is_full_block=False, if_buffer=True,
        topk_ratio=2.0 * 768 * 1280 / (TH * TW), kv_ratio=3.0,
        local_range=11, color_fix=True,
    )
    return out.detach().to(torch.float32).cpu()


def to_uint8_frames(video, n_real):
    """(C,T,H,W) or (1,C,T,H,W) fp32 [-1,1] -> list of n_real HxWx3 uint8."""
    v = video[0] if video.dim() == 5 else video
    v = ((v.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).numpy()  # T H W C
    return [v[i] for i in range(n_real)]
