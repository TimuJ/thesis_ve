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

def pad_amounts(h, w):
    """Reflect-pad amounts to make LR dims multiples of 32 (so x4 output is a
    128-multiple), split evenly (extra px to bottom/right). Returns
    ((top, bottom, left, right), (TH, TW)) with TH/TW the model I/O dims.
    320x180 -> pad (6,6,0,0), I/O 1280x768; UDM10 318x180 -> (6,6,1,1)."""
    th_lr = -(-h // 32) * 32
    tw_lr = -(-w // 32) * 32
    ph, pw = th_lr - h, tw_lr - w
    return (ph // 2, ph - ph // 2, pw // 2, pw - pw // 2), (th_lr * 4, tw_lr * 4)


def load_infer_module():
    sys.path.insert(0, os.getcwd())
    spec = importlib.util.spec_from_file_location(
        "infer_tiny", "infer_flashvsr_v1.1_tiny.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["infer_tiny"] = m
    spec.loader.exec_module(m)
    return m


def prepare(path, n_frames, device="cuda"):
    """First n_frames of an LR video file OR a PNG frames dir ->
    ((1,C,F,H,W) bf16 tensor on `device`, F, (TH, TW)).

    Any LR resolution: reflect-pads each dim to a multiple of 32 (so the x4
    output is a 128-multiple; cf. pad_amounts). Dup-pads the tail so that
    F % 8 == 1 exactly (stock 8n+1 rule with the +4 pad folded in); of the
    F frames, the first n_frames are real.
    """
    if os.path.isdir(path):
        import glob
        paths = sorted(glob.glob(os.path.join(path, "*.png")))[:n_frames]
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in paths]
    else:
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < n_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
    assert len(frames) == n_frames, f"only {len(frames)} frames in {path}"
    (pt, pb, pl, pr), (TH, TW) = pad_amounts(*frames[0].shape[:2])
    k = (5 - n_frames) % 8
    frames = frames + [frames[-1]] * (k + 4)
    ts = []
    for fr in frames:
        p = cv2.copyMakeBorder(fr, pt, pb, pl, pr, cv2.BORDER_REFLECT_101)
        up = cv2.resize(p, (TW, TH), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(up).to(torch.float32).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        ts.append(t.to(torch.bfloat16))
    vid = torch.stack(ts, 0).permute(1, 0, 2, 3).unsqueeze(0)
    # tiny (non-long) pipeline expects the LQ tensor on the GPU
    return vid.to(device), vid.shape[2], (TH, TW)


def run_once(pipe, LQ, F, th, tw):
    """One stock-parameter inference; returns fp32 CPU tensor (1,C,F,H,W)."""
    torch.cuda.empty_cache()
    out = pipe(
        prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1,
        seed=0, LQ_video=LQ, num_frames=F, height=th, width=tw,
        is_full_block=False, if_buffer=True,
        topk_ratio=2.0 * 768 * 1280 / (th * tw), kv_ratio=3.0,
        local_range=11, color_fix=True,
    )
    return out.detach().to(torch.float32).cpu()


def to_uint8_frames(video, n_real):
    """(C,T,H,W) or (1,C,T,H,W) fp32 [-1,1] -> list of n_real HxWx3 uint8."""
    v = video[0] if video.dim() == 5 else video
    v = ((v.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).numpy()  # T H W C
    return [v[i] for i in range(n_real)]
