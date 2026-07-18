#!/usr/bin/env python
"""Frame-wise RealESRGAN x4plus video SR — standalone, no basicsr dependency.

Vendored RRDBNet (matches basicsr.archs.rrdbnet_arch for scale=4) so the
`vsr` env (torch 1.13.1 + cv2) suffices. Preserves source fps on the writer
(fps-mismatch lesson). One frame per forward pass: ~2 GB VRAM.

Usage:
  python realesrgan_video.py --input_dir ~/synthetic_data/synthetic \
      --output_dir ~/results/realesrgan_synthetic_mp4 \
      --weights ~/weights/RealESRGAN_x4plus.pth [--videos a.mp4 b.mp4] \
      [--max_frames N] [--device cuda:0]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layer(block, n, **kw):
    return nn.Sequential(*[block(**kw) for _ in range(n)])


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """basicsr RRDBNet, scale=4 path (no pixel_unshuffle needed)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                 num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat,
                               num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def load_model(weights_path: str, device: str) -> RRDBNet:
    model = RRDBNet()
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    key = "params_ema" if "params_ema" in ckpt else "params"
    model.load_state_dict(ckpt[key], strict=True)
    model.eval().to(device)
    return model


def process_video(model, in_path, out_path, device, max_frames=None):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {in_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w * 4, h * 4))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open writer for {out_path}")
    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = torch.from_numpy(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
        out = out.squeeze(0).permute(1, 2, 0).clamp_(0, 1).cpu().numpy()
        writer.write(cv2.cvtColor((out * 255.0).round().astype(np.uint8),
                                  cv2.COLOR_RGB2BGR))
        n += 1
        if n % 500 == 0:
            dt = time.time() - t0
            print(f"  {n}/{n_total} frames  {n/dt:.1f} fps", flush=True)
        if max_frames and n >= max_frames:
            break
    cap.release()
    writer.release()
    return {"frames": n, "fps_tag": fps, "in_size": [w, h],
            "out_size": [w * 4, h * 4], "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--videos", nargs="*", default=None,
                    help="basenames; default = all .mp4 in input_dir")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(os.path.expanduser(args.weights), args.device)
    videos = args.videos or sorted(
        f for f in os.listdir(os.path.expanduser(args.input_dir))
        if f.endswith(".mp4"))
    log = {}
    for v in videos:
        in_path = os.path.join(os.path.expanduser(args.input_dir), v)
        out_path = os.path.join(os.path.expanduser(args.output_dir), v)
        print(f"[realesrgan] {v}", flush=True)
        log[v] = process_video(model, in_path, out_path, args.device,
                               args.max_frames)
        print(f"[realesrgan] {v} done: {log[v]}", flush=True)
    tag = "-".join(v.split(".")[0][:4] for v in videos)
    with open(os.path.join(os.path.expanduser(args.output_dir),
                           f"_run_log_{tag}.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
