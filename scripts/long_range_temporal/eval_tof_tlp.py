"""Long-range temporal-consistency evaluation: tOF (pixel L2) and tLP (LPIPS)
across multiple frame gaps k. See Chu et al. 2020 (TecoGAN) for the standard
definitions; this extends them to arbitrary k via the same RAFT + FB-consistency
mask + warped-difference pipeline.

Output per video: JSON with {tof[k], tlp[k], n_pairs_used[k], mean_mask_coverage[k]}.
Lower is better for both metrics. The per-k curve characterizes long-range
temporal stability — adjacent-frame metrics (k=1) cannot expose chunk-boundary
artefacts or slow drift over minutes.
"""
import argparse
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import lpips
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def fb_consistency_mask(flow_fwd, flow_bwd, tol=1.0):
    """Forward-backward consistency mask. Inputs are (1,2,H,W). Returns (1,1,H,W)."""
    _, _, H, W = flow_fwd.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=flow_fwd.device),
                            torch.arange(W, device=flow_fwd.device), indexing="ij")
    gx = xx.float() + flow_fwd[0, 0]
    gy = yy.float() + flow_fwd[0, 1]
    nx = 2 * gx / max(W - 1, 1) - 1
    ny = 2 * gy / max(H - 1, 1) - 1
    grid = torch.stack([nx, ny], dim=-1).unsqueeze(0)
    bwd_at_warp = F.grid_sample(flow_bwd, grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
    cycle = flow_fwd + bwd_at_warp
    diff = torch.norm(cycle, dim=1, keepdim=True)
    return (diff < tol).float()


def backward_warp(img, flow):
    """Warp img (1,3,H,W) using flow (1,2,H,W)."""
    _, _, H, W = img.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=img.device),
                            torch.arange(W, device=img.device), indexing="ij")
    gx = xx.float() + flow[0, 0]
    gy = yy.float() + flow[0, 1]
    nx = 2 * gx / max(W - 1, 1) - 1
    ny = 2 * gy / max(H - 1, 1) - 1
    grid = torch.stack([nx, ny], dim=-1).unsqueeze(0)
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)


def frame_to_tensor(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def process_one_video(vpath, k_values, max_pairs, fb_tol, raft_model, raft_transforms,
                      lpips_model, device):
    cap = cv2.VideoCapture(vpath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    cap.release()
    print(f"  frames={n_frames}, fps={fps}")

    per_k = {"tof": {}, "tlp": {}, "n_pairs_used": {},
             "n_pairs_valid_mask": {}, "mean_mask_coverage": {}}
    for k in k_values:
        n_possible = max(0, n_frames - k)
        if n_possible == 0:
            per_k["tof"][k] = None
            per_k["tlp"][k] = None
            per_k["n_pairs_used"][k] = 0
            continue
        stride = max(1, n_possible // max_pairs)
        t_indices = list(range(0, n_possible, stride))[:max_pairs]
        cap = cv2.VideoCapture(vpath)
        tof_vals, tlp_vals, mask_cov = [], [], []
        for idx, t0 in enumerate(t_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, t0)
            ok, fr0 = cap.read()
            if not ok:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, t0 + k)
            ok, fr1 = cap.read()
            if not ok:
                break
            im0 = frame_to_tensor(fr0, device)
            im1 = frame_to_tensor(fr1, device)
            with torch.no_grad():
                a, b = raft_transforms(im0, im1)
                flow_fwd = raft_model(a, b)[-1]
                flow_bwd = raft_model(b, a)[-1]
            mask = fb_consistency_mask(flow_fwd, flow_bwd, tol=fb_tol)
            warped = backward_warp(im1, flow_fwd)
            cov = mask.mean().item()
            mask_cov.append(cov)
            if cov < 1e-6:
                continue
            diff = (im0 - warped) * mask
            tof = (diff.pow(2).sum() / (mask.sum() * 3 + 1e-9)).sqrt().item()
            tof_vals.append(tof)
            im0_n = (im0 * 2 - 1) * mask
            wp_n = (warped * 2 - 1) * mask
            with torch.no_grad():
                tlp = lpips_model(im0_n, wp_n).item()
            tlp_vals.append(tlp)
            if (idx + 1) % 50 == 0:
                print(f"  k={k}: {idx+1}/{len(t_indices)} pairs, "
                      f"tOF_mean={np.mean(tof_vals):.4f}, tLP_mean={np.mean(tlp_vals):.4f}, "
                      f"cov_mean={np.mean(mask_cov):.2f}")
        cap.release()
        per_k["tof"][k] = float(np.mean(tof_vals)) if tof_vals else None
        per_k["tlp"][k] = float(np.mean(tlp_vals)) if tlp_vals else None
        per_k["n_pairs_used"][k] = len(t_indices)
        per_k["n_pairs_valid_mask"][k] = len(tof_vals)
        per_k["mean_mask_coverage"][k] = float(np.mean(mask_cov)) if mask_cov else 0
        print(f"  k={k} done: tOF={per_k['tof'][k]}, tLP={per_k['tlp'][k]}, "
              f"n_pairs={len(tof_vals)}/{len(t_indices)}, cov={per_k['mean_mask_coverage'][k]:.2f}")
    return n_frames, fps, per_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_path", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--k_values", default="1,5,10,30,60,120")
    ap.add_argument("--max_pairs", type=int, default=200)
    ap.add_argument("--fb_tol", type=float, default=1.0)
    ap.add_argument("--lpips_backbone", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    device = torch.device(args.device)
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading RAFT...")
    weights = Raft_Large_Weights.DEFAULT
    raft_model = raft_large(weights=weights, progress=False).to(device).eval()
    raft_transforms = weights.transforms()

    print(f"Loading LPIPS ({args.lpips_backbone})...")
    lpips_model = lpips.LPIPS(net=args.lpips_backbone).to(device).eval()

    video_files = sorted(
        f for f in os.listdir(args.videos_path)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )
    if not video_files:
        sys.exit(f"No videos in {args.videos_path}")

    for vname in video_files:
        vpath = os.path.join(args.videos_path, vname)
        base = os.path.splitext(vname)[0]
        out_file = os.path.join(args.output_path, base + "_tof_tlp.json")
        if os.path.isfile(out_file):
            print(f"[skip] {out_file} exists")
            continue
        print(f"\n=== {vname} ===")
        n_frames, fps, per_k = process_one_video(
            vpath, k_values, args.max_pairs, args.fb_tol,
            raft_model, raft_transforms, lpips_model, device,
        )
        payload = {
            "video_path": vpath,
            "n_frames": n_frames,
            "fps": fps,
            "k_values": k_values,
            "max_pairs": args.max_pairs,
            "fb_tol": args.fb_tol,
            "lpips_backbone": args.lpips_backbone,
            "timestamp": datetime.now().isoformat(),
            **per_k,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print("  wrote " + out_file)


if __name__ == "__main__":
    main()
