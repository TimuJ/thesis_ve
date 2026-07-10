"""Score sweep condition frames with pyiqa (IQA-PyTorch) — project convention.

PSNR/SSIM in RGB (`test_y_channel=False`, matching the DOVE eval convention
used for the verified MGLD/UAV baseline numbers — see
docs/private/server-setup.md "DOVE uses RGB PSNR/SSIM by default") and pyiqa
LPIPS. NOT scikit-image/lpips-pkg: probe numbers must be comparable with the
rest of the project.

Run ON THE SERVER in the **vsr** env (has pyiqa):
  cd ~/thesis_ve && PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python -m \
    scripts.rope_probe.score_conditions --sweep_dir ~/results/rope_probe/shift/hhsz85

For each `<cond>/` frame dir in the sweep dir: scores vs the baseline
condition's frames (self-consistency) and, if --gt_dir given, vs GT; rewrites
`<cond>.json` in place with the same key layout the analysis stage expects.
"""
import argparse
import glob
import json
import os

import cv2
import torch

BASELINE_ID = "shift0_stretch1.0"


def _load_frames(d):
    return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            for p in sorted(glob.glob(os.path.join(d, "*.png")))]


def _to_tensor(img, device):
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def center_crop(img, hw):
    """Center-crop an HxWxC array to (H, W) — used to trim the model's
    128-multiple output (e.g. 1280x768) back to the GT frame size
    (e.g. 1272x720). No-op when shapes already match."""
    h, w = hw
    H, W = img.shape[:2]
    assert H >= h and W >= w, (img.shape, hw)
    t, l = (H - h) // 2, (W - w) // 2
    return img[t:t + h, l:l + w]


def resize_to(img, hw):
    """Bicubic-resize an HxWxC array to (H, W) — for cross-resolution scoring
    (e.g. 1440x1440 output vs 1080x1080 GT). Documented confound: resizing
    the prediction before scoring measures resized-output quality."""
    h, w = hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def score_pair_lists(preds, refs, metrics, device, resize_pred=False):
    assert len(preds) == len(refs) and preds, (len(preds), len(refs))
    if resize_pred:
        preds = [resize_to(p, refs[0].shape[:2]) for p in preds]
    else:
        preds = [center_crop(p, refs[0].shape[:2]) for p in preds]
    per = {name: [] for name in metrics}
    with torch.no_grad():
        for p, r in zip(preds, refs):
            tp, tr = _to_tensor(p, device), _to_tensor(r, device)
            for name, m in metrics.items():
                per[name].append(float(m(tp, tr).item()))
    out = {}
    for name, vals in per.items():
        key = name.upper() if name != "lpips" else "LPIPS"
        out[f"{key}_mean"] = sum(vals) / len(vals)
        out[f"{key}_per_frame"] = vals
    out["backend"] = "pyiqa"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--gt_dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resize_to_ref", action="store_true",
                    help="bicubic-resize predictions to the reference size "
                         "instead of center-cropping (cross-resolution rungs)")
    args = ap.parse_args()

    import pyiqa
    device = args.device
    metrics = {
        "psnr": pyiqa.create_metric("psnr", test_y_channel=False, device=device),
        "ssim": pyiqa.create_metric("ssim", test_y_channel=False, device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
    }

    sweep = os.path.expanduser(args.sweep_dir)
    base_dir = os.path.join(sweep, BASELINE_ID)
    assert os.path.isdir(base_dir), f"no baseline frames at {base_dir}"
    baseline = _load_frames(base_dir)
    gt = _load_frames(os.path.expanduser(args.gt_dir)) if args.gt_dir else None

    for jpath in sorted(glob.glob(os.path.join(sweep, "*.json"))):
        cid = os.path.splitext(os.path.basename(jpath))[0]
        fdir = os.path.join(sweep, cid)
        if not os.path.isdir(fdir):
            print(f"skip {cid}: no frames dir", flush=True)
            continue
        frames = _load_frames(fdir)
        payload = json.load(open(jpath))
        payload["vs_baseline"] = (None if cid == BASELINE_ID else
                                  score_pair_lists(frames, baseline, metrics, device))
        payload["vs_gt"] = (score_pair_lists(frames, gt[:len(frames)], metrics, device,
                                             resize_pred=args.resize_to_ref)
                            if gt else None)
        with open(jpath, "w") as f:
            json.dump(payload, f, indent=2)
        vb = payload["vs_baseline"]
        print(f"scored {cid}: " +
              (f"PSNR={vb['PSNR_mean']:.2f} SSIM={vb['SSIM_mean']:.4f} "
               f"LPIPS={vb['LPIPS_mean']:.4f}" if vb else "baseline"), flush=True)

    print("SCORING_OK", flush=True)


if __name__ == "__main__":
    main()
