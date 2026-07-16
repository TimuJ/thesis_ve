"""Score resolution-ladder conditions vs their per-rung GT references.

Layout per clip dir (written by run_resolution_ladder):
  rung{R}_GT/       — per-rung GT crop (crop rungs); rung360 scores vs the
                      largest available GT crop with prediction resizing
  rung{R}_stock/ , rung{R}_pi/  — condition frames
  rung{R}_stock.json , rung{R}_pi.json — stub JSONs to fill (vs_gt only)

Run in the vsr env:
  PYTHONPATH=. python -m scripts.rope_probe.score_ladder \
      --clip_dir ~/results/rope_probe/res_ladder/000
"""
import argparse
import glob
import json
import os

from scripts.rope_probe.score_conditions import _load_frames, score_pair_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pyiqa
    device = args.device
    metrics = {
        "psnr": pyiqa.create_metric("psnr", test_y_channel=False, device=device),
        "ssim": pyiqa.create_metric("ssim", test_y_channel=False, device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
    }

    clip = os.path.expanduser(args.clip_dir)
    for jpath in sorted(glob.glob(os.path.join(clip, "rung*_*.json"))):
        cid = os.path.splitext(os.path.basename(jpath))[0]
        rung = cid.split("_")[0]              # e.g. "rung270"
        fdir = os.path.join(clip, cid)
        ref_dir = os.path.join(clip, rung + "_GT")
        if not (os.path.isdir(fdir) and os.path.isdir(ref_dir)):
            print(f"skip {cid}", flush=True)
            continue
        preds = _load_frames(fdir)
        refs = _load_frames(ref_dir)[: len(preds)]
        payload = json.load(open(jpath))
        mode = payload.get("condition", {}).get("score_mode")
        resize = (mode == "resize") if mode else (
            preds[0].shape[0] > refs[0].shape[0] * 1.2)
        payload["vs_gt"] = score_pair_lists(preds, refs, metrics, device,
                                            resize_pred=resize)
        with open(jpath, "w") as f:
            json.dump(payload, f, indent=2)
        g = payload["vs_gt"]
        print(f"scored {cid}: PSNR={g['PSNR_mean']:.2f} SSIM={g['SSIM_mean']:.4f} "
              f"LPIPS={g['LPIPS_mean']:.4f} (resize={resize})", flush=True)

    print("LADDER_SCORING_OK", flush=True)


if __name__ == "__main__":
    main()
