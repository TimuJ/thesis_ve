#!/bin/bash
set -eo pipefail
DISK2=/data/disk2/timur
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
python - <<PY
import sys, os
sys.path.insert(0, "/data/disk2/timur")
from pathlib import Path
import cv2
from scripts.synthetic_artefacts.flicker import apply_periodic_flicker
BASE_VIDEOS = ["hhszUXL1Cu8", "7WHI2L_FDNg"]
SEVERITIES = [0.02, 0.05, 0.10, 0.20, 0.40]
PERIOD = 15
SRC_DIR = Path("/data/disk2/timur/results/mgld_synthetic_mp4")
OUT_DIR = Path("/data/disk2/timur/results/synthetic_artefacts/flicker")
OUT_DIR.mkdir(parents=True, exist_ok=True)
for base in BASE_VIDEOS:
    src = SRC_DIR / (base + ".mp4")
    if not src.is_file():
        print("MISSING " + str(src)); continue
    for sev in SEVERITIES:
        out_name = base + "_sev" + format(sev, ".2f").replace(".", "p") + ".mp4"
        out_path = OUT_DIR / out_name
        if out_path.is_file():
            print("[skip] " + str(out_path)); continue
        cap = cv2.VideoCapture(str(src))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        wr = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok: break
            wr.write(apply_periodic_flicker(fr, i, PERIOD, sev))
            i += 1
        cap.release(); wr.release()
        print("wrote " + str(out_path) + " (" + str(i) + " frames)")
print("DONE")
PY
