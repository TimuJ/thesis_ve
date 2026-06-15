#!/bin/bash
set -eo pipefail
GPU=${1:-0}
DISK2="/data/disk2/timur"
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr

CUDA_VISIBLE_DEVICES=$GPU python -c "
import os, json, glob
import torch, pyiqa
import cv2
from tqdm import tqdm

device = 'cuda'
metrics = {
    'clipiqa': pyiqa.create_metric('clipiqa').to(device),
    'musiq': pyiqa.create_metric('musiq').to(device),
    'niqe': pyiqa.create_metric('niqe').to(device),
    'brisque': pyiqa.create_metric('brisque').to(device),
}

base = '/data/disk2/timur/results/uav_synthetic_mp4'
out = {}
for f in sorted(os.listdir(base)):
    if not f.endswith('.mp4'): continue
    name = f.replace('.mp4', '')
    cap = cv2.VideoCapture(os.path.join(base, f))
    scores = {k: [] for k in metrics}
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            for k, m in metrics.items():
                scores[k].append(float(m(t).item()))
    cap.release()
    out[name] = {k: sum(v)/len(v) for k,v in scores.items()}
    print(f'{name}: ' + ', '.join(f'{k}={v:.3f}' for k,v in out[name].items()))

avg = {k: sum(out[n][k] for n in out)/len(out) for k in metrics}
out['_mean'] = avg
print('Mean:', avg)

odir = '/data/disk2/timur/results/uav_synthetic_eval/nr'
os.makedirs(odir, exist_ok=True)
with open(os.path.join(odir, 'metrics.json'), 'w') as fp:
    json.dump(out, fp, indent=2)
print('Saved to', odir)
" 2>&1
