#!/bin/bash
set -e
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"

echo === Step 1: DOVE NR eval on MGLD synthetic ===
conda activate vsr
cd /data/disk2/timur/repos/DOVE
mkdir -p /data/disk2/timur/results/mgld_synthetic_eval

for vid in 7WHI2L_FDNg BrRLKMbBTYQ KZ8p6b1zJ9U hhszUXL1Cu8 mJog8DlRk_4; do
    echo Evaluating $vid...
    python eval_metrics.py         --pred /data/disk2/timur/results/mgld_synthetic/$vid         --metrics clipiqa         --out /data/disk2/timur/results/mgld_synthetic_eval/$vid 2>&1 | tail -5
done

echo === Step 2: Clone VBench 2.0 ===
cd /data/disk2/timur/repos
rm -rf VBench 2>/dev/null
git clone --depth 1 https://github.com/Vchitect/VBench.git 2>&1 | tail -5

echo === Step 3: Convert MGLD synthetic to MP4 ===
conda activate vsr
mkdir -p /data/disk2/timur/results/mgld_synthetic_mp4
python -c "
import cv2, os
src_root = '/data/disk2/timur/results/mgld_synthetic'
dst_root = '/data/disk2/timur/results/mgld_synthetic_mp4'
for vid in sorted(os.listdir(src_root)):
    src = os.path.join(src_root, vid)
    if not os.path.isdir(src): continue
    dst = os.path.join(dst_root, vid + '.mp4')
    frames = sorted([f for f in os.listdir(src) if f.endswith('.png')])
    img = cv2.imread(os.path.join(src, frames[0]))
    h, w = img.shape[:2]
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for f in frames:
        out.write(cv2.imread(os.path.join(src, f)))
    out.release()
    print(f'{vid}: {len(frames)} frames -> MP4')
"

echo === Step 4: Test VBench 2.0 on MGLD synthetic ===
conda activate vbench
if [ -f /data/disk2/timur/repos/VBench/vbench2_beta_long/eval_long.py ]; then
    echo VBench 2.0 available
    cd /data/disk2/timur/repos/VBench
    CUDA_VISIBLE_DEVICES=2 python vbench2_beta_long/eval_long.py         --videos_path /data/disk2/timur/results/mgld_synthetic_mp4/hhszUXL1Cu8.mp4         --dimension imaging_quality         --mode long_custom_input         --dev_flag         --output_path /data/disk2/timur/results/vbench2_mgld 2>&1
    echo VBENCH2_DONE
else
    echo VBench 2.0 clone failed — skipping
fi

echo ALL_DONE
