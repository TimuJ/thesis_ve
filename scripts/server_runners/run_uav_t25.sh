#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate uav_dove
cd /data/disk2/timur/repos/Upscale-A-Video
rm -rf /data/disk2/timur/results/uav_dove_torch25_test/000 2>/dev/null

echo === UAV clip 000 torch 2.5.1 transformers 4.37.0 ===
CUDA_VISIBLE_DEVICES=5 python inference_upscale_a_video.py     -i /data/disk2/timur/data/UDM10/LQ/000     -o /data/disk2/timur/results/uav_dove_torch25_test/000     -n 120 -g 6 -s 30     --no_llava --save_image 2>&1

echo === Eval ===
conda activate vsr
mkdir -p /data/disk2/timur/results/uav_t25_eval/000
src=$(find /data/disk2/timur/results/uav_dove_torch25_test/000/frame/ -maxdepth 1 -type d | tail -1)
if [ -n "$src" ] && [ "$src" != "/data/disk2/timur/results/uav_dove_torch25_test/000/frame/" ]; then
    cp $src/*.png /data/disk2/timur/results/uav_t25_eval/000/
    cd /data/disk2/timur/repos/DOVE
    python eval_metrics.py         --gt /data/disk2/timur/results/gt_clip000         --pred /data/disk2/timur/results/uav_t25_eval         --metrics psnr,ssim,lpips,dists 2>&1
else
    echo ERROR: No output frames found
fi
echo UAV_T25_DONE
