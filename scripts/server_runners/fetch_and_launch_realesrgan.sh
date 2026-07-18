#!/usr/bin/env bash
# Resume-download RealESRGAN weights until torch-loadable, then launch the
# full 5-video frame-wise SR run split across both GPUs (tmux sessions
# esrgan0/esrgan1). Run inside tmux: survives SSH drops on the flaky link.
W=$HOME/weights/RealESRGAN_x4plus.pth
URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
PY=$HOME/miniconda3/envs/vsr/bin/python
mkdir -p ~/logs ~/weights

pkill -u "$(whoami)" -x curl 2>/dev/null
rm -f ~/logs/esrgan_weights.ok
for i in $(seq 1 120); do
  curl -sL -C - --retry 3 -o "$W" "$URL"
  if $PY -c "
import sys, torch
try:
    d = torch.load('$W', map_location='cpu', weights_only=True)
    sys.exit(0 if 'params_ema' in d else 1)
except Exception:
    sys.exit(1)"; then
    echo "WEIGHTS_OK size=$(stat -c%s "$W")" | tee ~/logs/esrgan_weights.ok
    break
  fi
  echo "attempt $i size=$(stat -c%s "$W" 2>/dev/null)"
  sleep 5
done
[ -f ~/logs/esrgan_weights.ok ] || { echo GIVEUP | tee ~/logs/esrgan_weights.fail; exit 1; }

mkdir -p ~/results/realesrgan_synthetic_mp4
tmux new-session -d -s esrgan0 "CUDA_VISIBLE_DEVICES=0 $PY ~/realesrgan_video.py \
  --input_dir ~/synthetic_data/synthetic \
  --output_dir ~/results/realesrgan_synthetic_mp4 \
  --weights $W --videos mJog8DlRk_4.mp4 hhszUXL1Cu8.mp4 \
  2>&1 | tee ~/logs/realesrgan_gpu0.log; touch ~/logs/realesrgan_gpu0.done"
tmux new-session -d -s esrgan1 "CUDA_VISIBLE_DEVICES=1 $PY ~/realesrgan_video.py \
  --input_dir ~/synthetic_data/synthetic \
  --output_dir ~/results/realesrgan_synthetic_mp4 \
  --weights $W --videos 7WHI2L_FDNg.mp4 BrRLKMbBTYQ.mp4 KZ8p6b1zJ9U.mp4 \
  2>&1 | tee ~/logs/realesrgan_gpu1.log; touch ~/logs/realesrgan_gpu1.done"
echo LAUNCHED
