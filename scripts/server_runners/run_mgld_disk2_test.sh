#!/bin/bash
eval "$(/data/disk2/timur/miniconda3/bin/conda shell.bash hook)"
conda activate mgldvsr
cd /data/disk2/timur/repos/MGLD-VSR
export PYTHONPATH=$PWD:$PYTHONPATH

echo === MGLD UDM10 clip 000 — disk2 env test ===
mkdir -p /data/disk2/timur/data/UDM10_clip000/000
ln -sf /data/disk2/timur/data/UDM10/LQ/000/* /data/disk2/timur/data/UDM10_clip000/000/

CUDA_VISIBLE_DEVICES=6 python scripts/vsr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py     --config configs/mgldvsr/mgldvsr_512_realbasicvsr_deg.yaml     --ckpt checkpoints/mgldvsr_unet.ckpt     --vqgan_ckpt checkpoints/video_vae_cfw.ckpt     --seqs-path /data/disk2/timur/data/UDM10_clip000     --outdir /data/disk2/timur/results/mgld_disk2_env_test     --ddpm_steps 50 --dec_w 1.0 --colorfix_type adain --select_idx 0 --n_gpus 1     2>&1

echo === Eval ===
conda activate vsr
cd /data/disk2/timur/repos/DOVE
python eval_metrics.py     --gt /data/disk2/timur/results/gt_clip000     --pred /data/disk2/timur/results/mgld_disk2_env_test     --metrics psnr,ssim,lpips,dists 2>&1

echo MGLD_DISK2_TEST_DONE
