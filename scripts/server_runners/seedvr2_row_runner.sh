#!/usr/bin/env bash
# SeedVR2 4th-row pipeline: waits for the VBench SOTA queue to release the
# GPUs, then splits the 5 LR videos into 33-frame chunks and runs SeedVR2-3B
# with sequence parallelism over BOTH GPUs (sp_size=2), one torchrun per
# video (model loaded once per video), then reassembles full-length outputs.
# Markers: ~/logs/seedvr2_row.done / ~/logs/seedvr2_row.fail
set -u
H=$HOME
PYV=$H/miniconda3/envs/vsr/bin/python
TR=$H/miniconda3/envs/seedvr310/bin/torchrun
CHUNKS=$H/seedvr2_chunks
OUTS=$H/seedvr2_out
FINAL=$H/results/seedvr2_synthetic_mp4
mkdir -p ~/logs "$CHUNKS" "$OUTS" "$FINAL"
rm -f ~/logs/seedvr2_row.done ~/logs/seedvr2_row.fail

echo "waiting for VBench queue to finish..."
until [ -f ~/logs/vbench_consistency.done ]; do sleep 120; done
echo "GPUs released — starting SeedVR2 row"

$PYV ~/thesis_ve/scripts/server_runners/seedvr2_chunked.py split \
  --input_dir $H/synthetic_data/synthetic --chunks_root "$CHUNKS" \
  2>&1 | tee ~/logs/seedvr2_split.log

fail=0
for vd in "$CHUNKS"/*/; do
  name=$(basename "$vd")
  outd="$OUTS/$name"
  mkdir -p "$outd"
  n_in=$(ls "$vd"/*.mp4 2>/dev/null | wc -l)
  n_out=$(ls "$outd" 2>/dev/null | wc -l)
  if [ "$n_out" -ge "$n_in" ] && [ "$n_in" -gt 0 ]; then
    echo "[skip] $name already processed ($n_out/$n_in)"
    continue
  fi
  echo "=== SeedVR2 $name ($n_in chunks)"
  cd $H/repos/SeedVR
  PYTHONPATH=$H/repos/SeedVR PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $TR --nproc_per_node=2 --master_port=29521 \
    projects/inference_seedvr2_3b.py \
    --video_path "$vd" --output_dir "$outd" \
    --res_h 720 --res_w 1280 --sp_size 2 \
    2>&1 | tee ~/logs/seedvr2_${name}.log
  rc=${PIPESTATUS[0]}
  echo "=== $name EXIT=$rc" | tee -a ~/logs/seedvr2_row.log
  [ "$rc" -ne 0 ] && fail=1
done

$PYV ~/thesis_ve/scripts/server_runners/seedvr2_chunked.py assemble \
  --chunks_root "$CHUNKS" --outputs_root "$OUTS" --final_dir "$FINAL" \
  2>&1 | tee ~/logs/seedvr2_assemble.log

if [ "$fail" -eq 0 ] && [ "$(ls "$FINAL"/*.mp4 2>/dev/null | wc -l)" -ge 5 ]; then
  touch ~/logs/seedvr2_row.done
else
  touch ~/logs/seedvr2_row.fail
fi
echo PIPELINE_END
