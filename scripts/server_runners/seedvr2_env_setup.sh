#!/usr/bin/env bash
# SeedVR2 env standup v2: fresh py3.10 env (the shipped apex + prebuilt
# flash-attn wheels are cp310; the flashvsr env is py3.11 — clone unusable).
# CPU-only phase; logs to ~/logs/seedvr_env.log; markers seedvr_env.{done,fail}.
set -u
H=$HOME
LOG=~/logs/seedvr_env.log
mkdir -p ~/logs
exec > >(tee "$LOG") 2>&1

source $H/miniconda3/etc/profile.d/conda.sh
rm -f ~/logs/seedvr_env.done ~/logs/seedvr_env.fail

if [ ! -x "$H/miniconda3/envs/seedvr310/bin/python" ]; then
  echo "=== creating fresh py3.10 env"
  conda create -y -n seedvr310 python=3.10 || { echo ENV_CREATE_FAIL; touch ~/logs/seedvr_env.fail; exit 1; }
fi
PY=$H/miniconda3/envs/seedvr310/bin/python
PIP="$PY -m pip"

echo "=== torch 2.6.0 cu124 (cp310)"
$PY -c "import torch" 2>/dev/null || \
  $PIP install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 \
  || { echo TORCH_FAIL; touch ~/logs/seedvr_env.fail; exit 1; }
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'abi', torch._C._GLIBCXX_USE_CXX11_ABI)"

echo "=== apex wheel (shipped, cp310)"
$PIP install --no-deps "$H/weights/seedvr2/SeedVR2-3B/apex-0.1-cp310-cp310-linux_x86_64.whl" || echo APEX_WHEEL_FAIL

echo "=== flash-attn prebuilt wheels (proper filenames this time)"
FA_OK=0
for abi in FALSE TRUE; do
  for ver in 2.7.4.post1 2.7.3 2.6.3; do
    name="flash_attn-${ver}+cu12torch2.6cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
    url="https://github.com/Dao-AILab/flash-attention/releases/download/v${ver}/${name}"
    echo "--- trying $name"
    if curl -sfL --retry 3 -o "/tmp/$name" "$url"; then
      if $PIP install --no-deps "/tmp/$name" && $PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"; then
        FA_OK=1; rm -f "/tmp/$name"; break 2
      fi
    fi
    rm -f "/tmp/$name"
  done
done
[ "$FA_OK" = "1" ] || echo FLASH_ATTN_FAIL

echo "=== SeedVR repo deps"
$PIP install omegaconf einops einops-exts tensordict mediapy rotary_embedding_torch av opencv-python-headless diffusers transformers accelerate safetensors || true
$PIP install "setuptools<81"

echo "=== import smoke (CPU)"
if $PY - <<'EOF'
import importlib, sys
mods = ["torch", "apex", "flash_attn", "omegaconf", "einops",
        "rotary_embedding_torch", "mediapy", "tensordict", "diffusers"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        bad.append((m, str(e)[:120]))
for m, e in bad:
    print("IMPORT_FAIL", m, e)
sys.exit(1 if bad else 0)
EOF
then
  echo SEEDVR_ENV_OK; touch ~/logs/seedvr_env.done
else
  echo SEEDVR_ENV_PARTIAL; touch ~/logs/seedvr_env.fail
fi
