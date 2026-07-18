#!/usr/bin/env bash
# SeedVR2 env standup (CPU-only phase): clone flashvsr env (torch 2.6.0+cu124,
# py310), install shipped apex wheel, try prebuilt flash-attn wheels, install
# SeedVR repo deps. Logs everything; leaves ~/logs/seedvr_env.done or .fail.
set -u
H=$HOME
LOG=~/logs/seedvr_env.log
mkdir -p ~/logs
exec > >(tee "$LOG") 2>&1

source $H/miniconda3/etc/profile.d/conda.sh

if [ ! -d "$H/miniconda3/envs/seedvr" ]; then
  echo "=== cloning flashvsr env -> seedvr"
  conda create -y -n seedvr --clone flashvsr || { echo ENV_CLONE_FAIL; touch ~/logs/seedvr_env.fail; exit 1; }
fi
conda activate seedvr
PY=$H/miniconda3/envs/seedvr/bin/python
PIP="$PY -m pip"

echo "=== torch check"
$PY -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"

echo "=== apex wheel (shipped with SeedVR2-3B weights)"
$PIP install --no-deps "$H/weights/seedvr2/SeedVR2-3B/apex-0.1-cp310-cp310-linux_x86_64.whl" || echo APEX_WHEEL_FAIL

echo "=== flash-attn prebuilt wheels (try FALSE then TRUE abi)"
FA_OK=0
for abi in FALSE TRUE; do
  for ver in 2.7.4.post1 2.7.3 2.6.3; do
    url="https://github.com/Dao-AILab/flash-attention/releases/download/v${ver}/flash_attn-${ver}+cu12torch2.6cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
    echo "--- trying $url"
    if curl -sfL --retry 3 -o /tmp/fa.whl "$url"; then
      if $PIP install --no-deps /tmp/fa.whl && $PY -c "import flash_attn; print('flash_attn', flash_attn.__version__)"; then
        FA_OK=1; break 2
      fi
    fi
  done
done
[ "$FA_OK" = "1" ] || echo FLASH_ATTN_FAIL

echo "=== SeedVR repo deps"
cd $H/repos/SeedVR
# environment.yml pins a full env; install just the pip-level extras instead.
$PIP install --no-deps omegaconf einops tensordict mediapy rotary_embedding_torch || true
$PIP install av opencv-python-headless || true

echo "=== import smoke (CPU)"
if $PY - <<'EOF'
import importlib, sys
mods = ["torch", "apex", "flash_attn", "omegaconf", "einops",
        "rotary_embedding_torch", "mediapy", "tensordict"]
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
