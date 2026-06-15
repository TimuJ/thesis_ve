#!/bin/bash
# Generate the 6 flip artefacts on the 5 bases (5x6x5 = 150 new clips).
# Pure CPU, fast — should finish in ~30 min.
set -uo pipefail
DISK2=/data/disk2/timur
eval "$($DISK2/miniconda3/bin/conda shell.bash hook)"
conda activate vsr
cd $DISK2
export PYTHONPATH="$DISK2:${PYTHONPATH:-}"
python scripts/synthetic_artefacts/generate_all.py 2>&1 | tee /tmp/gen_flip.log
touch /tmp/gen_flip.done
