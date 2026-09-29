#!/usr/bin/env bash
# Test the "narrow bottleneck" hypothesis taken from TSHAE + STA-HPINN.
#   4 variants x 4 latent widths (2/3/8/32) x 4 subsets x 10 seeds = 640 runs
# The variants isolate each component's contribution: reconstruction and triplet.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_rve}
NW=${NW:-6}
V=${V:-full,notrip,norecon,plain}
L=${L:-2,3,8,32}
S=${S:-0,1,2,3,4,5,6,7,8,9}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== RVE · GPU ${GPUS[*]} · $NW x2 workers · variants=$V latents=$L =="
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY train_rve.py --root "$ROOT" --outdir "$OUT" --seeds "$S" --variants "$V" \
        --latents "$L" --subsets FD001,FD003 --shard "$i/$NW" > "$OUT/logs/s$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY train_rve.py --root "$ROOT" --outdir "$OUT" --seeds "$S" --variants "$V" \
        --latents "$L" --subsets FD002,FD004 --cond-norm --shard "$i/$NW" \
        > "$OUT/logs/c$i.log" 2>&1 &
done
wait
echo "== RVE done =="
