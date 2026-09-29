#!/usr/bin/env bash
# #8 — run isolated ablations of the final config. Call after autoresearch finishes.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
AR=${AR:-/home/lab/letuan/runs/rul_ar2}
OUT=${OUT:-/home/lab/letuan/runs/rul_abl}
NW=${NW:-10}
SEEDS=${SEEDS:-0,1,2,3,4}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
$PY ablate.py --ar-dir "$AR" --list
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY ablate.py --ar-dir "$AR" --outdir "$OUT" --seeds "$SEEDS" --shard "$i/$NW" \
        > "$OUT/logs/abl$i.log" 2>&1 &
done
wait
echo "== ablation done =="
