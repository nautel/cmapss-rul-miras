#!/usr/bin/env bash
# #2 — run the main methods with the SAME 10 seeds as the baselines, for a fair comparison.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_main10}
SEEDS=${SEEDS:-0,1,2,3,4,5,6,7,8,9}
NW=${NW:-4}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
        --ablations yes_yes_yes,yes_yes_no,no_no_yes --epochs 200 --patience 20 \
        --datasets FD001,FD003 --shard "$i/$NW" > "$OUT/logs/s$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
        --ablations yes_yes_yes,yes_yes_no,no_no_yes --epochs 200 --patience 20 \
        --datasets FD002,FD004 --cond-norm --shard "$i/$NW" > "$OUT/logs/c$i.log" 2>&1 &
done
wait
echo "== main10 done =="
