#!/usr/bin/env bash
# Experiment: do lessons from TSHAE/STA-HPINN carry over to the Miras framework?
# 12 variants x 4 subsets x 5 seeds = 240 runs. Each variant vs. `base` = vanilla Miras.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_m2}
NW=${NW:-6}
S=${S:-0,1,2,3,4}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== Miras-2 · GPU ${GPUS[*]} · $NW x2 workers · seeds=$S =="
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY train_miras2.py --root "$ROOT" --outdir "$OUT" --seeds "$S" \
        --subsets FD001,FD003 --shard "$i/$NW" > "$OUT/logs/s$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY train_miras2.py --root "$ROOT" --outdir "$OUT" --seeds "$S" \
        --subsets FD002,FD004 --cond-norm --shard "$i/$NW" > "$OUT/logs/c$i.log" 2>&1 &
done
wait
echo "== Miras-2 done =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
