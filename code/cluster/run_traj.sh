#!/usr/bin/env bash
# Qualitative RUL plots: retrain 4 methods x 4 subsets x 5 seeds with the SAME
# train/validation engine split (split seed 0) and export predictions along every
# engine life (test engines + run-to-failure validation engines).
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_traj}
NM=${NM:-12}; NS=${NS:-4}
S=0,1,2,3,4
cd "$(dirname "$0")/.." || exit 1   # run from code/
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
gpu() { echo "${GPUS[$(($1 % ${#GPUS[@]}))]}"; }
for ((i=0; i<NM; i++)); do
    CUDA_VISIBLE_DEVICES=$(gpu $i) OMP_NUM_THREADS=2 \
    $PY train_miras_fix.py --root "$ROOT" --outdir "$OUT" --ideas dcnn,memora,titans \
        --chunks 5 --seeds "$S" --split-seed 0 --traj --shard "$i/$NM" \
        > "$OUT/logs/m$i.log" 2>&1 &
done
for ((i=0; i<NS; i++)); do
    CUDA_VISIBLE_DEVICES=$(gpu $i) OMP_NUM_THREADS=2 \
    $PY train_sta.py --root "$ROOT" --outdir "$OUT" --no-physics --seeds "$S" \
        --split-seed 0 --traj --subsets FD001,FD003 --shard "$i/$NS" > "$OUT/logs/s$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$(gpu $i) OMP_NUM_THREADS=2 \
    $PY train_sta.py --root "$ROOT" --outdir "$OUT" --no-physics --cond-norm --seeds "$S" \
        --split-seed 0 --traj --subsets FD002,FD004 --shard "$i/$NS" > "$OUT/logs/c$i.log" 2>&1 &
done
wait
echo "== trajectory runs done =="
