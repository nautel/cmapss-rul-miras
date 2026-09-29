#!/usr/bin/env bash
# Rerun the Miras family after the bug fix (09-2026). Three groups run in parallel:
#   A  chunk-approximation effect: FD001+FD004 x 4 variants x chunk {1,5,20} x 5 seeds
#   B  main results: 4 subsets x 13 configs (8 Miras + titans+bilstm + 3 ideas + DCNN)
#      x chunk 5 x 10 seeds
#   C  same-pipeline SOTA reference: STA-HPINN w/o physics, 10 seeds, cond-norm FD002/FD004
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_mfix}
NA=${NA:-4}; NB=${NB:-8}; NC=${NC:-2}
S10=0,1,2,3,4,5,6,7,8,9
cd "$(dirname "$0")/.." || exit 1   # run from code/
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
gpu() { echo "${GPUS[$(($1 % ${#GPUS[@]}))]}"; }
echo "== Miras fix · GPU ${GPUS[*]} · A=$NA B=$NB C=$NC workers =="

for ((i=0; i<NA; i++)); do
    CUDA_VISIBLE_DEVICES=$(gpu $i) OMP_NUM_THREADS=2 \
    $PY train_miras_fix.py --root "$ROOT" --outdir "$OUT" --subsets FD001,FD004 \
        --ideas linear_attn,deltanet,gated_deltanet,titans --chunks 1,5,20 \
        --seeds 0,1,2,3,4 --shard "$i/$NA" > "$OUT/logs/A$i.log" 2>&1 &
done
for ((i=0; i<NB; i++)); do
    CUDA_VISIBLE_DEVICES=$(gpu $((i+1))) OMP_NUM_THREADS=2 \
    $PY train_miras_fix.py --root "$ROOT" --outdir "$OUT" --chunks 5 \
        --seeds "$S10" --shard "$i/$NB" > "$OUT/logs/B$i.log" 2>&1 &
done
for ((i=0; i<NC; i++)); do
    CUDA_VISIBLE_DEVICES=$(gpu $((i+2))) OMP_NUM_THREADS=2 \
    $PY train_sta.py --root "$ROOT" --outdir "$OUT" --no-physics --seeds "$S10" \
        --subsets FD001,FD003 --shard "$i/$NC" > "$OUT/logs/Cs$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$(gpu $((i+2))) OMP_NUM_THREADS=2 \
    $PY train_sta.py --root "$ROOT" --outdir "$OUT" --no-physics --cond-norm --seeds "$S10" \
        --subsets FD002,FD004 --shard "$i/$NC" > "$OUT/logs/Cc$i.log" 2>&1 &
done
wait
echo "== Miras fix done =="
