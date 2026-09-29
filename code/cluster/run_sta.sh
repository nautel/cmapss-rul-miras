#!/usr/bin/env bash
# Re-implement STA-HPINN (arXiv:2405.12377) + ablation of the physics-informed part.
#   group `phys`   : full version, 10 seeds (the paper averages 10 runs)
#   group `nophys` : physics loss removed, 5 seeds -> measures AHPINN's REAL contribution
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sta}
NW=${NW:-3}
cd "$(dirname "$0")/.." || exit 1   # run from code/
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== STA-HPINN · GPU ${GPUS[*]} · $NW x2 workers =="
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY train_sta.py --root "$ROOT" --outdir "$OUT" --seeds 0,1,2,3,4,5,6,7,8,9 \
        --shard "$i/$NW" > "$OUT/logs/phys$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY train_sta.py --root "$ROOT" --outdir "$OUT" --seeds 0,1,2,3,4 --no-physics \
        --shard "$i/$NW" > "$OUT/logs/nophys$i.log" 2>&1 &
done
wait
echo "== STA done =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
