#!/usr/bin/env bash
# Thi nghiem: bai hoc tu TSHAE/STA-HPINN co dung duoc cho khung Miras khong?
# 12 bien the x 4 bo con x 5 seed = 240 run. Moi bien the so voi `base` = Miras nguyen ban.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_m2}
NW=${NW:-6}
S=${S:-0,1,2,3,4}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== Miras-2 · GPU ${GPUS[*]} · $NW x2 worker · seeds=$S =="
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
echo "== xong Miras-2 =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
