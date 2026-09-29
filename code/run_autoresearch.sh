#!/usr/bin/env bash
# Autoresearch loop: 3 successive-halving stages on VAL, then open TEST once.
#
#   stage 1: 40 configs x 30-epoch budget   (FD001 + FD004, 1 seed)
#   stage 2: top 25% + mutations -> 16 configs x 70 epochs
#   stage 3: top 50% + mutations ->  8 configs x 150 epochs, 2 seeds
#   final  : top 3 -> all 4 subsets x 3 seeds x 200 epochs, TEST EVALUATION (only once)
#
# The small epoch budget is justified: phase 2 measured that running the full 600 epochs
# gives no improvement over early stopping at ~80 epochs.
#
# FD001 + FD004 are a representative pair: one subset with 1 operating condition / 1 fault
# mode, one with 6 conditions / 2 fault modes. A config good on both likely generalizes.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_ar}
NW=${NW:-12}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== autoresearch · GPU ${GPUS[*]} · $NW workers · out=$OUT =="

sweep() {   # $1=stage  $2=budget  $3=seeds
    local st=$1 bud=$2 sds=$3
    for ((i=0; i<NW; i++)); do
        local g=${GPUS[$((i % ${#GPUS[@]}))]}
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY autoresearch.py run --outdir "$OUT" --stage "$st" --budget "$bud" \
            --seeds "$sds" --subsets FD001,FD004 --root "$ROOT" --shard "$i/$NW" \
            > "$OUT/s${st}_w$i.log" 2>&1 &
    done
    wait
    echo "-- stage $st done --"
}

$PY autoresearch.py propose --outdir "$OUT" --stage 1 --n 40
sweep 1 30 0
$PY autoresearch.py select --outdir "$OUT" --stage 1 --frac 0.25 --n 16
sweep 2 70 0
$PY autoresearch.py select --outdir "$OUT" --stage 2 --frac 0.5 --n 8
sweep 3 150 0,1

echo "== opening TEST (only once) — top 3, all 4 subsets, 5 seeds =="
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY autoresearch.py final --outdir "$OUT" --stage 3 --k 3 --budget 200 \
        --seeds 0,1,2,3,4 --root "$ROOT" --shard "$i/$NW" > "$OUT/final_w$i.log" 2>&1 &
done
wait
echo "== autoresearch done =="
cat "$OUT"/final_w*.log | grep TEST | sort
