#!/usr/bin/env bash
# Phase 2 — run AFTER the main grid (run_repro.sh) finishes.
# Goal: disentangle two possible causes of the gap vs. the paper
#   (a) too-strict early stop -> --full-epochs (run the full 200/600 epochs of Table 2)
#   (b) global Z-score        -> --cond-norm  (normalize per the 6 operating conditions)
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-6}
cd "$(dirname "$0")/.." || exit 1   # run from code/
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU: ${GPUS[*]} · $NW workers · seeds=$SEEDS =="

launch() {   # $1=label  $2..=extra args
    local name=$1; shift
    local pids=()
    for ((i=0; i<NW; i++)); do
        local g=${GPUS[$((i % ${#GPUS[@]}))]}
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
            --shard "$i/$NW" "$@" > "$OUT/logs/${name}$i.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "-- $name done --"
    grep -h '^RESULT' "$OUT"/logs/${name}*.log | sort
}

# The two groups run in PARALLEL (2 x NW processes). The longest job (FD004, 600 epochs,
# ~65 min) sets the wall-clock floor; running sequentially doubles the time for no gain.
# (a) full epochs, global Z-score — all 4 subsets, full model only
launch fe --ablations yes_yes_yes --full-epochs &
A=$!
# (b) full epochs + op-condition normalization — FD002/FD004 only (FD001/FD003: 1 condition)
launch fecn --datasets FD002,FD004 --ablations yes_yes_yes --full-epochs --cond-norm &
B=$!
wait $A; wait $B

echo "== tables + figures (refresh) =="
$PY report.py  --outdir "$OUT" --md "$OUT/REPRODUCTION.md"
$PY figures.py --outdir "$OUT"
echo "== phase 2 done =="
