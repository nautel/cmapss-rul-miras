#!/usr/bin/env bash
# Phase 4 — test assumption #2 (README): seq_len.
# Table 2 states seq_len = 45, but Table 1's "Training samples" match a window of 30 EXACTLY
# on FD001 (20631 - 100x29 = 17731) and FD003 (24720 - 100x29 = 21820). Rerun with 30 to
# see which value the paper actually used.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
L=${L:-30}
NW=${NW:-6}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU: ${GPUS[*]} · seq_len=$L · $NW workers =="

launch() {
    local name=$1; shift
    local pids=()
    for ((i=0; i<NW; i++)); do
        local g=${GPUS[$((i % ${#GPUS[@]}))]}
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" --seq-len "$L" \
            --ablations yes_yes_yes --full-epochs --shard "$i/$NW" "$@" \
            > "$OUT/logs/${name}$i.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    grep -h '^RESULT' "$OUT"/logs/${name}*.log | sort
}
launch L${L} &
A=$!
launch L${L}cn --datasets FD002,FD004 --cond-norm &
B=$!
wait $A; wait $B
echo "== phase 4 done =="
