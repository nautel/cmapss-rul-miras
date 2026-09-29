#!/usr/bin/env bash
# Phase 3 — full ablation with --full-epochs.
#
# Why: in the main grid, early stopping (patience=10 as in Table 2) fires at epoch 22-24 on
# FD001/FD003. The full model (49k params) is cut earlier than the small models, so the
# ablation comparison is NOT fair. All 5 configs must be rerun with the full 200/600 epochs
# before drawing conclusions about Table 4/5.
#
# Can run in batches: SHARD_LIST picks which shards run this time, NTOT = total shards.
#   SHARD_LIST="0 1 2 3 4 5" NTOT=12 bash cluster/run_phase3.sh
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
NTOT=${NTOT:-12}
SHARD_LIST=${SHARD_LIST:-"0 1 2 3 4 5 6 7 8 9 10 11"}
cd "$(dirname "$0")/.." || exit 1   # run from code/
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU: ${GPUS[*]} · shard [$SHARD_LIST]/$NTOT · seeds=$SEEDS =="

pids=(); k=0
for i in $SHARD_LIST; do
    g=${GPUS[$((k % ${#GPUS[@]}))]}; k=$((k+1))
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
        --ablations no_no_yes,yes_no_no,yes_no_yes,yes_yes_no --full-epochs \
        --shard "$i/$NTOT" > "$OUT/logs/ab$i.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "-- shards [$SHARD_LIST] done --"
grep -h '^RESULT' "$OUT"/logs/ab*.log | sort
