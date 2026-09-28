#!/usr/bin/env bash
# Giai doan 3 — ablation day du voi --full-epochs.
#
# Vi sao can: o luoi chinh, early stopping (patience=10 nhu Table 2) ban o epoch 22-24 tren
# FD001/FD003. Mo hinh day du (49k tham so) bi cat som hon mo hinh nho, nen so sanh ablation
# KHONG cong bang. Phai chay lai ca 5 cau hinh voi du 200/600 epoch moi ket luan duoc ve
# Table 4/5.
#
# Chay duoc nhieu dot: SHARD_LIST chon shard nao chay lan nay, NTOT la tong so shard.
#   SHARD_LIST="0 1 2 3 4 5" NTOT=12 bash run_phase3.sh
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
NTOT=${NTOT:-12}
SHARD_LIST=${SHARD_LIST:-"0 1 2 3 4 5 6 7 8 9 10 11"}
cd "$(dirname "$0")" || exit 1
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
echo "-- xong shard [$SHARD_LIST] --"
grep -h '^RESULT' "$OUT"/logs/ab*.log | sort
