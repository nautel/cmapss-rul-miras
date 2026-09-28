#!/usr/bin/env bash
# #6 — chay cac baseline kinh dien cua van lieu trong CUNG pipeline.
# Cung tien xu ly (z-score cho FD001/FD003, cond-norm cho FD002/FD004), cung seq_len 45,
# cung nhan cat 125, cung tap val, cung so seed. Chi khac kien truc.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_bl}
SEEDS=${SEEDS:-0,1,2,3,4}
NW=${NW:-5}
EP=${EP:-200}
PAT=${PAT:-20}
ARCHS=${ARCHS:-bl:dcnn,bl:lstm,bl:bilstm,bl:gru,bl:tcn,bl:cnn_lstm,bl:mlp}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== baselines · GPU ${GPUS[*]} · $NW x2 worker · seeds=$SEEDS =="
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" --archs "$ARCHS" \
        --epochs "$EP" --patience "$PAT" --datasets FD001,FD003 --shard "$i/$NW" \
        > "$OUT/logs/s$i.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" --archs "$ARCHS" \
        --epochs "$EP" --patience "$PAT" --datasets FD002,FD004 --cond-norm --shard "$i/$NW" \
        > "$OUT/logs/c$i.log" 2>&1 &
done
wait
echo "== xong baselines =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
