#!/usr/bin/env bash
# Phase 5 — try the Miras model family (paper 2, arXiv 2504.13173) on C-MAPSS.
#
# Preprocessing uses the BEST option shown in earlier phases:
#   FD001/FD003 -> global Z-score (only 1 operating condition, so both are identical)
#   FD002/FD004 -> normalization per the 6 operating conditions (--cond-norm)
# epochs=200, patience=15 for every subset: earlier phases showed that running the
# full 600 epochs gives NO improvement over early stopping (see REPRODUCTION.md).
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_miras}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-7}
EP=${EP:-200}
PAT=${PAT:-15}
ARCHS=${ARCHS:-linear_attn,mamba2,deltanet,gated_deltanet,titans,moneta,yaad,memora}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU: ${GPUS[*]} · archs=$ARCHS · $NW x2 workers =="

launch() {
    local name=$1; shift
    for ((i=0; i<NW; i++)); do
        local g=${GPUS[$((i % ${#GPUS[@]}))]}
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
            --archs "$ARCHS" --epochs "$EP" --patience "$PAT" \
            --shard "$i/$NW" "$@" > "$OUT/logs/${name}$i.log" 2>&1 &
    done
}
launch simple --datasets FD001,FD003
launch cond   --datasets FD002,FD004 --cond-norm
wait
echo "== phase 5 done =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
