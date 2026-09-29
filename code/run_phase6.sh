#!/usr/bin/env bash
# Phase 6 — two open questions after the Miras leaderboard:
#   (a) Does Miras trail SBi due to its MECHANISM or just FEWER PARAMS? -> attach paper 1's
#       BiLSTM head to the Miras backbone (9k -> ~49k params, like SBi) and re-measure.
#   (b) Do paper 2's secondary variants (elastic net, robust, RetNet) and DEEP MLP MEMORY
#       (titans_mlp, §4) change the conclusion?
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_miras}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-7}
EP=${EP:-200}
PAT=${PAT:-15}
# (a) top 3 architectures + BiLSTM head
BL=${BL:-gated_deltanet+bilstm,titans+bilstm,mamba2+bilstm}
EXTRA=${EXTRA:-retnet,elastic,robust}
# (b) deep MLP memory: measured 73.9 s/epoch = 13x the linear version (2 ep, FD001, V100).
# 200 epochs x 4 runs = 16 hours -> infeasible. Compare separately with linear titans at
# the SAME 60-epoch budget, in a separate dir so it doesn't mix with the main grid.
MLP=${MLP:-titans_mlp,titans}
MLP_OUT=${MLP_OUT:-/home/lab/letuan/runs/rul_mlp}
MLP_EP=${MLP_EP:-60}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU: ${GPUS[*]} =="

launch() {
    local name=$1 archs=$2 nw=$3; shift 3
    for ((i=0; i<nw; i++)); do
        local g=${GPUS[$((i % ${#GPUS[@]}))]}
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY run_all.py --root "$ROOT" --outdir "$OUT" --archs "$archs" \
            --epochs "$EP" --patience "$PAT" --shard "$i/$nw" "$@" \
            > "$OUT/logs/${name}$i.log" 2>&1 &
    done
}
launch bl_s   "$BL"    4 --seeds "$SEEDS" --datasets FD001,FD003
launch bl_c   "$BL"    4 --seeds "$SEEDS" --datasets FD002,FD004 --cond-norm
launch ex_s   "$EXTRA" 3 --seeds 0,1      --datasets FD001,FD003
launch ex_c   "$EXTRA" 3 --seeds 0,1      --datasets FD002,FD004 --cond-norm
mkdir -p "$MLP_OUT/logs"
for i in 0 1; do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$MLP_OUT" --archs "$MLP" \
        --epochs "$MLP_EP" --patience 60 --seeds 0,1 --datasets FD001 \
        --shard "$i/2" > "$MLP_OUT/logs/mlp$i.log" 2>&1 &
done
wait
echo "== phase 6 done =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
