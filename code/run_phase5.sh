#!/usr/bin/env bash
# Giai doan 5 — thu ho mo hinh Miras (paper 2, arXiv 2504.13173) tren C-MAPSS.
#
# Tien xu ly dung cai TOT NHAT da chung minh o cac giai doan truoc:
#   FD001/FD003 -> Z-score toan cuc (chi 1 che do van hanh nen 2 cach la mot)
#   FD002/FD004 -> chuan hoa theo 6 che do van hanh (--cond-norm)
# epochs=200, patience=15 cho moi subset: cac giai doan truoc da do duoc rang chay
# du 600 epoch KHONG cai thien gi so voi dung som (xem REPRODUCTION.md).
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
echo "== GPU: ${GPUS[*]} · archs=$ARCHS · $NW x2 worker =="

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
echo "== xong giai doan 5 =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
