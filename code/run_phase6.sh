#!/usr/bin/env bash
# Giai doan 6 — hai cau hoi con lai sau khi co bang xep hang Miras:
#   (a) Miras thua SBi vi CO CHE hay chi vi IT THAM SO? -> gan dau BiLSTM cua paper 1
#       len backbone Miras (9k -> ~49k tham so, bang SBi) va do lai.
#   (b) Cac bien the phu cua paper 2 (elastic net, robust, RetNet) va BO NHO MLP SAU
#       (titans_mlp, §4) co thay doi ket luan khong?
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_miras}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-7}
EP=${EP:-200}
PAT=${PAT:-15}
# (a) 3 kien truc dan dau + dau BiLSTM
BL=${BL:-gated_deltanet+bilstm,titans+bilstm,mamba2+bilstm}
EXTRA=${EXTRA:-retnet,elastic,robust}
# (b) bo nho MLP sau: do duoc 73,9 s/epoch = 13x ban tuyen tinh (2 ep, FD001, V100).
# 200 epoch x 4 run = 16 gio -> khong kha thi. Tach ra so sanh rieng voi titans tuyen tinh
# o CUNG ngan sach 60 epoch, ghi vao thu muc rieng de khong lan voi luoi chinh.
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
echo "== xong giai doan 6 =="
grep -h '^RESULT' "$OUT"/logs/*.log | sort
