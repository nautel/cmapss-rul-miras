#!/usr/bin/env bash
# Vong lap autoresearch: 3 giai doan successive halving tren VAL, roi mo TEST mot lan.
#
#   stage 1: 40 cau hinh x ngan sach 30 epoch   (FD001 + FD004, 1 seed)
#   stage 2: top 25% + dot bien -> 16 cau hinh x 70 epoch
#   stage 3: top 50% + dot bien ->  8 cau hinh x 150 epoch, 2 seed
#   final  : top 3 -> ca 4 subset x 3 seed x 200 epoch, DANH GIA TEST (lan duy nhat)
#
# Ngan sach epoch nho la co can cu: giai doan 2 da do duoc rang chay du 600 epoch
# khong cai thien gi so voi dung som o ~80 epoch.
#
# FD001 + FD004 la cap dai dien: mot bo 1 che do van hanh / 1 kieu hong, mot bo
# 6 che do / 2 kieu hong. Cau hinh tot cho ca hai thi kha nang tong quat cao.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_ar}
NW=${NW:-12}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== autoresearch · GPU ${GPUS[*]} · $NW worker · out=$OUT =="

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
    echo "-- stage $st xong --"
}

$PY autoresearch.py propose --outdir "$OUT" --stage 1 --n 40
sweep 1 30 0
$PY autoresearch.py select --outdir "$OUT" --stage 1 --frac 0.25 --n 16
sweep 2 70 0
$PY autoresearch.py select --outdir "$OUT" --stage 2 --frac 0.5 --n 8
sweep 3 150 0,1

echo "== mo TEST (lan duy nhat) — top 3, ca 4 subset, 5 seed =="
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY autoresearch.py final --outdir "$OUT" --stage 3 --k 3 --budget 200 \
        --seeds 0,1,2,3,4 --root "$ROOT" --shard "$i/$NW" > "$OUT/final_w$i.log" 2>&1 &
done
wait
echo "== autoresearch xong =="
cat "$OUT"/final_w*.log | grep TEST | sort
