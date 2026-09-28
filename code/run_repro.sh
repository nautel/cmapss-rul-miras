#!/usr/bin/env bash
# Chay toan bo tai lap tren cassio:
#   CASSIO_GPUS=2 CASSIO_CPUS=12 CASSIO_MEM=48G \
#     ./cassio.sh train --cwd '~/code/rul_sbi' 'bash run_repro.sh'
#
# Mo hinh rat nho (49k tham so) nen nut co chai la do TRE PHAT LENH CUDA tren may
# dang tai ~80/80, khong phai tinh toan. Vi vay chay NHIEU tien trinh song song
# tren cung GPU thay vi mot tien trinh nhanh hon.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-6}                       # so tien trinh song song
EP=${EPOCHS:+--epochs $EPOCHS}    # ghi de so epoch (chi de chay thu)
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT" "$OUT/logs"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU kha dung: ${GPUS[*]} · $NW worker · seeds=$SEEDS · out=$OUT =="

pids=()
for ((i=0; i<NW; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
        --shard "$i/$NW" $EP > "$OUT/logs/shard$i.log" 2>&1 &
    pids+=($!)
    echo "  worker $i -> GPU $g (pid ${pids[-1]})"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "== luoi chinh xong (fail=$fail) =="
grep -h '^RESULT' "$OUT"/logs/shard*.log | sort

echo "== bien the chuan hoa theo che do van hanh (FD002/FD004) =="
for ((i=0; i<2; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
        --datasets FD002,FD004 --ablations yes_yes_yes --cond-norm $EP \
        --shard "$i/2" > "$OUT/logs/cn$i.log" 2>&1 &
done
wait

echo "== bang + hinh =="
$PY report.py  --outdir "$OUT" --md "$OUT/REPRODUCTION.md"
$PY figures.py --outdir "$OUT"
echo "== xong =="
