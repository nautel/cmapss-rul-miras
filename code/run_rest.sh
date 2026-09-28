#!/usr/bin/env bash
# Chay not: nua sau cua giai doan 3, roi giai doan 4 khi giai doan 3 da xong han.
set -u
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
cd "$(dirname "$0")" || exit 1

echo "=== giai doan 3, shard 6-11 ==="
SHARD_LIST="6 7 8 9 10 11" NTOT=12 OUT="$OUT" bash run_phase3.sh

# shard 0-11 cua giai doan 3 = 48 file, cong 18 file cua giai doan 2 = 66.
# Ten file cua giai doan 4 ket thuc bang "_L30.json" nen khong lot vao glob nay.
echo "=== cho shard 0-5 (job khac) xong ==="
for _ in $(seq 1 240); do
    n=$(ls "$OUT"/res_*_fe.json 2>/dev/null | wc -l)
    [ "$n" -ge 66 ] && break
    echo "  $n/66 ... $(date +%H:%M)"; sleep 60
done

echo "=== giai doan 4: seq_len=30 ==="
OUT="$OUT" bash run_phase4.sh
echo "=== TAT CA XONG ==="
