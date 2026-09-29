#!/usr/bin/env bash
# Run the rest: second half of phase 3, then phase 4 once phase 3 has fully finished.
set -u
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
cd "$(dirname "$0")/.." || exit 1   # run from code/

echo "=== phase 3, shards 6-11 ==="
SHARD_LIST="6 7 8 9 10 11" NTOT=12 OUT="$OUT" bash cluster/run_phase3.sh

# phase 3 shards 0-11 = 48 files, plus 18 files from phase 2 = 66.
# Phase 4 file names end in "_L30.json", so they don't match this glob.
echo "=== waiting for shards 0-5 (other job) to finish ==="
for _ in $(seq 1 240); do
    n=$(ls "$OUT"/res_*_fe.json 2>/dev/null | wc -l)
    [ "$n" -ge 66 ] && break
    echo "  $n/66 ... $(date +%H:%M)"; sleep 60
done

echo "=== phase 4: seq_len=30 ==="
OUT="$OUT" bash cluster/run_phase4.sh
echo "=== ALL DONE ==="
