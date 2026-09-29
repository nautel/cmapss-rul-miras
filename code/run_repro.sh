#!/usr/bin/env bash
# Run the full reproduction on cassio:
#   CASSIO_GPUS=2 CASSIO_CPUS=12 CASSIO_MEM=48G \
#     ./cassio.sh train --cwd '~/code/rul_sbi' 'bash run_repro.sh'
#
# The model is tiny (49k params), so the bottleneck is CUDA LAUNCH LATENCY on a machine
# loaded at ~80/80, not compute. Hence MANY parallel processes on the same GPU are
# faster than a single process.
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-6}                       # number of parallel processes
EP=${EPOCHS:+--epochs $EPOCHS}    # override epoch count (test runs only)
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT" "$OUT/logs"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== available GPUs: ${GPUS[*]} · $NW workers · seeds=$SEEDS · out=$OUT =="

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
echo "== main grid done (fail=$fail) =="
grep -h '^RESULT' "$OUT"/logs/shard*.log | sort

echo "== operating-condition normalization variant (FD002/FD004) =="
for ((i=0; i<2; i++)); do
    g=${GPUS[$((i % ${#GPUS[@]}))]}
    CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 \
    $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
        --datasets FD002,FD004 --ablations yes_yes_yes --cond-norm $EP \
        --shard "$i/2" > "$OUT/logs/cn$i.log" 2>&1 &
done
wait

echo "== tables + figures =="
$PY report.py  --outdir "$OUT" --md "$OUT/REPRODUCTION.md"
$PY figures.py --outdir "$OUT"
echo "== done =="
