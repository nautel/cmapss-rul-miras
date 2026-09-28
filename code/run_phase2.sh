#!/usr/bin/env bash
# Giai doan 2 — chay SAU khi luoi chinh (run_repro.sh) xong.
# Muc dich: tach roi hai nguyen nhan co the lam lech so voi paper
#   (a) dung som qua ngat  -> --full-epochs (chay du 200/600 epoch cua Table 2)
#   (b) Z-score toan cuc   -> --cond-norm  (chuan hoa theo 6 che do van hanh)
set -u
PY=${PY:-/home/lab/letuan/envs/rul/bin/python}
ROOT=${ROOT:-/home/lab/letuan/data/cmapss}
OUT=${OUT:-/home/lab/letuan/runs/rul_sbi}
SEEDS=${SEEDS:-0,1,2}
NW=${NW:-6}
cd "$(dirname "$0")" || exit 1
mkdir -p "$OUT/logs"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
echo "== GPU: ${GPUS[*]} · $NW worker · seeds=$SEEDS =="

launch() {   # $1=nhan  $2..=tham so them
    local name=$1; shift
    local pids=()
    for ((i=0; i<NW; i++)); do
        local g=${GPUS[$((i % ${#GPUS[@]}))]}
        CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
        $PY run_all.py --root "$ROOT" --outdir "$OUT" --seeds "$SEEDS" \
            --shard "$i/$NW" "$@" > "$OUT/logs/${name}$i.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    echo "-- $name xong --"
    grep -h '^RESULT' "$OUT"/logs/${name}*.log | sort
}

# Hai nhom chay SONG SONG (2 x NW tien trinh). Job dai nhat (FD004, 600 epoch) ~65 phut la
# san cua wall-clock; chay noi tiep thi mat gap doi ma khong loi gi.
# (a) du epoch, Z-score toan cuc — ca 4 subset, chi mo hinh day du
launch fe --ablations yes_yes_yes --full-epochs &
A=$!
# (b) du epoch + chuan hoa theo che do van hanh — chi FD002/FD004 (FD001/FD003 chi 1 che do)
launch fecn --datasets FD002,FD004 --ablations yes_yes_yes --full-epochs --cond-norm &
B=$!
wait $A; wait $B

echo "== bang + hinh (cap nhat lai) =="
$PY report.py  --outdir "$OUT" --md "$OUT/REPRODUCTION.md"
$PY figures.py --outdir "$OUT"
echo "== xong giai doan 2 =="
