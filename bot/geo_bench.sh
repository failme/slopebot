#!/usr/bin/env bash
# Evaluate the geometry controller on a range of seeds with 4 parallel processes.
#   ./bot/geo_bench.sh OUTDIR [FIRST LAST]      (default seeds 1..12)
cd "$(dirname "$0")/.."
out=${1:-/tmp/geo_bench}
first=${2:-1}; last=${3:-12}
mkdir -p "$out"
for w in 0 1 2 3; do
  (for s in $(seq $((first + w)) 4 "$last"); do
     python3 bot/geo_play.py --seed "$s" --max-steps "${MAXSTEPS:-8000}"
   done) > "$out/w$w.log" 2>&1 &
done
wait
grep -h "^seed" "$out"/w*.log | sort -t' ' -k2 -n
grep -h "^seed" "$out"/w*.log | awk '{s+=$4; n++; if ($4 >= 300) w++} END {printf "mean %.1f over %d seeds, %d reached 300\n", s/n, n, w}'
