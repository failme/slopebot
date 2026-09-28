#!/usr/bin/env bash
# Evaluate the geometry controller on seeds 1..12 using 4 parallel processes.
cd "$(dirname "$0")/.."
out=${1:-/tmp/geo_bench}
mkdir -p "$out"
for r in 1-3 4-6 7-9 10-12; do
  python3 bot/geo_play.py --seeds $r --max-steps ${MAXSTEPS:-8000} > "$out/$r.log" 2>&1 &
done
wait
grep -h "^seed" "$out"/*.log | sort -t' ' -k2 -n
grep -h "^seed" "$out"/*.log | awk '{s+=$4; n++} END {printf "mean %.1f over %d seeds\n", s/n, n}'
