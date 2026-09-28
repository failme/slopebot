#!/usr/bin/env bash
# Download the offline Slope build (Unity WebGL) into ./game
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="https://raw.githubusercontent.com/slopenexus/slopenexus.github.io/test/games/19-04v2.1"
mkdir -p game/style
for f in index.html UnityLoader.js UnityProgress.js slope_new_19_04_v2.json \
         slope_new_19_04_v2.data.unityweb slope_new_19_04_v2.wasmcode.unityweb \
         slope_new_19_04_v2.wasmframework.unityweb; do
  echo "fetching $f"
  curl -fsSL "$BASE/$f" -o "game/$f"
done
for f in style.css fullscreen.png progressEmpty.Dark.png progressFull.Dark.png progressLogo.Dark.png; do
  curl -fsSL "$BASE/style/$f" -o "game/style/$f" || true
done
echo "done: game/ is ready"
