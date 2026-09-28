#!/usr/bin/env bash
# Tai bo C-MAPSS (12 MB) ve data/CMAPSSData. Nguon: NASA PCoE mirror tren S3.
set -euo pipefail
DST=${1:-"$(cd "$(dirname "$0")/.." && pwd)/data/CMAPSSData"}
URL='https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip'
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$DST"
curl -fL --retry 3 -o "$TMP/outer.zip" "$URL"
unzip -qo "$TMP/outer.zip" -d "$TMP"
unzip -qo "$TMP/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" -d "$DST"
echo "OK -> $DST"
