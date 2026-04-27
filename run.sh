#!/usr/bin/env bash
set -e

DEVICE=${DEVICE:-1}
DATASET=${DATASET:-ottqa}
EMBED=${EMBED:-uae}
LM=${LM:-qwen7}

export CUDA_VISIBLE_DEVICES="$DEVICE"

python align_info.py -d "$DATASET" -embed "$EMBED" -lm "$LM"
mkdir -p "./keyword_objects/${DATASET}/${EMBED}_${LM}"
cp "./results/${DATASET}/${EMBED}_${LM}/base.json" "./keyword_objects/${DATASET}/${EMBED}_${LM}/base.json"

python align_structure_expand.py -d "$DATASET" -embed "$EMBED" -lm "$LM"

python align_structure_filter.py -d "$DATASET" -embed "$EMBED" -lm "$LM"

python verify.py -d "$DATASET" -embed "$EMBED" -lm "$LM"

for k in 3 4 5; do
  python verify.py -d "$DATASET" -embed "$EMBED" -lm "$LM" -k "$k"
done