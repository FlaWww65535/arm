#!/usr/bin/env bash
set -e

DATASET=${DATASET:-ottqa}
EMBED=${EMBED:-uae}
LM=${LM:-qwen7}
GPU_NUM=${GPU_NUM:-2}
CPU_NUM=${CPU_NUM:-8}

if [[ "$DATASET" == "bird" ]]; then
  VERIFY_KS=${VERIFY_KS:-"2 3 4"}
elif [[ "$DATASET" == "ottqa" ]]; then
  VERIFY_KS=${VERIFY_KS:-"3 4 5"}
elif [[ "$DATASET" == "wikihop" ]]; then
  VERIFY_KS=${VERIFY_KS:-"1 2 3"}
else
  echo "Unsupported DATASET: $DATASET" >&2
  exit 1
fi

python align_info.py -d "$DATASET" -embed "$EMBED" -lm "$LM" -gpu_num "$GPU_NUM"
mkdir -p "./keyword_objects/${DATASET}/${EMBED}_${LM}"
cp "./results/${DATASET}/${EMBED}_${LM}/base.json" "./keyword_objects/${DATASET}/${EMBED}_${LM}/base.json"

python align_structure_expand.py -d "$DATASET" -embed "$EMBED" -lm "$LM" -parallel_num "$CPU_NUM"

python align_structure_filter.py -d "$DATASET" -embed "$EMBED" -lm "$LM" -parallel_num "$CPU_NUM"

for k in $VERIFY_KS; do
  python verify.py -d "$DATASET" -embed "$EMBED" -lm "$LM" -k "$k" -gpu_num "$GPU_NUM"
done

python aggregate.py -d "$DATASET" -embed "$EMBED" -lm "$LM"