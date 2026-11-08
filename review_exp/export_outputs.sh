#!/usr/bin/env bash
# Usage: MODEL_NAME_OR_PATH=/server/foundation/clip-vit-base-patch32 \
#   QUERY_MANIFEST=... CHECKPOINT=... OUTPUT=... MODEL_ID=... ./review_exp/export_outputs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODEL_NAME_OR_PATH:?Set the existing server foundation-model directory}"
: "${QUERY_MANIFEST:?Set the frozen query_manifest.json path}"
: "${CHECKPOINT:?Set a COMPLETE model checkpoint path}"
: "${OUTPUT:?Set the output .npz path}"
: "${MODEL_ID:?Set a unique model ID, including training seed}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
launch=(--num_processes "$NUM_PROCESSES" --num_machines 1 --machine_rank 0
        --mixed_precision "${MIXED_PRECISION:-bf16}" --main_process_port "${PORT:-29610}")
if (( NUM_PROCESSES > 1 )); then launch+=(--multi_gpu); fi
accelerate launch "${launch[@]}" -m review_exp.common.collect_outputs collect \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_name "${DATASET_NAME:-./dataset/cifar10.py}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --checkpoint "$CHECKPOINT" --query-manifest "$QUERY_MANIFEST" \
  --model-id "$MODEL_ID" --expected "${EXPECTED:-reference}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-128}" --output "$OUTPUT" "$@"