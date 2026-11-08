#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODEL_NAME_OR_PATH:?Set the existing foundation-model directory}"
: "${MODELS:?Set 01 or 02 training output models.json}"
: "${QUERY_MANIFEST:?Set query_manifest.json}"
: "${VERIFIER:?Set the frozen verifier directory}"
: "${POISON_OUTPUT:?Set the exact P.npz used to fit C}"
: "${OUTPUT_DIR:?Set a new evaluation output directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"${PYTHON:-python}" -m review_exp.common.evaluate_checkpoints \
  --models "$MODELS" --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --dataset_name "${DATASET_NAME:-./dataset/cifar10.py}" \
  --query-manifest "$QUERY_MANIFEST" --verifier "$VERIFIER" --poison "$POISON_OUTPUT" \
  --output-dir "$OUTPUT_DIR" --num-processes "${NUM_PROCESSES:-1}" \
  --mixed-precision "${MIXED_PRECISION:-bf16}" --port "${PORT:-29620}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-128}" "$@"