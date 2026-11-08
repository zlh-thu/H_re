#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${MANIFEST:?Set MANIFEST to the completed cross-dataset JSON manifest}"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/review_exp/03_cross_dataset/results}"
"$PYTHON" review_exp/03_cross_dataset/batch_evaluate.py \
  --manifest "$MANIFEST" --output-dir "$OUTPUT_DIR" \
  --device "${DEVICE:-cuda:0}" "$@"