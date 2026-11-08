#!/usr/bin/env bash
# Run ONCE before inspecting any new suspicious-model results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${MODEL_NAME_OR_PATH:?Set the existing local foundation-model directory}"
BASELINE_DIR="${BASELINE_DIR:-review_exp/00_prepare_baseline/results}"
ARTIFACT_DIR="${ARTIFACT_DIR:-review_exp/artifacts/cifar10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_NAME_OR_PATH
export DATASET_NAME="${DATASET_NAME:-./dataset/cifar10.py}"
PYTHON="${PYTHON:-python}"
if [[ -e "$ARTIFACT_DIR/query_manifest.json" || -e "$ARTIFACT_DIR/verifier/verifier.json" ]]; then
  printf 'Use a new ARTIFACT_DIR; query identities and verifier must be frozen once: %s\n' "$ARTIFACT_DIR" >&2
  exit 2
fi
"$PYTHON" -m review_exp.common.collect_outputs prepare \
  --model_name_or_path "$MODEL_NAME_OR_PATH" --dataset_name "$DATASET_NAME" \
  --indices "${QUERY_INDICES:-$BASELINE_DIR/query_indices.json}" --split train \
  --output "$ARTIFACT_DIR/query_manifest.json" "$@"
for role in V B P; do
  case "$role" in
    V) checkpoint="$BASELINE_DIR/victim/checkpoint.pt" ;;
    B) checkpoint="$BASELINE_DIR/benign/checkpoint.pt" ;;
    P) checkpoint="$BASELINE_DIR/poisoned/checkpoint.pt" ;;
  esac
  QUERY_MANIFEST="$ARTIFACT_DIR/query_manifest.json" CHECKPOINT="$checkpoint" \
    OUTPUT="$ARTIFACT_DIR/$role.npz" MODEL_ID="baseline_$role" EXPECTED=reference \
    bash review_exp/export_outputs.sh "$@"
done
"$PYTHON" -m review_exp.common.verifier fit \
  --victim "$ARTIFACT_DIR/V.npz" --benign "$ARTIFACT_DIR/B.npz" --poison "$ARTIFACT_DIR/P.npz" \
  --output "$ARTIFACT_DIR/verifier" --device "${C_DEVICE:-auto}" \
  --norm --scale --no-softmax --epochs "${C_EPOCHS:-300}" --seed "${C_SEED:-42}" \
  --lr "${C_LR:-0.01}" --train-fraction 0.6 --validation-fraction 0.2 \
  --primary "${PRIMARY:-legacy_harmonic}" --alpha "${ALPHA:-0.01}" --tau "${TAU:-0}" \
  --sample-num "${SAMPLE_NUM:-100}" --repeats "${REPEATS:-40}" --query-seed "${QUERY_SEED:-100}"
for role in V B; do
  "$PYTHON" -m review_exp.common.verifier evaluate --verifier "$ARTIFACT_DIR/verifier" \
    --suspect "$ARTIFACT_DIR/$role.npz" --poison "$ARTIFACT_DIR/P.npz" \
    --model-id "baseline_$role" --expected reference --device "${C_DEVICE:-auto}" \
    --output "$ARTIFACT_DIR/controls/$role.result.json"
done
printf 'Frozen query manifest, P, C and control results: %s\n' "$ARTIFACT_DIR"