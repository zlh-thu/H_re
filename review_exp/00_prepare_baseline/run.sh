#!/usr/bin/env bash
# ⚠️ MODEL_NAME_OR_PATH=/server/model/clip-vit-base-patch32 ./review_exp/00_prepare_baseline/run.sh all
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
: "${MODEL_NAME_OR_PATH:?Set the existing local foundation-model directory on the server}"
STAGE="${1:-all}"
if (( $# > 0 )); then shift; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
BASELINE_DIR="${BASELINE_DIR:-review_exp/00_prepare_baseline/results}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
launch=(--num_processes "$NUM_PROCESSES" --num_machines 1 --machine_rank 0
        --mixed_precision "$MIXED_PRECISION" --main_process_port "${PORT:-29600}")
if (( NUM_PROCESSES > 1 )); then launch+=(--multi_gpu); fi
common=(--model_name_or_path "$MODEL_NAME_OR_PATH"
        --dataset_name "${DATASET_NAME:-./dataset/cifar10.py}"
        --baseline-dir "$BASELINE_DIR" --mixed_precision "$MIXED_PRECISION"
        --split_seed "${SPLIT_SEED:-42}" --train_val_split "${TRAIN_VAL_SPLIT:-0.1}"
        --num_workers "${NUM_WORKERS:-0}" --seed "${SEED:-42}"
        --learning_rate "${LR:-1e-5}" --lr_scheduler_type cosine
        --num_warmup_steps "${WARMUP_STEPS:-1000}"
        --classifier_lr_multiplier "${CLASSIFIER_LR_MULTIPLIER:-10}"
        --weight_decay "${WEIGHT_DECAY:-0}" --optimizer adamw
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
        --train_module all
        --poison-fraction "${POISON_FRACTION:-0.1}" --drop-fraction "${DROP_FRACTION:-0.5}"
        --target-label "${TARGET_LABEL:-0}" --trigger-size "${TRIGGER_SIZE:-4}"
        --trigger-location "${TRIGGER_LOCATION:-18}")
if [[ -n "${TRAIN_DIR:-}" ]]; then common+=(--train_dir "$TRAIN_DIR"); fi
if [[ -n "${VALIDATION_DIR:-}" ]]; then common+=(--validation_dir "$VALIDATION_DIR"); fi
if [[ "${USE_PT_IMAGEFOLDER:-0}" == "1" ]]; then common+=(--use_pt_imagefolder); fi
if [[ -n "${DS_INDICES:-}" ]]; then common+=(--ds-indices "$DS_INDICES"); fi
run_stage() {
    local current="$1"
    shift
    local -a stage_args
    stage_args=(--stage "$current" --output_dir "$BASELINE_DIR/$current")
    case "$current" in
        victim)
            stage_args+=(--num_train_epochs "${V_EPOCHS:-20}" --seed "${V_SEED:-${SEED:-42}}"
                        --per_device_train_batch_size "${V_BATCH_SIZE:-256}"
                        --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-256}") ;;
        rank)
            stage_args+=(--checkpoint "$BASELINE_DIR/victim/checkpoint.pt"
                        --rank-batch-size "${RANK_BATCH_SIZE:-128}") ;;
        prepare) ;;
        poisoned)
            stage_args+=(--checkpoint "$BASELINE_DIR/victim/checkpoint.pt"
                        --num_train_epochs "${P_EPOCHS:-5}" --seed "${P_SEED:-${SEED:-42}}"
                        --per_device_train_batch_size "${P_BATCH_SIZE:-256}"
                        --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-256}") ;;
        benign)
            stage_args+=(--num_train_epochs "${B_EPOCHS:-20}" --seed "${B_SEED:-${SEED:-42}}"
                        --per_device_train_batch_size "${B_BATCH_SIZE:-16}"
                        --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-256}") ;;
        *) printf 'Unknown stage: %s\n' "$current" >&2; exit 2 ;;
    esac
    accelerate launch "${launch[@]}" -m review_exp.00_prepare_baseline.prepare_baseline \
        "${common[@]}" "${stage_args[@]}" "$@"
}
