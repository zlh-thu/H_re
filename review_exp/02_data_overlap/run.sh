#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${MODEL_NAME_OR_PATH:-${MODEL_PATH:-/path_to_model/clip-vit-base-patch32}}"
DATASET="${DATASET_NAME:-${DATASET:-./dataset/cifar10.py}}"
OUTPUT_DIR="${OUTPUT_DIR:-review_exp/results/02_data_overlap}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
PORT="${PORT:-28102}"
BS="${BS:-16}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
launch=(accelerate launch --mixed_precision "$MIXED_PRECISION" --num_processes "$NUM_PROCESSES" --num_machines 1 --machine_rank 0 --main_process_port "$PORT")
if (( NUM_PROCESSES > 1 )); then launch+=(--multi_gpu); fi
"${launch[@]}" review_exp/02_data_overlap/run_experiment.py \
    --model_name_or_path "$MODEL_PATH" --dataset_name "$DATASET" --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$BS" --per_device_eval_batch_size "$BS" \
    --learning_rate 1e-5 --num_train_epochs 20 --classifier_lr_multiplier 10 \
    --lr_scheduler_type cosine --num_warmup_steps 1000 --mixed_precision "$MIXED_PRECISION" \
    --split_seed 42 --data_seed 2026 --overlap_rates 1.0 --seeds 43 44 45 "$@"