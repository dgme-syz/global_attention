#!/bin/bash

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
DATASET_ID="mathinstruct"
OUTPUT_DIR="./saves/sft"
TEMPLATE="qwen"
FINETUNING="full"

llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_ID" \
    --dataset "$DATASET_ID" \
    --dataset_dir "LLaMA-Factory/data" \
    --use_global_attn 1 \
    --template "$TEMPLATE" \
    --finetuning_type "$FINETUNING" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 5000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --val_size 0.01 \
    --plot_loss \
    --max_samples 10000 \
    --save_total_limit 2 \
    --bf16
