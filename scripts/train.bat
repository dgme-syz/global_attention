@echo off

set MODEL_ID="E:\pretrained_models\AI-ModelScope\TinyLlama-1___1B-Chat-v1___0"
set DATASET_ID="mathinstruct"
set OUTPUT_DIR="./saves/llama"
set TEMPLATE="tinyllama"
set FINETUNING="ga"

llamafactory-cli train ^
    --stage sft ^
    --do_train ^
    --label_names "labels" ^
    --model_name_or_path %MODEL_ID% ^
    --dataset %DATASET_ID% ^
    --dataset_dir "LLaMA-Factory/data" ^
    --template %TEMPLATE% ^
    --finetuning_type %FINETUNING% ^
    --output_dir %OUTPUT_DIR% ^
    --overwrite_cache ^
    --overwrite_output_dir ^
    --cutoff_len 1024 ^
    --per_device_train_batch_size 2 ^
    --per_device_eval_batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --lr_scheduler_type cosine ^
    --logging_steps 5 ^
    --warmup_steps 50 ^
    --save_steps 200 ^
    --eval_steps 100 ^
    --evaluation_strategy steps ^
    --learning_rate 5e-5 ^
    --num_train_epochs 5.0 ^
    --val_size 0.01 ^
    --plot_loss ^
    --max_samples 10000 ^ 
    --save_total_limit 10 ^ 
    --pure_bf16 1 ^