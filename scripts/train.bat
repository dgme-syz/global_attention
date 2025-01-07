@echo off

set MODEL_ID="E:\pretrained_models\Qwen\Qwen2___5-0___5B-Instruct"
set DATASET_ID="gsm8k"
set OUTPUT_DIR="./saves/sft"
set TEMPLATE="qwen"
set FINETUNING="full"

llamafactory-cli train ^
    --stage sft ^
    --do_train ^
    --model_name_or_path %MODEL_ID% ^
    --dataset %DATASET_ID% ^
    --dataset_dir "LLaMA-Factory/data" ^
    --use_global_attn 1 ^
    --template %TEMPLATE% ^
    --finetuning_type %FINETUNING% ^
    --output_dir %OUTPUT_DIR% ^
    --overwrite_cache ^
    --overwrite_output_dir ^
    --cutoff_len 768 ^
    --per_device_train_batch_size 2 ^
    --per_device_eval_batch_size 2 ^
    --gradient_accumulation_steps 8 ^
    --lr_scheduler_type cosine ^
    --logging_steps 5 ^
    --warmup_steps 20 ^
    --save_steps 200 ^
    --eval_steps 100 ^
    --evaluation_strategy steps ^
    --load_best_model_at_end ^
    --learning_rate 5e-5 ^
    --num_train_epochs 5.0 ^
    --val_size 0.1 ^
    --plot_loss ^
    --save_total_limit 10 ^ 