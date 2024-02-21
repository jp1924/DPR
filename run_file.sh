#!/bin/bash
NUM_GPU=4
GPU_IDS="0,1,2,3"

export WANDB_DISABLE_CODE="false"
export WANDB_DISABLED="true"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export OMP_NUM_THREADS=10

# torchrun --nproc_per_node $NUM_GPU \
deepspeed --num_gpus $NUM_GPU \
    "/root/main.py" \
    --output_dir=/root/dpr_output_dir \
    --run_name=dpr_test \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=1 \
    --eval_accumulation_steps=1 \
    --evaluation_strategy=epoch \
    --num_train_epochs=54 \
    --max_length=2048 \
    --save_strategy=epoch \
    --logging_strategy=steps \
    --logging_steps=1 \
    --lr_scheduler_type=cosine \
    --learning_rate=1e-5 \
    --warmup_ratio=0.1 \
    --optim=adamw_torch \
    --report_to=none \
    --dataloader_num_workers=2 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --remove_unused_columns=false \
    --fp16=true \
    --group_by_length=false \
    --deepspeed="/root/zero_config.json"