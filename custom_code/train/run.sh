#!/bin/bash

# ================= 环境变量 =================
export OMP_NUM_THREADS=4  # 增加 CPU 线程数处理数据加载
export NCCL_P2P_DISABLE=0
# L20 支持 P2P 通信，通常不需要禁用。如果遇到多卡卡死不动，再取消下面这行的注释
# export NCCL_P2P_DISABLE=1

# ================= 路径配置 =================
# 1. 指向刚才新建的 DeepSpeed 配置文件 (假设在当前目录)
DS_CONFIG_PATH="hydit/constants/deepspeed_config_z2.json"

# 2. 模型和数据路径 (请根据实际情况确认)
MODEL_ROOT="/data/aigc/HunyuanDiT-main/models"
CSV_PATH="/data/aigc/HunyuanDiT-main/datasets/train_dataset_test.csv"
OUTPUT_DIR="output"

# ================= 启动命令 =================
# 注意：mixed_precision 必须设为 bf16 以配合 L20 和上面的 json 配置
accelerate launch \
    --num_machines 1 \
    --num_processes 8 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file $DS_CONFIG_PATH \
    hydit/train_multi_gpu.py \
    --model "Hunyuan-DiT-v1.2-Diff" \
    --resume-from-checkpoint $MODEL_ROOT \
    --task-flag "l20_2m_finetune" \
    --results-dir $OUTPUT_DIR \
    --csv-path $CSV_PATH \
    --image-size 1024 \
    --batch-size 4 \
    --gradient-accumulation-steps 2 \
    --lr 1e-4 \
    --lr-scheduler "cosine" \
    --lr-warmup-steps 2000 \
    --use-flash-attn \
    --gradient-checkpointing \
    --num-train-steps 65000 \
    --save-every 5000 \
    --log-every 50 \
    --dataloader-num-workers 8