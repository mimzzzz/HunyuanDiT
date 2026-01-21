#!/bin/bash

# ================= 配置区 (请修改为你自己的路径) =================
# 任务名称
task_flag="l20_2m_finetune"

# ⚠️ 关键：这里要填第一步生成的 JSON 文件路径，而不是 CSV！
index_file="/data/aigc/HunyuanDiT-main/datasets/arrow_index1/train_dataset.json"

# 模型权重路径
resume_module_root="/data/aigc/HunyuanDiT-main/ckpts/t2i/model/pytorch_model_distill.pt"
resume_ema_root="/data/aigc/HunyuanDiT-main/ckpts/t2i/model/pytorch_model_ema.pt"
results_dir="./log_EXP"

# ================= 训练参数 (适配 L20 48G) =================
# L20 显存较大，Batch Size 可以尝试 4 (如果 OOM 就改 2)
batch_size=4
image_size=1024

# 梯度累积：8卡 x Batch4 x Accu2 = Global Batch 64
grad_accu_steps=2

# 学习率：全量微调建议 1e-4
lr=0.0001

# 训练轮数：200万数据建议跑 1-2 个 Epoch
epochs=5

# 保存频率：每 5000 步存一次
ckpt_every=5000
ckpt_latest_every=5000

# ================= 启动命令 =================
# 注意：这里调用的是 run_g.sh，它内部会调用 deepspeed
sh $(dirname "$0")/run_g.sh \
    --task-flag ${task_flag} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0.1 \
    --uncond-p-t5 0.1 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 42 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps 2000 \
    --use-flash-attn \
    --use-zero-stage 2 \
#    --gradient-checkpointing \
    --results-dir ${results_dir} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --resume-ema-root ${resume_ema_root} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --rope-img base1024 \
    --no-fp16 \
    --use-ema \
    "$@"

# 解释：
# --rope-img base1024 : 适配 1024x1024 分辨率
# --no-fp16 : 强制关闭 FP16，配合修改后的 ds_config.py 启用 BF16
