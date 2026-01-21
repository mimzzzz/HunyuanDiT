build train dataset
python hydit/data_loader/csv2arrow.py datasets/train_dataset.csv datasets/arrow_index1 8

build train index
python hydit/index_kits/index_v2_builder.py datasets/arrow_index1/train_dataset.json datasets/arrow_index1
or
python index.poy

官方model
https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2/tree/main/t2i
和
https://huggingface.co/Tencent-Hunyuan/Distillation-v1.2/tree/main

PYTHONPATH=. bash hydit/train.sh

第一部分：核心“火候”控制 (训练动力学)
这部分参数直接决定模型能不能学会，学得快还是慢。

1. batch_size (批大小) & grad_accu_steps (梯度累积)
含义：

batch_size：单张显卡一次性“吞”进去多少张图。

grad_accu_steps：显卡由于显存不够，不能一次吞太多，就分几次吞，吞完几次后再算一次总账（更新权重）。

Global Batch Size (全局批大小) = 卡数 × batch_size × grad_accu_steps。这是真正影响模型收敛的核心数值。

如何调整：

在你的配置中，两者最终的 Global Batch 都是 64 (8卡×1×8 或 8卡×8×1)。

黄金法则：微调任务通常 Global Batch 在 32~128 之间比较合适。太小（如 <16）梯度震荡大，难以收敛；太大（如 >256）收敛变慢，且对显存要求极高。

影响：

调大 batch_size：速度变快（并行度高），但显存占用暴增。

调大 grad_accu_steps：显存占用不变，但能模拟出更大的 Global Batch，代价是训练步数变慢（因为要多跑几轮才更新一次）。

2. lr (Learning Rate / 学习率)
含义：模型每次改错的幅度。设得太大是“步子大了容易扯着蛋”，设得太小是“磨洋工”。

如何调整：

你目前是 1e-4 (0.0001)。这是全量微调（Full Finetune）的标准起点。

如果是 LoRA 微调，通常可以更大一点（如 1e-4 ~ 5e-4）。

如果是微调非常细微的风格（如画风模仿），可以降低到 1e-5。

影响：

过大：Loss 不降反升，或者在一个数值附近剧烈震荡（爆 Loss）。

过小：Loss 降得极慢，训练在这个 Epoch 结束了还没学会。

3. epochs (训练轮数)
含义：模型把所有素材从头到尾看几遍。

如何调整：

你现在设的是 2。对于 200万 (2M) 这样的大数据集，1-2 个 Epoch 是非常明智的选择。

数据量少（如几百张图）：可能需要 10~100 个 Epoch。

数据量大（如百万级）：通常 1 个 Epoch 就足够模型“见过世面”了，跑太多会过拟合（Overfit），即模型只会背图，不会创新。

4. warmup-num-steps (热身步数)
含义：刚开始训练时，学习率从 0 慢慢爬升到设定的 lr，给模型一个适应过程，防止一开始梯度太大把模型搞懵了。

如何调整：

通常设为总步数的 1% ~ 5%。你设置的 2000 步对于 200万数据的训练量是合理的。

影响：如果训练一开始 Loss 就炸飞（NaN），尝试增加这个数值。

1. --noise-schedule, --beta-start/end, --predict-type
含义：这些是扩散模型的数学定义，决定了加噪和去噪的过程。

建议：严禁随意修改！ 除非你非常清楚自己在做什么。这些参数必须和预训练模型（底模）保持一致，改了就相当于你要从零开始训练一个新模型，微调会直接崩盘。

现状：你的脚本里用的 scaled_linear, v_prediction 都是 HunyuanDiT 官方底模的默认配置，保持不动即可。

2. --uncond-p & --uncond-p-t5 (CFG Dropout)
含义：训练时有百分之多少的概率“把提示词扔掉”，强迫模型只看图学习。这是为了让模型在推理时能支持 CFG Scale（提示词相关性） 的调节。

如何调整：

默认 0.1 (10%) 是通用标准。

如果你希望模型极其听话（提示词遵循度极高），可以适当降低（如 0.05）。

如果你希望模型生成的图更多样化、更发散，可以适当提高。

3. --use-ema (指数移动平均)
含义：不仅保存当前学到的参数，还额外保存一份“过去一段时间参数的平均值”。

影响：

开启：生成的图画面更平滑、更稳定，不容易出现奇怪的伪影。强烈建议开启，尤其是做产品级模型时。

代价：显存占用翻倍（因为要存两份模型）。如果你显存实在不够，这是第一个可以被砍掉的“奢侈品”。

1. --use-flash-attn
含义：使用 Flash Attention 技术加速注意力计算。

建议：必须开启。它不仅快，还能省显存，而且对效果无损。

2. --gradient-checkpointing (梯度检查点)
含义：不存中间过程，反向传播时重新算。

大神经验：

以时间换空间的终极手段。开启后显存能省一半，但速度慢 30%。

显存 < 80G 几乎必开。除非你的 Batch Size 只有 1 且不开 EMA，否则很难扛住 1024 分辨率的训练。

3. --use-zero-stage 2 (DeepSpeed ZeRO)
含义：多卡训练时，怎么切分模型和优化器。

Stage 2：切分优化器状态和梯度（常用，平衡速度和显存）。

Stage 3：连模型参数都切分（极度省显存，但通信开销大，速度慢）。

建议：如果你开启 Gradient Checkpointing 后还爆显存，就改用 Stage 3。

4. --no-fp16 (精度设置)
含义：你的脚本里用了 --no-fp16，结合我们之前的对话，你是为了强制使用 BF16 (BFloat16)。

建议：绝对正确。对于 HunyuanDiT 这种新架构，FP16 容易溢出（导致训练 Loss 变 NaN），BF16 更加稳定且显存占用一样。