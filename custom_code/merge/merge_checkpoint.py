import torch
import safetensors.torch
import os
from transformers import T5Tokenizer, T5EncoderModel

# ================= 1. 配置路径 (自动适配你的环境) =================
# 你的微调模型 (DiT)
input_diffusion = "./hunyuan_dit_v1.2_finetuned_e2.pt"

# 官方底模路径根目录
base_root = "/data/aigc/HunyuanDiT-main/ckpts/t2i"

# CLIP (BERT) 路径
input_bert = os.path.join(base_root, "clip_text_encoder/pytorch_model.bin")

# VAE 路径 (注意：通常是 diffusion_pytorch_model.safetensors)
# 如果你的 VAE 文件夹里是 .bin，请看代码下方的自动判断逻辑
input_vae_dir = os.path.join(base_root, "sdxl-vae-fp16-fix")
input_vae = os.path.join(input_vae_dir, "diffusion_pytorch_model.safetensors")

# T5 路径 (直接加载本地，不用下载)
input_mt5_dir = os.path.join(base_root, "mt5")

# 输出文件名
output = "HunyuanDiT_v1.2_Finetuned_Full.safetensors"

print(f"🚀 开始合并模型...")
print(f"   DiT: {input_diffusion}")
print(f"   CLIP: {input_bert}")
print(f"   VAE: {input_vae}")
print(f"   T5: {input_mt5_dir}")

# ================= 2. 加载并处理组件 =================
out_sd = {}

# --- A. 处理 CLIP (BERT) ---
print("⏳ [1/4] Loading CLIP (Bert)...")
bert_sd = torch.load(input_bert, map_location="cpu", weights_only=False)  # 兼容性修改
for k in bert_sd:
    if not k.startswith("visual."):
        out_sd["text_encoders.hydit_clip.transformer.{}".format(k)] = bert_sd[k].half()
del bert_sd

# --- B. 处理 T5 (mT5) ---
print("⏳ [2/4] Loading T5 (mT5-XL)... 这是个大块头，请耐心等待")
try:
    # 优先加载本地
    mt5 = T5EncoderModel.from_pretrained(input_mt5_dir, local_files_only=True)
    tokenizer = T5Tokenizer.from_pretrained(input_mt5_dir, local_files_only=True)
except Exception as e:
    print(f"⚠️ 本地加载 T5 失败: {e}")
    print("尝试从 HuggingFace 在线加载 (google/mt5-xl)...")
    mt5 = T5EncoderModel.from_pretrained("google/mt5-xl")
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-xl")

# 处理 T5 权重
t5_sd = mt5.state_dict()
for k in t5_sd:
    out_sd["text_encoders.mt5xl.transformer.{}".format(k)] = t5_sd[k].half()

# 处理 Tokenizer (spiece.model) 并嵌入文件
if hasattr(tokenizer, "sp_model"):
    print("   Embedding T5 spiece.model...")
    sp_model = torch.ByteTensor(list(tokenizer.sp_model.serialized_model_proto()))
    out_sd["text_encoders.mt5xl.spiece_model"] = sp_model
del mt5, t5_sd

# --- C. 处理 DiT (你的微调模型) ---
print("⏳ [3/4] Loading DiT (Finetuned)...")
hydit = torch.load(input_diffusion, map_location="cpu", weights_only=False)
# 自动判断 Key 格式
first_key = next(iter(hydit))
prefix = ""
if first_key.startswith("module."):
    prefix = "module."
    print("   Detected DeepSpeed prefix 'module.', removing it.")

for k, v in hydit.items():
    # 清理 DeepSpeed 前缀
    if prefix and k.startswith(prefix):
        clean_k = k[len(prefix):]
    else:
        clean_k = k

    # 这里的 key 映射非常关键
    out_sd["model.{}".format(clean_k)] = v.half()
del hydit

# --- D. 处理 VAE ---
print("⏳ [4/4] Loading VAE...")
if not os.path.exists(input_vae):
    # 尝试找 .bin
    input_vae_bin = os.path.join(input_vae_dir, "diffusion_pytorch_model.bin")
    if os.path.exists(input_vae_bin):
        print(f"   Found .bin VAE: {input_vae_bin}")
        vae_sd = torch.load(input_vae_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"找不到 VAE 文件，请检查路径: {input_vae_dir}")
else:
    vae_sd = safetensors.torch.load_file(input_vae)

for k in vae_sd:
    out_sd["vae.{}".format(k)] = vae_sd[k].half()
del vae_sd

# ================= 3. 保存 =================
print(f"💾 Saving to {output} ...")
safetensors.torch.save_file(out_sd, output)
print("✅ 合并成功！这就是你要的完整单文件 Checkpoint。")
print("👉 请在 ComfyUI 中使用 'Load Checkpoint' 节点直接加载它。")