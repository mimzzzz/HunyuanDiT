import torch
import os
import argparse


def convert(src_folder, output_path):
    print(f"正在加载 DeepSpeed 权重: {src_folder}")

    # 1. 定位 model_states 文件
    state_file = os.path.join(src_folder, "mp_rank_00_model_states.pt")

    if not os.path.exists(state_file):
        raise FileNotFoundError(f"找不到权重文件: {state_file}")

    # 2. 加载权重 (修复点：添加 weights_only=False)
    # DeepSpeed 的 checkpoint 包含 args 配置信息，必须允许非纯权重加载
    try:
        payload = torch.load(state_file, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch (没有 weights_only 参数的情况)
        payload = torch.load(state_file, map_location="cpu")

    # 3. 提取 state_dict
    if "module" in payload:
        state_dict = payload["module"]
    else:
        state_dict = payload

    # 4. 清理 key 名称
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_key = k[7:]  # 去掉 module.
        else:
            new_key = k
        new_state_dict[new_key] = v

    print(f"权重提取完成，Key 数量: {len(new_state_dict)}")

    # 5. 保存
    print(f"正在保存到: {output_path} ...")
    torch.save(new_state_dict, output_path)
    print("✅ 转换成功！")


if __name__ == "__main__":
    # 请确认这里的路径是你报错的那个目录
    src = "/data/aigc/HunyuanDiT-main/log_EXP/001-l20_2m_finetune/checkpoints/0000020.pt"
    dst = "/data/aigc/HunyuanDiT-main/hunyuan_dit_v1.2_finetuned_e2.pt"

    convert(src, dst)