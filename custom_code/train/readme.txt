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

PYTHONPATH=. nohup bash hydit/train.sh &