import os.path
import pandas as pd

file_name = 'model_0117.txt'
with open(file_name, 'r') as f:
    res = f.read().splitlines()


def get_data(json_data):
    data1 = [json_data['title'], json_data['prompt']]

    data_tags = []

    data_tags.extend(json_data['conceptual_keywords'])

    data_tags.extend(json_data['descriptive_keywords'][:5])

    data1.append(','.join(data_tags))

    return data1


import json

data = []
for i in res[:1000]:
    try:
        imgid, json_str = i.split('\t')[:2]
        json_str = json.loads(json_str)
        data1 = get_data(json_str)
        image_path = f'/data/aigc/download/images/{imgid}.jpg'
        dst_path = f'/data/aigc/HunyuanDiT-main/datasets/vcg_images_test/'
        if os.path.exists(image_path) is False:
            continue
        for ind,j in enumerate(data1,start=1):
            dst_local_path = f"{dst_path}/{imgid}_{ind}.jpg"
            os.symlink(image_path, dst_local_path)
            data.append(
                {"image_path": dst_local_path,
                 "text": j}
            )
    except:
        pass

# 保存 CSV
df = pd.DataFrame(data)
# 使用制表符或逗号均可，pandas默认逗号，Hunyuan脚本默认读取csv
df.to_csv("train_dataset_test.csv", index=False)
