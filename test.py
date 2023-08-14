import os

from datasets import Dataset

path = "/mnt/petrelfs/chenkeyu1/datasets/zhouyunhua/qwen/"
file_list = []
file_name = []
# 遍历目录中的所有文件
for filename in os.listdir(path):
    if filename.startswith('long_text'):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            file_list.append(content)
            file_name.append(file_name)
data_dict = {
    'text': file_list
}
# import pdb;pdb.set_trace()
dataset = Dataset.from_dict(data_dict)