<<<<<<< HEAD
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
=======
import torch
import torch.nn as nn

# 定义一个简单的线性模型
model = nn.Linear(10, 5)

# 定义一个CrossEntropyLoss对象
criterion = nn.CrossEntropyLoss()
a = torch.load("/mnt/petrelfs/chenkeyu1/datasets/a.pt",map_location="cuda:0")
b = torch.load("/mnt/petrelfs/chenkeyu1/datasets/b.pt",map_location="cuda:0")

c = torch.load("/mnt/petrelfs/chenkeyu1/datasets/c.pt",map_location="cpu")
d = torch.load("/mnt/petrelfs/chenkeyu1/datasets/d.pt",map_location="cpu")

print(criterion(c,d))
>>>>>>> cky/0.6.1
