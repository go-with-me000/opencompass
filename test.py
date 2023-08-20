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