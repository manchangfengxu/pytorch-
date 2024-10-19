#多个GPU,数据集并行处理

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#创建一个虚拟数据集
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    # 获取数据
    def __getitem__(self, index):
        return self.data[index]
    # 获取长度
    def __len__(self):
        return self.len
    
# 用于批量加载数据(传入的对象的结构是固定的)
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), 
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())
        
        return output
    
#-----------------------核心：创建一个模型和数据并行-------------------

# 创建模型
model = Model(input_size, output_size)
# 检查是否有多个GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    # 将模型包装，将输入分成多个小批次，在多个GPU上并行处理，然后将结果合并
    model = nn.DataParallel(model)

model.to(device)

# 运行模型

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
    
# DataParallel会自动的划分数据，并将作业发送到多个GPU上的多个模型。 
# 并在每个模型完成作业后，收集合并结果并返回。

