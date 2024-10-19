import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入通道为1，输入一个二维图像
        # 通过6个不同的5*5的卷积核，得到6个不同的特征图
        # 输出通道为6，输出6个二维特征图
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 通过16个不同的5*5的卷积核，每种卷积核对6个特征图进行卷积并相加
        # 得到16个不同的特征图
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 对输入 x 应用第一个卷积层，然后使用 ReLU 激活函数。
        # 并应用2 * 2的最大池化（取2 * 2中最大值） 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 将输出展平为一维，准备进入全连接层。-1 表示自动计算批次大小。(展成一层需要的空间)
        # self.num_flat_features(x) 计算的是 x 中除了批量维度以外的所有维度的乘积。
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 去除批量维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
input = torch.randn(1, 1, 32, 32)
# 目标
target = torch.randn(10)
target = target.view(1, -1)
# 损失函数
criterion = nn.MSELoss()    

# 使用随机梯度下降（SGD）优化器，学习率为0.01
# net.parameters() 返回模型的所有可学习参数（权重和偏置）
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)

loss.backward()
optimizer.step()


