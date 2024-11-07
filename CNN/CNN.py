import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_epochs = 5; #训练轮数
num_classes = 10; #分类数10(数据集有10个类别)
batch_size = 100; #批处理大小
learning_rate = 0.001; #学习率

# MNIST数据集下载
train_dataset = torchvision.datasets.MNIST(root='../../data/', 
                                           train=True, 
                                           transform=transforms.ToTensor(), 
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True) #shuffle=True表示每次迭代时，对数据进行重新排序

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False) #!/shuffle=False表示每次迭代时，不对数据进行重新排序

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        # 第一个卷积层
        self.layer1 = nn.Sequential(
            # 输入通道数为1，输出通道数为16，卷积核大小为5x5，步长为1，填充为2(保证输入输出大小相同)
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 输出通道数为16，卷积核大小为2x2，步长为2，填充为0
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第二个卷积层
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 全连接层
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # 将数据展平，-1表示自动计算维度
        out = self.fc(out)
        return out
    
model = ConvNet(num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss() # nn.CrossEntropyLoss()为LogSoftmax和NLLLoss的组合
# Adam优化器. model.parameters()为可学习参数,包括卷积核和偏置
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
total_step = len(train_loader)
for epoch in range(num_epochs): # 遍历所有轮数
    for i, (images, labels) in enumerate(train_loader): # 遍历所有批次
        images = images.to(device) # 将图像数据转移到GPU上
        labels = labels.to(device) # 将标签数据转移到GPU上

        outputs = model(images)
        loss = criterion(outputs, labels) # 计算损失

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # 更新权重

        if(i+1) % 100 == 0: # 每100个批次输出一次损失   
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
model.eval() # 将模型设置为评估模式，这会关闭dropout和batch normalization等操作，以获得更准确的结果
with torch.no_grad(): # 关闭梯度计算，以节省内存和加速计算
    correct = 0
    total = 0
    for images, labels in test_loader: # 遍历所有测试集批次
        images = images.to(device) # 将图像数据转移到GPU上
        labels = labels.to(device) # 将标签数据转移到GPU上
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # 返回每行最大值的索引，即预测的类别
        total += labels.size(0) # 更新总样本数
        correct += (predicted == labels).sum().item() # 更新正确预测的样本数
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), '../models/CNN.ckpt') # 保存模型参数到文件