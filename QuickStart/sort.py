#做一个图像分类器

import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

# ---------------1.使用torchvision加载和归一化CIFAR10训练集和测试集--------------
if __name__ == '__main__': #请勿在意
    # 组合预处理链
    transforms = transforms.Compose(
        [transforms.ToTensor(),# 转化为Tensor
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
         #张量归一化
    )
    
    #张量归一化，数据压缩-1~1
    #transforms.Normalize((mean1,mean2..), (std1,std2..))均值和方差
    #（data-mean）/ std
    
    #加载训练集,(root:数据集存放路径，train:是否为训练集，download:是否下载，transform:对数据集进行预处理)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms)
    
    #数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transforms)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    #类别标签
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    #展示图像的函数
    
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        #改变npimg的维度
        #从 (C, H, W)（通道，高度，宽度）转换为 (H, W, C)，以符合 matplotlib 的要求。
        plt.imshow(np.transpose(npimg, (1,2,0)))
    
    #创建迭代器
    dataiter = iter(trainloader)
    images, labels = dataiter.__next__()  # 返回图片及标签
    
    #torchvision.utils.make_grid(images) 将多张图像合并成一张网格图，方便可视化。
    imshow(torchvision.utils.make_grid(images))
    #打印标签
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    
    import torch.nn as nn
    import torch.nn.functional as F
    # -----------------------2.定义卷积神经网络-----------------------------
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 通道数为3，RGB
            self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数3，输出通道数6，卷积核大小5x5
            self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，池化核大小2x2，步长2
            self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道数6，输出通道数16，卷积核大小5x5
            self.fc1 = nn.Linear(16 * 5 * 5,120)    
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
    
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)  # 将张量的形状改变为一维向量，以便输入全连接层
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Net()
    net.to(device)
    
    # -----------------------3.定义损失函数和优化器------------------------
    import torch.optim as optim
    
    #nn.CrossEntropyLoss()为LogSoftmax和NLLLoss的组合
    #LogSoftmax将最终对每个标签的概率进行归一化（概率之和为1）,公式自查
    #NLLLoss计算负对数似然损失:对关注的真实标签（应该得到的推理结果）的预测取负，并将多轮预测得到的负对数似然损失求和
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # -----------------------4.训练网络-----------------------------------
    for epoch in range(2):  # 循环数据集两次
        #init loss
        running_loss = 0.0
        #enumerate()参数：可迭代对象，默认从0开始计数的索引。
        for i ,data in enumerate(trainloader, 0):
            inputs, labels =data
            # 将数据移动到GPU上
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # 将梯度清零，以免梯度累积
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()     # 反向传播，计算梯度
            optimizer.step()    # 更新权重
    
            #打印状态信息
            running_loss += loss.item()  # 累加损失
            if i % 2000 == 1999:    # 每2000个小批量打印一次状态信息
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))  # 输出当前损失
                running_loss = 0.0
    
    print('Finished Training')
    
    # -----------------------5.根据测试机测试网络--------------------------
    dataiter = iter(testloader)
    images, labels = dataiter.__next__()
    # 将图像和标签移动到GPU上
    images, labels = images.to(device), labels.to(device)

    # 显示图片
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # 放入n批次图像
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)  # 返回每行中最大值的索引
    # 打印前4批次预测结果
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    
    #------------------------整个测试集的正确率-------------------------
    correct = 0  # 初始化正确预测的数量
    total = 0    # 初始化总数量
    with torch.no_grad():  # 不需要计算梯度，节省内存和加速计算
        for data in testloader:  # 遍历测试集中的所有数据
            images, labels = data  # 获取输入图像和标签
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)  # 前向传播，获取输出预测
            _, predicted = torch.max(outputs.data, 1)  # 获取预测的类别索引(每列中最大值的索引)
            total += labels.size(0)  # 更新总数量,size(0)返回这个批次的第一维数据：标签数
            correct += (predicted == labels).sum().item()  # 更新正确预测的数量,对布尔张量求和取标量

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))  # 输出准确率

    #-------------------------每个类别的准确率--------------------------
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别索引(每列中最大值的索引)
            c = (predicted == labels).squeeze()
            for i in range(4):  # 遍历这个批次的所有数据
                label = labels[i]  # 获取真实标签
                class_correct[label] += c[i].item()  # 如果预测正确，则增加该类别的正确预测数量
                class_total[label] += 1  # 增加该类别的总数量

    for i in range(10):        
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

