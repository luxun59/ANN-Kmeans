import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms

from mydataset import MyDataset

# def main():
#     transform = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#     # 50000张训练图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
#     train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                              download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
#                                                shuffle=True, num_workers=0)

#     # 10000张验证图片
#     # 第一次使用时要将download设置为True才会自动去下载数据集
#     val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                            download=True, transform=transform)
#     val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
#                                              shuffle=True, num_workers=0)
#     val_data_iter = iter(val_loader)
#     val_image, val_label = val_data_iter.next()

#     # classes = ('plane', 'car', 'bird', 'cat',
#     #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#     net = LeNet()
#     loss_function = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(net.parameters(), lr=0.001)

#     for epoch in range(5):  # loop over the dataset multiple times

#         running_loss = 0.0
#         for step, data in enumerate(train_loader, start=0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, labels = data

#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = loss_function(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             # print statistics
#             running_loss += loss.item()
#             if step % 500 == 499:  # print every 500 mini-batches
#                 with torch.no_grad():
#                     outputs = net(val_image)  # [batch, 10]
#                     predict_y = torch.max(outputs, dim=1)[1]
#                     accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

#                     print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
#                           (epoch + 1, step + 1, running_loss / 500, accuracy))
#                     running_loss = 0.0

#     print('Finished Training')

#     save_path = './Lenet.pth'
#     torch.save(net.state_dict(), save_path)


# if __name__ == '__main__':
#     main()



# transform = transforms.Compose(
#         [transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


pipline_train = transforms.Compose([
    #随机旋转图片
    transforms.RandomHorizontalFlip(),
    #将图片尺寸resize到32x32
    transforms.Resize((32,32)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transforms.Normalize((0.1307,),(0.3081,))    ])

pipline_test = transforms.Compose([
    #将图片尺寸resize到32x32
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #transforms.Normalize((0.1307,),(0.3081,))])

train_data = MyDataset('./data/train.txt', transform=pipline_train)
test_data = MyDataset('./data/test.txt', transform=pipline_test)

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4, shuffle=False)





def train_runner(trainloader,testloader):
    # #训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    # # model.train()
    # #创建模型，部署gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # #model = LeNet().to(device)
    # #定义优化器
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # total = 0
    # correct =0.0
    
    val_data_iter = iter(testloader)
    val_image, val_label = val_data_iter.next()

    batches = 8

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(trainloader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(type(inputs))
            # print(step)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % batches == batches-1:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / batches, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':

    train_runner(trainloader,testloader)

    # #enumerate迭代已加载的数据集,同时获取数据和数据下标
    # for i, data in enumerate(trainloader, 0):
    #     inputs, labels = data
    #     #把模型部署到device上
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     #初始化梯度
    #     optimizer.zero_grad()
    #     #保存训练结果
    #     outputs = model(inputs)
    #     #计算损失和
    #     #多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
        
    #     loss = loss_function(outputs, labels)
    #     #获取最大概率的预测结果
    #     #dim=1表示返回每一行的最大值对应的列下标
    #     predict = outputs.argmax(dim=1)
    #     total += labels.size(0)
    #     correct += (predict == labels).sum().item()
    #     #反向传播
    #     loss.backward()
    #     #更新参数
    #     optimizer.step()
    #     if i % 100 == 0:
    #         #loss.item()表示当前loss的数值
    #         print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, loss.item(), 100*(correct/total)))
    #         Loss.append(loss.item())
    #         Accuracy.append(correct/total)
    # return loss.item(), correct/total
