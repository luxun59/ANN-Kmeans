'''
Author: luxun59 luxun59@126.com
Date: 2022-12-04 20:13:04
LastEditors: luxun59 luxun59@126.com
LastEditTime: 2022-12-04 22:18:07
FilePath: \pytorch_bp\train.py
Description: 

Copyright (c) 2022 by luxun59 luxun59@126.com, All Rights Reserved. 
'''

import torch
import torchvision
import torch.nn as nn
from model import BPnet
import torch.optim as optim










def main():
    net = BPnet()

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)

    loss_function = nn.MSELoss()
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03)


    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X) ,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')