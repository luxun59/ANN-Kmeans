'''
Author: luxun59 luxun59@126.com
Date: 2022-12-04 20:07:56
LastEditors: luxun59 luxun59@126.com
LastEditTime: 2022-12-04 20:13:15
FilePath: \pytorch_bp\model.py
Description: 

Copyright (c) 2022 by luxun59 luxun59@126.com, All Rights Reserved. 
'''
import torch.nn as nn
import torch.nn.functional as F



class BPnet(nn.Module):
    def __init__(self):
        super(BPnet,self).__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)


    def forward(self, x):
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x