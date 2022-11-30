'''
Author: luxun59 luxun59@126.com
Date: 2022-05-05 18:45:29
LastEditors: luxun59 luxun59@126.com
LastEditTime: 2022-11-30 10:54:03
FilePath: \lenet_test\predict.py
Description: 

Copyright (c) 2022 by luxun59 luxun59@126.com, All Rights Reserved. 
'''
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet

import time
import numpy as np
import cv2





class lenet_class():
    mydict = {'write_dotted':(0,1),'write_solid':(0,1),
          'yellow_dotted':(0,1),'yellow_solid':(0,1),
          'yellow_double':(0,1)}
    classes = ('write_dotted','write_solid','yellow_dotted','yellow_solid','yellow_double')
    def __init__(self):
        self.transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.net = LeNet()
        self.net.load_state_dict(torch.load('Lenet4.pth'))

    def lenent_predict(self,frame):
        #im = Image.open('1231.png')
        im = Image.fromarray(np.uint8(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
        time_start = time.time()
        im = self.transform(im)  # [C, H, W]
        im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

        with torch.no_grad():
            outputs = self.net(im)
            predict = torch.max(outputs, dim=1)[1].data.numpy()
        time_end = time.time()
        print(self.classes[int(predict)])
        #print(time_end - time_start)
        return self.mydict[self.classes[int(predict)]]

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('write_dotted','write_solid','yellow_dotted','yellow_solid','yellow_double')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet1.pth'))

    im = Image.open('4.png')
    time_start = time.time()
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    time_end = time.time()
    print(classes[int(predict)])
    print(time_end - time_start)
    


if __name__ == '__main__':
    #main()
    mylenet = lenet_class()
    #im = Image.open('1.png')
    im = cv2.imread('4.png')
    # cv2.imshow('0',im)
    # cv2.waitKey(1)
    # himg = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('1',himg)
    # cv2.waitKey(1)


    # im1 = cv2.imread('1.png')
    # cv2.imshow('01',im1)
    # cv2.waitKey(1)
    # himg1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('11',himg1)
    # cv2.waitKey(1)

    mylenet.lenent_predict(im)