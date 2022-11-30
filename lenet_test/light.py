
import os
import cv2

import numpy as np
import random

# right_path = os.path.join("mypng","right")
# left_path = os.path.join("mypng","left")

# data_dir = os.path.join('png')

datapath='data/LANE'
savepath = 'data/LANE1'

# os.makedirs("data/LANE1/write_dotted")
os.makedirs("data/LANE1/write_soild")
os.makedirs("data/LANE1/yellow_soild")



def imgBrightness(img1, b, filename, savepath,savedir):
    h, w, ch = img1.shape
    blank = np.zeros([h, w, ch], img1.dtype)
    c = random.uniform(0.2, 1.8)
    rst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    cv2.imwrite(savepath + '/' + savedir + "/" + "l" + filename, rst)  # 保存图片

file_pathname = datapath + "/write_dotted"
for filename in os.listdir(file_pathname):
    img_pat = file_pathname + "/" + filename  # 图片地址
    img = cv2.imread(img_pat)  # 读图片
    imgBrightness(img, 3, filename, savepath,'write_dotted')


file_pathname = datapath + "/write_soild"
for filename in os.listdir(file_pathname):
    img_pat = file_pathname + "/" + filename  # 图片地址
    img = cv2.imread(img_pat)  # 读图片
    imgBrightness(img, 3, filename, savepath,'write_soild')


file_pathname = datapath + "/yellow_soild"
for filename in os.listdir(file_pathname):
    img_pat = file_pathname + "/" + filename  # 图片地址
    img = cv2.imread(img_pat)  # 读图片
    imgBrightness(img, 3, filename, savepath,'yellow_soild')












