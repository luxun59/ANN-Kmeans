
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim


import torchvision.transforms as transforms



class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        classes = ('write_dotted','write_soild','yellow_dotted','yellow_soild','yellow_double')
        for line in fh:
            line = line.rstrip()
            if(len(line)>0):
                words = line.split()
                imgs.append((words[0], classes.index(words[1])))
                self.imgs = imgs 
                self.transform = transform
                self.target_transform = target_transform
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB') 
        # img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    def __len__(self):
        return len(self.imgs)





