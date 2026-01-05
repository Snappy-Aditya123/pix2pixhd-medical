import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from tqdm import tqdm
import random
import shutil
import glob
import torchvision
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

def one_hot_encode_label(label_map, num_classes):
    B, H, W = label_map.shape
    out = torch.zeros(B, num_classes, H, W, device=label_map.device)
    out.scatter_(1, label_map.unsqueeze(1), 1.0)
    return out


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return x, self.pool(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size()[2:] != skip.size()[2:]:
            x = F.interpolate(x, size=skip.shape[2:])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (5 levels)
        self.d1 = Down(3, 64)         # 256 → 128
        self.d2 = Down(64, 128)       # 128 → 64
        self.d3 = Down(128, 256)      # 64 → 32
        self.d4 = Down(256, 512)      # 32 → 16
        self.d5 = Down(512, 1024)     # 16 → 8

        # Bottleneck
        self.mid = DoubleConv(1024, 2048)

        # Decoder (upsampling)
        self.u5 = Up(2048, 1024)
        self.u4 = Up(1024, 512)
        self.u3 = Up(512, 256)
        self.u2 = Up(256, 128)
        self.u1 = Up(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1, p1 = self.d1(x)
        s2, p2 = self.d2(p1)
        s3, p3 = self.d3(p2)
        s4, p4 = self.d4(p3)
        s5, p5 = self.d5(p4)

        m = self.mid(p5)

        x = self.u5(m, s5)
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        x = self.u1(x, s1)

        return self.out(x)
