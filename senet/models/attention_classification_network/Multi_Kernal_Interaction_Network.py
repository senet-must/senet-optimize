import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MKI_Net(nn.Module):
    def __init__(self, in_channels, out_channels, M=3, G=8, stride=1):
        print("MKI_Net")
        super(MKI_Net, self).__init__()
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3 + i * 2, stride=stride, padding=0, groups=G),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.map1 = nn.Linear(out_channels * 3, 512)
        self.map2 = nn.Linear(512, out_channels)
        self.fc = nn.Linear(out_channels, 200)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    # V1
    # def forward(self, x, train_flag="train"):
    #     for i, conv in enumerate(self.convs):
    #         fea = conv(x)
    #         fea = self.avgpool(fea)
    #         fea = torch.flatten(fea, 1)
    #         if i == 0:
    #             feas = fea
    #         else:
    #             feas = torch.cat([feas, fea], dim=1)
    #
    #     feas = self.map1(feas)
    #     feas = self.drop(feas)
    #     feas = self.map2(feas)
    #
    #     return feas

    # V2
    def forward(self, x, train_flag="train"):
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            fea = self.avgpool(fea)
            fea = torch.flatten(fea, 1)  # [64, 1, 1, 512]
            if i == 0:
                feas = fea
                x1 = fea
            if i == 1:
                x2 = fea
            if i == 2:
                x3 = fea
            else:
                feas = torch.cat([feas, fea], dim=1)    # [64, 1, 1, 1536]

        if train_flag == "train":
            feas = self.map1(feas)  # [64, 1, 1, 512]
            feas = self.drop(feas)
            feas = self.map2(feas)   # [64, 1, 1, 512]


            gate1 = torch.mul(feas, x1)
            gate1 = self.sigmoid(gate1)
            gate2 = torch.mul(feas, x2)
            gate2 = self.sigmoid(gate2)
            gate3 = torch.mul(feas, x3)
            gate3 = self.sigmoid(gate3)

            x1 = torch.mul(gate1, x1) + x1
            x2 = torch.mul(gate1, x2) + x2
            x3 = torch.mul(gate1, x3) + x3


        features = torch.cat([x1, x2, x3], dim=1)   # [64, 1536]

        return features