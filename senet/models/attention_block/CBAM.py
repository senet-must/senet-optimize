import torch
import torch.nn as nn
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        print("cbam_channel")
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class DCT_1d_Pooling(nn.Module):
    def __init__(self, channel, height, width):
        super(DCT_1d_Pooling, self).__init__()
        self.register_buffer('weight', self.dct_spatial_pooling(channel, height,
            width))
        
    # 一维DCT
    def get_1d_dct(self, i, freq, L):
        result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
        return result

    # 空间维度的1维频域变换
    def dct_spatial_pooling(self, channel, height, width):
        dct_1d_weight = torch.zeros(channel, height, width)
        for freq in range(0, channel):
            for i in range(0, channel):
                dct_1d_weight[freq, :, :] += self.get_1d_dct(i, freq, channel)
        
        return dct_1d_weight

    def forward(self, x):
        x = x * self.weight
        result = torch.sum(x, dim=1, keepdim=True)
        return result



class SpatialAttention(nn.Module):
    def __init__(self, channel, height, width, kernel_size=7):
        print("cbam_spatial")
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.DCT_1d = DCT_1d_Pooling(channel, height, width)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    # CBAM中的GAP方法的注意力机制
    def forward(self, x):
        y = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * y + y

    # 一维DCT的空间注意力机制
    # def forward(self, x):
    #     y = x
    #     x = self.DCT_1d(x)
    #     x = self.conv1(x)
    #     return self.sigmoid(x) * y + y

    # 一维DCT池化
    # def forward(self, x):
    #     y = self.DCT_1d(x)
    #     return y
