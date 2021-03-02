import torch
from torch import nn
from ..attention_block.FcaNet import MultiSpectralAttentionLayer

class Multi_Kernal(nn.Module):
    def __init__(self, in_channels, out_channels, M=3, G=8, stride=1):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(Multi_Kernal, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        self.M = M
        self.features = in_channels
        self.convs = nn.ModuleList([])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fcapool = MultiSpectralAttentionLayer(512, 512, c2wh[512], c2wh[512], reduction=16,
                                                       freq_sel_method='top16')
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3 + i * 2, stride=stride, padding=0, groups=G),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        # self.fc = nn.Linear(in_channels, d)
        # self.fcs = nn.ModuleList([])
        # for i in range(M):
        #     self.fcs.append(
        #         nn.Linear(d, features)
        #     )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            # print(fea.size())
            # GAP方法
            # fea = self.avgpool(fea)
            # 频域方法
            fea = self.fcapool(fea)
            fea = torch.flatten(fea, 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
            # print(fea.size())

        x = feas
        # print(x.size())
        return x
