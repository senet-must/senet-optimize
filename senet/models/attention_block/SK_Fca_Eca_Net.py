import torch
from torch import nn
from .Fca_Plus_Eca_Net import MultiSpectralAttentionLayer

class SKConv(nn.Module):
    def __init__(self, features, M=3, G=8, r=16, stride=1, L=32, k_size=5):
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
        super(SKConv, self).__init__()
        c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7)])
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        self.fpe = MultiSpectralAttentionLayer(2048, c2wh[512], c2wh[512], freq_sel_method = 'top16', k_size=k_size)
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv1d(features, features, kernel_size=k_size, bias=False, groups=features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        y = x
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        print(feas.size())
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        # fea_s = fea_U.mean(-1).mean(-1)
        print(fea_U.size())
        fea_s = self.fpe(fea_U)
        # fea_z = self.fc(fea_s)
        # fea_s = fea_s.transpose(0, 1)
        b, c, _, _ = fea_s.size()
        print(fea_s.size())
        fea_s = fea_s.view(b, 1, c)
        print(fea_s.size())
        print(c, b)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_s).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
            print(attention_vectors.size())
        attention_vectors = self.softmax(fea_s)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        print(feas.size())
        print(attention_vectors.size())
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v + y
