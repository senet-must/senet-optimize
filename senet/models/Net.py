from torch import nn
import torch

# pretrained_params = torch.load('Pretrained_Model'）
# model = The_New_Model(xxx)
# model.load_state_dict(pretrained_params.state_dict(), strict=False)

class Net(nn.Module):
    def __init__(self, model, CLASS=102):
        super(Net, self).__init__()
        # 选择resnet 除最后一层的全连接，改为CLASS输出
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # 可以选择冻结卷积层
        # for p in self.parameters():
        #     p.requires_grad = False
        # self.fc = nn.Linear(in_features=512, out_features=CLASS)
        self.fc = nn.Linear(in_features=2048, out_features=CLASS)
    def forward(self, x):
        x = self.resnet(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class _inception_v3(nn.Module):

    def __init__(self, model, CLASS=102):
        super(_inception_v3, self).__init__()
        # 选择resnet 除最后一层的全连接，改为CLASS输出
        self.inception_v3 = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(in_features=2048, out_features=CLASS)


    def forward(self, x):
        x = self.inception_v3(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x




