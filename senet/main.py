import torch

from torch.autograd import Variable
from models.Net import Net
from models._Alexnet import  BuildAlexNet
from models.resnet_base import CBMA_resnet
import os
from utils import saveModel,loadModel,chooseData,writeHistory,writeLog
import time



def train(modelConfig,dataConfig,logConfig):
    """
    训练
    :param modelConfig: 模型配置
    :param dataConfig: 数据配置
    :param logConfig:  日志配置
    :return:
    """
    # 模型配置
    model = modelConfig['model']
    criterion = modelConfig['criterion']
    optimzer = modelConfig['optimzer']
    epochs =  modelConfig['epochs']
    device = modelConfig['device']

    #数据加载器
    trainLoader = dataConfig['trainLoader']
    validLoader = dataConfig['validLoader']
    trainLength =  dataConfig['trainLength']
    validLength = dataConfig['validLength']

    # 日志及模型保存
    modelPath = logConfig['modelPath']
    historyPath = logConfig['historyPath']
    logPath = logConfig['logPath']
    lastModelPath = logConfig['lastModelPath']


    trainLosses = []
    trainAcces = []
    validLosses = []
    validAcces = []
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('train is starting in ' + now)
    bestAcc = 0.

    for epoch in range(epochs):
        print("Epoch{}/{}".format(epoch, epochs))
        print("-" * 10)

        trainLoss, trainAcc = oneEpoch_train(model,trainLoader,optimzer,criterion,device)
        validLoss, validAcc = oneEpoch_valid(model,validLoader,criterion,device)

        trainLoss = trainLoss / len(trainLoader)
        trainAcc =  trainAcc / trainLength
        validLoss = validLoss / len(validLoader)
        validAcc = validAcc / validLength

        # trainLosses.append(trainLoss)
        # trainAcces.append(trainAcc)
        #
        # validLosses.append(validLoss)
        # validAcces.append(validAcc)
        # 模型验证有进步时,保存模型
        if validAcc > bestAcc:
            bestAcc = validAcc
            # saveModel(model,modelPath)

        # 训练日志
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        trainLog = now + " Train loss is :{:.4f},Train accuracy is:{:.4f}%\n".format(trainLoss, 100 * trainAcc)
        validLog = now + " Valid loss is :{:.4f},Valid accuracy is:{:.4f}%\n".format(validLoss, 100 * validAcc)
        bestlog = now + ' bestAcc is {:.4f}%'.format(100 * bestAcc)
        log = trainLog + validLog

        print(log+bestlog)

        # 训练历史 每个EPOCH都覆盖一次
        # history = {
        #     'trainLosses':trainLosses,
        #     'trainAcces':trainAcces,
        #     'validLosses':validLosses,
        #     'validAcces':validAcces
        # }

        writeLog(logPath,log)
        # writeHistory(historyPath,history)

        # 保存最新一次模型
        # saveModel(model,lastModelPath)


def oneEpoch_train(model,dataLoader,optimzer,criterion,device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param optimzer: 优化器
    :param criterion: loss计算函数
    :return: loss acc
    """
    # 模式

    model.train()
    loss = 0.
    acc = 0.
    for (inputs, labels) in dataLoader:
        # 使用某个GPU加速图像 label 计算
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels = Variable(inputs), Variable(labels)

        # 梯度设为零，求前向传播的值
        optimzer.zero_grad()

        outputs = model(inputs)
        _loss = criterion(outputs, labels)

        # 反向传播
        _loss.backward()
        # 更新网络参数
        optimzer.step()


        _, preds = torch.max(outputs.data, 1)
        loss += _loss.item()
        acc += torch.sum(preds == labels).item()

    return loss,acc


def oneEpoch_valid(model,dataLoader,criterion,device):
    """
    训练一次 或者 验证/测试一次
    :param model: 模型
    :param dataLoader: 数据加载器
    :param criterion: loss计算函数
    :return: loss acc
    """
    with torch.no_grad():
        model.eval()
        loss = 0.
        acc = 0.
        for (inputs, labels) in dataLoader:

            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)
            _loss = criterion(outputs, labels)


            _, preds = torch.max(outputs.data, 1)
            loss += _loss.item()
            acc += torch.sum(preds == labels).item()

    return loss,acc

def _stanfordDogs():
    """
     StanfordDogs数据集
     :return:
     """

    # 定义模型 定义评价 优化器等
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Net(models.resnet50(pretrained=True), 120)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, )
    torch.optim.lr_scheduler.StepLR(optimzer, 50, gamma=0.1, last_epoch=-1)
    epochs = 150
    batchSize = 80
    worker = 2

    modelConfig = {
        'model':model,
        'criterion':criterion,
        'optimzer':optimzer,
        'epochs':epochs,
        'device':device
    }


    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('STANFORDDOGS', batchSize,worker)
    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader':trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanforddogs.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanforddogs_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_stanforddogs.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_stanforddogs.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath':historyPath,
        'logPath':logPath,
        'lastModelPath':lastModelPath
    }

    train(modelConfig,dataConfig,logConfig)

def _CUB200():
    """
    CUB200数据集
    :return:
    """
    # 定义模型 定义评价 优化器等
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = Net(CBMA_resnet.resnet50(pretrained=True), 200)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, )
    torch.optim.lr_scheduler.StepLR(optimzer, 50, gamma=0.1, last_epoch=-1)
    epochs = 150
    batchSize = 64
    worker = 2

    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('CUB200', batchSize,worker)

    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_CUB200.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_CUB200_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_CUB200.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_CUB200.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)

def _stanfordCars():
    """
       StanfordCars数据集
       :return:
       """
    # 定义模型 定义评价 优化器等
    n_output = 196
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = Net(models.resnet50(pretrained=True), n_output)
    # model = BuildAlexNet('pre', n_output)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimzer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.90, )
    torch.optim.lr_scheduler.StepLR(optimzer, 30, gamma=0.1, last_epoch=-1)
    epochs = 150
    batchSize = 24
    worker = 2

    modelConfig = {
        'model': model,
        'criterion': criterion,
        'optimzer': optimzer,
        'epochs': epochs,
        'device': device
    }

    from torchvision import transforms as T
    # 自定义数据增强方式
    # normalize 加快收敛
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trainTransforms = T.Compose([
        T.Resize(512),
        T.RandomRotation(15),
        # T.RandomResizedCrop(224,scale=(0.85,1.15)),
        T.RandomCrop(448),
        T.ToTensor(),
        normalize
    ])

    testTransforms = T.Compose([
        T.Resize(512),
        # T.RandomCrop(224),
        T.CenterCrop(448),
        T.ToTensor(),
        normalize
    ])

    trainLoader, testLoader, validLoader, trainLength, testLength, validLength = chooseData('STANFORDCARS', batchSize, worker,trainTransforms,testTransforms)

    # 没有验证集，所以使用测试集来做验证集
    dataConfig = {
        'trainLoader': trainLoader,
        'validLoader': testLoader,
        'trainLength': trainLength,
        'validLength': testLength
    }

    modelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanfordcars.pth')
    lastModelPath = os.path.join(os.getcwd(), 'checkpoints', '_stanfordcars_last.pth')
    historyPath = os.path.join(os.getcwd(), 'historys', '_stanfordcars.npy')
    logPath = os.path.join(os.getcwd(), 'logs', '_stanfordcars.txt')

    logConfig = {
        'modelPath': modelPath,
        'historyPath': historyPath,
        'logPath': logPath,
        'lastModelPath': lastModelPath
    }

    train(modelConfig, dataConfig, logConfig)



if __name__ == '__main__':
    print(torch.__version__)
    # _stanfordCars()
    # _stanfordDogs()
    _CUB200()
