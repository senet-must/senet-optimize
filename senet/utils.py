import torch
import numpy as np


from torch.utils.data import DataLoader
import torchvision
import os
# from .RandomScale import RandomScale
from torchvision import transforms as T


def writeLog(path,log):
    """
    写日志,每个EPOCH写入一次日志
    :param path: 日志路径
    :param logs: 日志,list格式
    :return: None
    """
    f = open(path,'a+')
    f.write(log)


def writeHistory(path,historys):
    """
    保存训练历史
    :param path: 保存的路径
    :param historys: 保存数据（字典格式的数据）
    :return: None
    """
    np.save(path, historys)


def readHistory(path):
    """
    读取训练历史 字典格式
    :param path: 读取文件的路径
    :return: 字典格式数据
    """
    data = np.load(path,allow_pickle=True).item()
    return data


def saveModel(model,path):
    """
    保存模型
    :param model: 模型
    :param path: 保存路径
    :return: None
    """
    torch.save(model, path)
    print('save model in {}'.format(path))


def loadModel(path):
    """
    加载模型
    :param path:模型路径
    :return: 模型
    """
    model = torch.load(path)
    print('load model in {}'.format(path))
    return model

def dataTransform(mode='train'):
    """
    默认数据增强
    :param mode: 训练模式 or 其他模式
    :return: transforms
    """
    print('默认数据增强方式')
    transforms = None
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if mode == 'train':
        transforms = T.Compose([
            T.Resize(256),
            T.RandomRotation(15),
            # T.RandomResizedCrop(224,scale=(0.85,1.15)),
            T.RandomCrop(224),
            T.ToTensor(),
            normalize
        ])
    else:
        transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

    return transforms



def chooseData(name,batchSize,worker,trainTransforms=None,testTransforms=None):
    """
    选择数据集
    :param name: 数据集名称
    :param batchSize: batchSize大小
    :param worker: 线程数
    :return: 训练、测试、验证集的loader以及 各个数据集的长度
    """
    trainLoader = None
    testLoader = None
    validLoader = None

    dataTrain = []
    dataTest = []
    dataValid = []


    datasets = os.path.join(os.getcwd(), 'datasets')

    if trainTransforms is None:
        trainTransforms = dataTransform('train')
    if testTransforms is None:
        testTransforms = dataTransform('test')


    # CUB200
    if name == 'CUB200':
        root = os.path.join(datasets, 'CUB_200_2011')
        trainRoot = os.path.join(root,'train')
        testRoot = os.path.join(root,'test')

        dataTrain = torchvision.datasets.ImageFolder(trainRoot,transform=trainTransforms)
        dataTest = torchvision.datasets.ImageFolder(testRoot,transform=testTransforms)

        trainLoader = torch.utils.data.DataLoader(dataTrain,
                                                   batch_size=batchSize,
                                                   shuffle=True)


        testLoader = torch.utils.data.DataLoader(dataTest,
                                                   batch_size=batchSize,
                                                   shuffle=False)

    if name == "STANFORDCARS":
        # 数据
        from datasets.Car import Car
        root = os.path.join(datasets, 'StanfordCars')

        dataTrain = Car(root, 'train', trainTransforms)
        dataTest = Car(root, 'test', testTransforms)

        # 数据生成器
        trainLoader = DataLoader(dataset=dataTrain,
                                 batch_size=batchSize,
                                 num_workers=worker,
                                 shuffle=True)

        testLoader = DataLoader(dataset=dataTest,
                                batch_size=batchSize,
                                num_workers=worker,
                                shuffle=False)

    if name == "STANFORDDOGS":
        # 数据
        from datasets.Dog import Dog
        root = os.path.join(datasets, 'StanfordDogsDataset')
        dataTrain = Dog(root, 'train')
        dataTest = Dog(root, 'test')

        # 数据生成器
        trainLoader = DataLoader(dataset=dataTrain,
                                 batch_size=batchSize,
                                 num_workers=worker,
                                 shuffle=True)

        testLoader = DataLoader(dataset=dataTest,
                                batch_size=batchSize,
                                num_workers=worker,
                                shuffle=False)


    return trainLoader, testLoader, validLoader,len(dataTrain),len(dataTest),len(dataValid)

