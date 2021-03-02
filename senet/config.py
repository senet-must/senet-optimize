# 配置文件 暂未使用
class Config():
    env = 'default'  # visdom 环境
    model = 'AlexNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    flowers_root = './dataset/102flowers/'  # 训练集存放路

    batch_size = 128  # batch size
    use_gpu = True  # use GPU or not
    num_workers = 2  # how many workers for loading data

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

config = Config()