### 1.文件目录及用途
```text
    checkpoints: 保存模型
    datasets:数据集存放位置
    historys:训练历史
    logs:训练日志
    models:模型定义

    config.py:配置文件
    main.py:主函数
    utils.py:存放辅助函数
```

### 2.运行（base环境下即可）
```text
    python main.py
```

### 3.常用linux 命令
```text
    查看GPU
    watch -n 1 nvidia-smi 每秒更新GPU信息，ctrl + c 结束
    分屏
    tmux
    左右分屏：ctrl + b 然后 按 %
    上下分屏：ctrl + b 然后 按 "
    切换屏： ctrl +b 然后按 上下左右键选择想要的屏
    其他命令linux自己百度吧
```

### 4.pycharm连接服务器
```text
    原理：本地修改代码后自动同步到服务器相应的区域，运行时，调用服务器解释器运行服务器上的代码，并把输出结果回传到本地
    步骤: 1.克隆代码到本地
          2.将本地代码和服务代码绑定，使之修改后能同步到服务器
          3.将解释器设为远程的解释器
    参考链接：https://www.matpool.com/supports/doc-pycharm-connect-matpool/
```

### 5.调参
| 数据集 | 较优参数 | 图片增强|准确率(使用预训练权重)|备注
|---|----|---|---|---|
|CUB200|SGD(model.parameters(), lr=1e-4, momentum=0.99,epoch<=150 )| Resize(256),RandomRotation(15),RandomCrop(224)|78-79%|21-02-02
|DOG|SGD(model.parameters(), lr=1e-4, momentum=0.99, epoch<=150)| Resize(256),RandomRotation(15),RandomCrop(224)|86-87%|21-02-02
|CAR|SGD(model.parameters(), lr=1e-2, momentum=0.90, epoch<=150,每30epoch学习率衰减为原来的0.1)| Resize(512),RandomRotation(15),RandomCrop(448)|84-85%|21-02-02
    

