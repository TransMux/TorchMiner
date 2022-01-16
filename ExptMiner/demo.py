# -*- coding:utf-8 -*-
import numpy as np
import torch
from TorchMiner import Miner
from torch import nn
from torch import tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        self.fn1 = nn.Linear(1, 10)
        self.fn2 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        return self.fn2(self.fn1(x)).squeeze(dim=1)


class iData(torch.utils.data.Dataset):
    def __init__(self, ran):
        self.ran = ran

    @staticmethod
    def f(x):
        return np.cos(4 * x) + np.arctan(5 * x) + 3

    def __getitem__(self, item):
        return tensor(self.ran[item], dtype=torch.float), tensor(self.f(self.ran[item]), dtype=torch.float)

    def __len__(self):
        return len(self.ran)


train_dataset = iData(range(0, 100000))
val_dataset = iData(range(-10000, 0))
model = Regression()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)

optim = AdamW(model.parameters(), lr=5e-10)
loss_function = nn.MSELoss()
# loss_function = nn.CrossEntropyLoss()

trainer = Miner(
    alchemistic_directory='./experiments/',  # 日志根目录
    train_dataloader=train_loader,  # 训练dataloader
    val_dataloader=val_loader,  # 验证dataloader

    model=model,  # 模型
    loss_func=loss_function,  # 损失函数
    optimizer=optim,  # 优化器
    experiment="exp1",
    resume=True,  # 是否自动加载之前的last模型接着训练
    eval_epoch=1,  # 多少轮进行一次评估
    persist_epoch=1,  # 多少轮进行一次checkpoint的保存
    max_epochs=50,  # 训练轮数
    # plugins=[
    #     MultiClassesClassificationMetricWithLogic(),
    #     #             NoiseSampleDetector(metric=torch.nn.CrossEntropyLoss(reduction='none')) ## 这个插件也不能用，会导致 IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
    # ],
    accumulated_iter=1,  # 累积多少次迭代后进行参数更新
    #         sheet=GoogleSheet('1DnEbU6LMKrER03YxsRONB141tK8JNQ7xb0k3wejQjLE', './trainer.io')
)

trainer.train()
