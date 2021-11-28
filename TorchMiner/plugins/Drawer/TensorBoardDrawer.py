# -*- coding:utf-8 -*-
import os

from TorchMiner.plugins.Drawer import Drawer
from tensorboardX import SummaryWriter


class TensorboardDrawer(Drawer):
    """To visualize everything in training process using tensorboard"""

    def __init__(self, miner, state=None):
        super().__init__(miner, state)
        self.writer = SummaryWriter(
            log_dir=os.path.join(miner.alchemistic_directory, miner.experiment)
        )

    def scalars(self, x, value, graph):
        """
        Add a scalar on a graph
        :param x:
        :param value:
        :param graph:
        :return:
        """
        if graph not in self.state:
            self.state[graph] = 0
        key = "{}/{}".format(self.miner.experiment, graph)
        if isinstance(value, dict):
            self.writer.add_scalars(key, value, self.state[graph])
        else:
            self.writer.add_scalar(key, value, self.state[graph])
        self.state[graph] += 1
