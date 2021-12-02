# -*- coding:utf-8 -*-
from tensorboardX import SummaryWriter

from TorchMiner.plugins import BasePlugin


class TensorboardDrawer(BasePlugin):
    """To visualize everything in training process using tensorboard"""

    def __init__(self, input_to_model):
        super(TensorboardDrawer, self).__init__()
        self.input_to_model = input_to_model

    def prepare(self, miner, *args, **kwargs):
        super(TensorboardDrawer, self).prepare(miner)
        self.writer = SummaryWriter(log_dir=self.miner.experiment_dir)

    def after_init(self, **ignore):
        self.writer.add_graph(self.miner.model, self.input_to_model)

    def after_train_epoch_end(self, train_loss, epoch, **ignore):
        self.writer.add_scalar("Loss/train", train_loss, global_step=epoch)

    def after_val_epoch_end(self, val_loss, epoch, **ignore):
        self.writer.add_scalar("Loss/val", val_loss, global_step=epoch)

    # --- Outer APIs ---
    def scalar(self, label, value, epoch=None, **ignore):
        if not epoch:
            epoch = self.miner.current_epoch
        self.writer.add_scalar(label, value, epoch)

    def figure(self, label, value, epoch=None, **ignore):
        if not epoch:
            epoch = self.miner.current_epoch
        self.writer.add_image(label, value, epoch)
# def scalars(self, x, value, graph):
#     """
#     Add a scalar on a graph
#     :param x:
#     :param value:
#     :param graph:
#     :return:
#     """
#     if graph not in self.state:
#         self.state[graph] = 0
#     key = "{}/{}".format(self.miner.experiment, graph)
#     if isinstance(value, dict):
#         self.writer.add_scalars(key, value, self.state[graph])
#     else:
#         self.writer.add_scalar(key, value, self.state[graph])
#     self.state[graph] += 1
