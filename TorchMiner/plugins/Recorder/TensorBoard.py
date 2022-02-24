# -*- coding:utf-8 -*-
from tensorboardX import SummaryWriter

from TorchMiner.plugins import BasePlugin


class TensorboardDrawer(BasePlugin):
    """To visualize everything in training process using tensorboard"""

    def __init__(self, input_to_model=None):
        super(TensorboardDrawer, self).__init__()
        self.input_to_model = input_to_model

    def prepare(self, miner, *args, **kwargs):
        super(TensorboardDrawer, self).prepare(miner)
        self.writer = SummaryWriter(log_dir=self.miner.experiment_dir)

    def after_init(self, **ignore):
        if self.input_to_model:
            try:
                self.writer.add_graph(self.miner.model, self.input_to_model)
            except Exception as e:
                self.logger.error(f"{e} occurred when visializing model")

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
