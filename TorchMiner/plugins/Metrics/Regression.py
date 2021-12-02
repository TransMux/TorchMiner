# -*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

from TorchMiner import BasePlugin


class RegressionAccuracy(BasePlugin):
    def before_train_epoch_start(self, *args, **kwargs):
        self.predicts = []
        self.label = []

    @staticmethod
    def forward(predicts, original_data):
        raw_output = predicts.detach().cpu().numpy()
        # predicts = np.argmax(raw_output, axis=1)  # Batch first
        label = original_data[1].cpu().numpy()  # label
        return raw_output, label  # 1 dim ndarray

    def after_val_iteration_ended(self, predicts, data, *args, **kwargs):
        data = self.forward(predicts, data)
        self.predicts.append(data[0])
        self.label.append(data[1])

    def after_epoch_end(self, *args, **kwargs):
        self.predicts = np.concatenate(self.predicts)
        self.label = np.concatenate(self.label)
        acc = mean_absolute_percentage_error(self.label, self.predicts)
        self.logger.info(f"Val Accuracy:{acc}")
