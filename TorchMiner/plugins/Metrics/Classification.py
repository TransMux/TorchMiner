# -*- coding:utf-8 -*-
import io

import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from TorchMiner import BasePlugin


class MultiClassesClassificationMetric(BasePlugin):
    """MultiClassesClassificationMetric
    This can be used directly if your loss function is torch.nn.CrossEntropy
    """
    requirements = ["TensorboardDrawer"]

    def __init__(
            self,
            accuracy=True,
            confusion=True,
            kappa_score=True,
            report=True,
            backend="TensorboardDrawer",
            forward=None
    ):
        super().__init__()
        self.accuracy = accuracy
        self.confusion_matrix = confusion
        self.kappa_score = kappa_score
        self.classification_report = report
        self.backend = backend  # TODO:Can be used for sheet recorder
        self.forward = forward

    # def before_init(self):
    #     self.create_sheet_column("latest_confusion_matrix", "Latest Confusion Matrix")
    #     self.create_sheet_column("kappa_score", "Kappa Score")
    #     self.create_sheet_column("accuracy", "Accuracy")

    def after_init(self, *args, **kwargs):
        self.recorder = self.miner.plugins.get(self.backend)

    def before_val_epoch_start(self, epoch, **ignore):
        # TODO:Recommended to attach these attributes to Miner Obj, for combination with Other Plugins
        self.predicts = []
        self.label = []

    def after_val_iteration_ended(self, predicts, data, **ignore):
        raw_output = predicts.detach().cpu().numpy()
        if self.forward:
            predicts, label = self.forward(raw_output, data)
        else:
            predicts = np.argmax(raw_output, axis=1)  # Batch first
            label = data[1].cpu().numpy()  # label
        self.predicts.append(predicts)
        self.label.append(label)

    def after_val_epoch_end(self, val_loss, **ignore):
        predicts = np.concatenate(self.predicts)
        label = np.concatenate(self.label)
        if self.accuracy:
            accuracy = (predicts == label).sum() / len(predicts)
            self.logger.info(f"Val Accuracy:{accuracy}")
            self.recorder.scalar("Val/Accuracy", accuracy)

        if self.confusion_matrix:
            matrix = confusion_matrix(label, predicts)
            df_cm = pd.DataFrame(matrix)
            svm = sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt=".3g")
            figure = svm.get_figure()
            if val_loss < self.miner.lowest_val_loss:
                # Solution to store Matplotlib Figure in TensorBoardX
                # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
                with io.BytesIO() as buff:
                    figure.savefig(buff, format='raw')
                    buff.seek(0)
                    data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
                w, h = figure.canvas.get_width_height()
                im = data.reshape((int(h), int(w), -1))
                im = im.transpose((2, 0, 1))  # Change to CHW
                self.recorder.figure("Val/ConfusionMatrix", im)
            plt.close(figure)

        if self.kappa_score:
            kappa = cohen_kappa_score(label, predicts, weights="quadratic")
            self.recorder.scalar("Val/KappaScore", kappa)

        if self.classification_report:
            # TODO:Design a better way to output or reord classification report
            self.logger.info(classification_report(label, predicts))
