# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sn
from TorchMiner import BasePlugin
from TorchMiner.utils import figure2numpy
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score


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
            forward=None,
            labels=None
    ):
        super().__init__()
        self.accuracy = accuracy
        self.confusion_matrix = confusion
        self.kappa_score = kappa_score
        self.classification_report = report
        self.backend = backend  # TODO:Can be used for sheet recorder
        self.forward = forward
        self.labels = labels

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

    def predicts_and_labels(self, predicts, data):
        """
        Inherit this function to use custom classification function.
        :param predicts:
        :param data:
        :return:
        """
        predict = np.argmax(predicts, axis=1)  # Batch first
        label = data[1].cpu().numpy()  # label
        return predict, label

    def after_val_iteration_ended(self, predicts, data, **ignore):
        raw_output = predicts.detach().cpu().numpy()
        predict, label = self.predicts_and_labels(raw_output, data)
        self.predicts.append(predict)  # fix #17
        self.label.append(label)

    def after_val_epoch_end(self, val_loss, **ignore):
        try:
            predicts = np.concatenate(self.predicts)
            label = np.concatenate(self.label)
        except Exception as e:  # Fix #13
            self.logger.warning(f"{e} when concatenating val predictions, maybe only have one batch of val iteration.")
            predicts = self.predicts[0]
            label = self.label[0]

        if self.accuracy:
            accuracy = np.array(predicts == label).sum() / len(predicts)
            self.logger.info(f"Val Accuracy:{accuracy}")
            self.recorder.scalar("Val/Accuracy", accuracy)

        if self.confusion_matrix:
            matrix = confusion_matrix(label, predicts, labels=self.labels)  # Now can pass string items
            df_cm = pd.DataFrame(matrix)
            svm = sn.heatmap(df_cm, annot=True, cmap="OrRd", fmt=".3g")
            figure = svm.get_figure()
            if val_loss < self.miner.lowest_val_loss:
                self.recorder.figure("Val/ConfusionMatrix", figure2numpy(figure))
            plt.close(figure)

        if self.kappa_score:
            kappa = cohen_kappa_score(label, predicts, weights="quadratic")
            self.recorder.scalar("Val/KappaScore", kappa)

        if self.classification_report:
            # TODO:Design a better way to output or record classification report
            print(classification_report(label, predicts))
