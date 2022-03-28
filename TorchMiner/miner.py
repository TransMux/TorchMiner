from pathlib import Path
from typing import List

import torch
import tqdm
from TorchMiner import BasePlugin
from torch.optim import Optimizer

from TorchMiner.Logger import ColoredLogger
from TorchMiner.plugins import PluginManager
from . import utils
from TorchMiner.utils import find_resume_target, TorchMinerBackBone, TorchMinerMetrics


class Miner(object):
    def __init__(
            self,
            backbone: TorchMinerBackBone,
            plugins: List[BasePlugin],
    ):
        """

        :param backbone:
        :param plugins:
        """
        self.backbone: TorchMinerBackBone = backbone
        self.metrics = TorchMinerMetrics()
        # After read in
        self._create_dirs()
        # --- Init Plugin ---
        self.plugins = PluginManager(self, [backbone, self.metrics] + plugins)
        self.logger = self.backbone.get_logger("Miner")
        # --- Before Init ---
        self.plugins.call("before_init")
        self.backbone.prepare(self)
        self.metrics.prepare(self)
        # --- After Init ---
        self.plugins.call("after_init")

    # else:

    def train(self):
        """
        Core Function:start to train the model
        :return:
        """

        while True:
            # TODO: can we make a shortcut for metrics?
            self.metrics.current_epoch += 1
            self.plugins.call("before_train_epoch_start")
            self.settings.model.train()  # Set Train Mode
            self.metrics.train_iters = len(self.settings.train_dataloader)  # For future change during training
            self.metrics.total_train_loss = 0
            self.logger.info(f"start to train epoch {self.metrics.current_epoch}")
            t = self.settings.tqdm(self.settings.train_dataloader)

            for index, data in enumerate(t):
                self.metrics.data = data
                self.metrics.data_index = index

                self.plugins.call("before_train_iteration_start")
                self.metrics.train_loss = self._run_train_iteration(data)
                t.set_postfix({"train loss": self.metrics.train_loss})

                self.metrics["total_train_loss"] += self.metrics["train_loss"]

                self.plugins.call("after_train_iteration_end")

                # TODO: simplify amp procedure
                if int((index + 1) % self.accumulated_iter) == 0:
                    if self.amp and self.amp_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    if self.amp and self.amp_scaler:
                        self.optimizer.zero_grad()
                    else:
                        self.optimizer.zero_grad(set_to_none=True)
            # DataLoader End
            if self.amp and self.amp_scaler:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)

            self.metrics['total_train_loss'] = self.metrics["total_train_loss"] / self.metrics["train_iters"]
            self.logger.info(
                f"training of epoch {self.metrics['current_epoch']} finished, "
                f"loss is {self.metrics['total_train_loss']}"
            )

            self.plugins.call("after_train_epoch_end")

            # Begin eval
            if not self.metrics["current_epoch"] % self.eval_epoch and self.val_dataloader:
                self.plugins.call(
                    "before_val_epoch_start",
                    epoch=self.current_epoch,
                )
                self.model.eval()
                total_val_loss = 0
                val_iters = len(self.val_dataloader)
                with torch.no_grad():
                    self.logger.info(f"validate epoch {self.current_epoch}")
                    t = self.tqdm(self.val_dataloader)
                    for index, data in enumerate(t):
                        self.plugins.call(
                            "before_val_iteration_start",
                            data=data,
                            index=index,
                            val_iters=val_iters,
                            iteration=self.current_val_iteration,
                        )
                        predict, val_loss = self._run_val_iteration(data)
                        self.plugins.call(
                            "after_val_iteration_ended",
                            predicts=predict,
                            loss=val_loss,
                            data=data,
                            index=index,
                            val_iters=val_iters,
                            iteration=self.current_val_iteration,
                        )
                        t.set_postfix({"val loss": val_loss})
                        total_val_loss += val_loss

                total_val_loss = total_val_loss / val_iters
                self.logger.info(
                    f"validation of epoch {self.current_epoch} "
                    f"finished, loss is {total_val_loss}"
                )
                # persist model
                if total_val_loss < self.lowest_val_loss:
                    self.logger.info(
                        f"current val loss {total_val_loss} is lower than lowest {self.lowest_val_loss}, "
                        f"persist this model as best one"
                    )
                    self.lowest_val_loss = total_val_loss
                    self.persist("best")

                self.plugins.call(
                    "after_val_epoch_end",
                    val_loss=total_val_loss,
                    epoch=self.current_epoch,
                )
            self.persist("latest")
            # if self.drawer is not None:
            #     png_file = self.drawer.scalars(
            #         self.current_epoch,
            #         {"train": total_train_loss, "val": total_val_loss},
            #         "loss",
            #     )
            #     if png_file is not None:
            #         self.update_sheet(
            #             "loss", {"raw": png_file, "processor": "upload_image"}
            #         )

            if total_train_loss < self.lowest_train_loss:
                self.lowest_train_loss = total_train_loss

            if not self.current_epoch % self.persist_stride:
                self.persist("epoch_{}".format(self.current_epoch))

            if self.current_epoch >= self.max_epochs:
                self.plugins.call("before_quit")
                self.logger.warning("exceed max epochs, quit!")
                break

    def _run_train_iteration(self, data):
        self.status = "train"  # TODO:self.status Unused
        self.current_train_iteration += 1
        if self.amp and self.amp_scaler:
            with torch.cuda.amp.autocast():
                _, loss = self.forward(data)
                separate_loss = loss / self.accumulated_iter
            separate_loss = self.scaler.scale(separate_loss)
        else:
            _, loss = self.forward(data)
            separate_loss = loss / self.accumulated_iter  # TODO:实现accumulated_iter
        separate_loss.backward()
        loss = loss.detach().cpu().item()
        return loss

    def _run_val_iteration(self, data):
        self.status = "val"
        self.current_val_iteration += 1
        predict, loss = self.forward(data)
        loss = loss.detach().cpu().item()
        return predict, loss

    def forward(self, data):
        """
        A Function to calculate Network Forward results.
        The custom Forward_fn should return Network Output and Loss together.
        If Error Occurs in this Phase, Please use custom forward function
        :param data:
        :return:
        """
        predict = self.model(data[0].to(self.devices))
        loss = self.loss_func(predict, data[1].to(self.devices))
        return predict, loss

    def persist(self, name):
        """save the model to disk"""
        self.plugins.call("before_checkpoint_persisted", checkpoint_name=name)

        if isinstance(self.model, torch.nn.DataParallel):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        state = {
            "state_dict": model_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "train_iteration": self.current_train_iteration,
            "val_iteration": self.current_val_iteration,
            "lowest_train_loss": self.lowest_train_loss,
            "lowest_val_loss": self.lowest_val_loss,
        }

        state.update(self.plugins.save())

        if self.amp and self.amp_scaler:
            state["scaler"] = self.scaler.state_dict()

        modelpath = self._standard_model_path(name)
        torch.save(state, modelpath)
        self.logger.info(f"save checkpoint to {self._standard_model_path(name)}")
        self.plugins.call("after_checkpoint_persisted", modelpath=modelpath, checkpoint_name=name)

    def _standard_model_path(self, model_name):
        return self.models_dir / f"{model_name}.pth.tar"

    # # TODO: implement methods below
    # def graceful_stop(self):
    #     """stop train and exist after this epoch"""
    #     pass
    #
    # def save_and_stop(self):
    #     """save the model immediately and stop training"""
    #     pass

    def _create_dirs(self):
        """Create directories"""
        if self.alchemy_directory:
            utils.create_dir(self.alchemy_directory)
            utils.create_dir(self.alchemy_directory, self.experiment)
            utils.create_dir(self.alchemy_directory, self.experiment, "models")
