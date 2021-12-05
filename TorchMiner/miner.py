import os
from pathlib import Path

import torch
import tqdm

from TorchMiner.Logger import ColoredLogger
from TorchMiner.plugins import PluginManager
from . import utils


class Miner(object):
    def __init__(
            self,
            alchemistic_directory,
            model,
            optimizer,
            loss_func,
            experiment,
            train_dataloader=None,
            val_dataloader=None,
            resume=True,
            eval_epoch=1,
            persist_epoch=1,
            gpu=True,
            max_epochs=9999999,
            in_notebook=False,
            plugins=None,
            accumulated_iter=1,
            ignore_optimizer_resume=False,
            forward=None,
            amp=False,
            amp_scaler=True,
    ):
        """
        Core Of TorchMiner
        :param alchemistic_directory: The directory which TorchMiner will use to Store Everything in
        :param torch.nn.Module model: Target
        :param torch.optim.Optimizer optimizer:
        :param loss_func: A function to compute Loss
            A Special Function, the function receives 2 variable:
            * Miner: The Miner Object
            * Data: The Batch data yield by the loader
            return Value should be a float number of the loss.
        :param string experiment: Experiment Name
        :param torch.utils.data.DataLoader train_dataloader:

            --- Optional ---
        :param torch.utils.data.DataLoader val_dataloader: Default None. If None, skip Validation
        :param bool resume: Default True.
        :param int eval_epoch: Default 1. Validate every 'eval_epoch'
        :param int persist_epoch: Default 1. Save model every 'persist_epoch'
        :param gpu:
        :param plugins:
            The Differences between Hooks and Plugins:
                Hooks are Functions,They receive Miner and Payloads
                 - Each Hook can load One Function Each Time
                Plugins are Classes succeed to `TorchMiner.Plugin`
                 - Receives Miner when Plugin Inits
                 - Receives only payload when Hooks in Plugin was called
                 - One can use many plugins in a miner
        :param max_epochs:
        :param in_notebook:
        :param accumulated_iter:
        :param ignore_optimizer_resume:
        :param forward:
        :param amp:
        :param amp_scaler:
        """
        self.alchemistic_directory = alchemistic_directory  # working dir
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.experiment = experiment
        self.logger_prototype = ColoredLogger
        self.val_dataloader = val_dataloader
        self.gpu = gpu
        self.in_notebook = in_notebook
        self.ignore_optimizer_resume = ignore_optimizer_resume
        self._create_dirs()
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_dir = os.path.join(alchemistic_directory, self.experiment)
        self.models_dir = os.path.join(alchemistic_directory, self.experiment, "models")
        self.accumulated_iter = float(accumulated_iter)
        self.loss_func = loss_func
        self.resume = resume
        self.eval_epoch = eval_epoch
        self.persist_stride = persist_epoch
        self.lowest_train_loss = float("inf")
        self.lowest_val_loss = float("inf")
        self.current_epoch = 0
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        self.max_epochs = max_epochs
        if forward:
            self.logger.warning("Forward Function is Discarded in v0.4.2, Please inherit _forward function")
        self.forward_fn = forward
        self.amp = amp
        self.amp_scaler = amp_scaler
        if self.amp and self.amp_scaler:
            self.scaler = torch.cuda.amp.GradScaler()
        self.tqdm = tqdm.tqdm

        # --- Init Plugin ---
        self.plugins = PluginManager(self, plugins)
        self.logger = self.get_logger("Miner")
        # --- Before Init ---
        self.status = "init"
        self.plugins.call("before_init")
        self._init_model()
        # --- After Init ---
        self.plugins.call("after_init")

    @staticmethod
    def get_logger(name):
        return ColoredLogger(name)

    def _init_model(self):
        """resume from some checkpoint"""
        if isinstance(self.model, torch.nn.DataParallel):
            raise Exception(
                "Don't parallel the model yourself, instead, if the "
                "`gpu` option is true(default), TorchMiner will do this for you."
            )

        # TODO:简化Resume from pretrained的流程 添加对自定义路径的支持
        if self.resume is True:  # Find by TorchMiner
            # resume from the newest model
            if self._search_model_file("latest"):
                checkpoint_path = self._search_model_file("latest")
            else:
                checkpoint_path = None
                self.logger.warning("Could not find checkpoint to resume, " "train from scratch")
        elif isinstance(self.resume, str):  # specify model file name
            checkpoint_path = self._search_model_file(self.resume)
        elif isinstance(self.resume, int):  # specify train epoch
            checkpoint_path = self._search_model_file(self.resume)
        else:
            checkpoint_path = None

        if self.resume is not True and self.resume and checkpoint_path is None:
            # user has specified a none existed model, should raise a error
            raise Exception(f"Could not find model {self.resume}")

        if checkpoint_path is not None:
            # TODO:After Loading Checkpoint, output basic information
            self.logger.info(f"Start to load checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            # Read Train Process From Resumed Data
            self.current_epoch = checkpoint.get("epoch", 0)
            self.current_train_iteration = checkpoint.get("train_iteration", 0)
            self.current_val_iteration = checkpoint.get("val_iteration", 0)
            self.lowest_train_loss = checkpoint.get("lowest_train_loss", 9999)
            self.lowest_val_loss = checkpoint.get("lowest_val_loss", 9999)

            # load model state
            try:
                self.model.load_state_dict(checkpoint["state_dict"], strict=True)
            except Exception as e:
                self.logger.warning(
                    f"load checkpoint failed with {e}, the state in the "
                    "checkpoint is not matched with the model, "
                    "try to reload checkpoint with unstrict mode"
                )
                # UnStrict Mode
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            # load optimizer state
            if "optimizer" in checkpoint and not self.ignore_optimizer_resume:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    self.logger.warning(
                        f"load optimizer state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )

            # load scaler state
            if self.amp and self.amp_scaler:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    self.logger.warning(
                        f"load scaler state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )

            self.plugins.load(checkpoint)

            self.logger.info(f"Checkpoint {checkpoint_path} Successfully Loaded")
        self.model = self._parallel_model(self.model)

    def _parallel_model(self, model):
        # TODO:统一 miner.gpu 和 miner.device 的设置
        # TODO:探索模型平行的原理，如何完成，数据集需要吗
        if self.gpu:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                self.logger.warning("no GPU detected, will train on CPU.")
            else:
                self.logger.info(f"found {gpu_count} GPUs, will use all of them to train")
                devices = list(range(gpu_count))
                model.cuda()
                model = torch.nn.DataParallel(model, devices)
        return model

    def train(self):
        """
        Core Function:start to train the model
        :return:
        """
        while True:
            self.current_epoch += 1
            self.plugins.call("before_train_epoch_start", epoch=self.current_epoch)
            self.model.train()  # Set Train Mode
            train_iters = len(self.train_dataloader)  # For future change during training

            total_train_loss = 0
            self.logger.info(f"start to train epoch {self.current_epoch}")
            t = self.tqdm(self.train_dataloader)
            for index, data in enumerate(t):
                self.plugins.call(
                    "before_train_iteration_start",
                    data=data,
                    index=index,
                    total_iters=train_iters,
                    iteration=self.current_train_iteration,
                )
                train_loss = self._run_train_iteration(data)
                self.plugins.call(
                    "after_train_iteration_end",
                    loss=train_loss,
                    data=data,
                    index=index,
                    total_iters=train_iters,
                    iteration=self.current_train_iteration,
                )
                t.set_postfix({"train loss": train_loss})
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
                total_train_loss += train_loss
            # DataLoader End
            if self.amp and self.amp_scaler:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)

            total_train_loss = total_train_loss / train_iters
            self.logger.info(
                f"training of epoch {self.current_epoch} finished, "
                f"loss is {total_train_loss}"
            )

            self.plugins.call(
                "after_train_epoch_end",
                train_loss=total_train_loss,
                epoch=self.current_epoch,
            )

            # Begin eval
            if not self.current_epoch % self.eval_epoch and self.val_dataloader:
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
                    val_loss=total_train_loss,
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
                _, loss = self._forward(data)
                separate_loss = loss / self.accumulated_iter
            separate_loss = self.scaler.scale(separate_loss)
        else:
            _, loss = self._forward(data)
            separate_loss = loss / self.accumulated_iter  # TODO:实现accumulated_iter
        separate_loss.backward()
        loss = loss.detach().cpu().item()
        return loss

    def _run_val_iteration(self, data):
        self.status = "val"
        self.current_val_iteration += 1
        predict, loss = self._forward(data)
        loss = loss.detach().cpu().item()
        return predict, loss

    def before_forward(self, data):
        """
        Inherit this function to pre-process the data from the input model
        :param data:
        :return:
        """
        return data

    def _forward(self, data):
        """
        A Function to calculate Network Forward results.
        The custom Forward_fn should return Network Output and Loss together.
        :param data:
        :return:
        """
        if self.forward_fn:
            return self.forward_fn(self, data)
        else:
            predict = self.model(self.before_forward(data[0]).to(self.devices))
            loss = self.loss_func(predict, data[1].to(self.devices))
            return self.after_forward(predict, loss)

    def after_forward(self, predict, loss):
        """
        Inherit this function to view or change the prediction.
        :param predict:
        :param loss:
        :return:
        """
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
        return os.path.join(self.models_dir, f"{model_name}.pth.tar")

    def _search_model_file(self, model_name):
        model_name_path = Path(str(model_name))
        models_dir_path = Path(self.models_dir)

        search_paths = [
            model_name_path,
            models_dir_path / model_name_path,
            models_dir_path / f"{model_name}.pth.tar",
            models_dir_path / f"epoch_{model_name}.pth.tar",
        ]

        for path in search_paths:
            if path.is_file():
                return path.resolve()

        return None

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
        if self.alchemistic_directory:
            utils.create_dir(self.alchemistic_directory)
            utils.create_dir(self.alchemistic_directory, self.experiment)
            utils.create_dir(self.alchemistic_directory, self.experiment, "models")
