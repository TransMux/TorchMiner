import logging
import math
import os
# import time
# from datetime import datetime
from pathlib import Path

import torch
import tqdm
# from IPython.core.display import HTML, display
#
# from . import drawers
from . import utils


class Miner(object):
    def __init__(
            self,
            alchemistic_directory,
            model,
            optimizer,
            loss_func,
            experiment="geass",
            train_dataloader=None,
            val_dataloader=None,
            resume=True,
            eval_epoch=1,
            persist_epoch=1,
            gpu=True,
            # drawer="matplotlib",
            max_epochs=9999999,
            statable=None,
            logging_format=None,
            in_notebook=False,
            plugins=None,
            logger=None,
            sheet=None,
            accumulated_iter=1,
            ignore_optimizer_resume=False,
            forward=None,
            verbose=False,
            amp=False,
            amp_scaler=True,
    ):
        """
        Core Of TorchMiner
        :param alchemistic_directory: The directory which TorchMiner will use to Store Everything in
        :param torch.nn.Module model: Target
        :param torch.optim.Optimizer optimizer:
        :param loss_func: A function to compute Loss
            A Special Hook Function, the function receives 2 variable:
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
        :param statable:
        :param logging_format:
        :param in_notebook:
        :param logger:
        :param sheet:
        :param accumulated_iter:
        :param ignore_optimizer_resume:
        :param forward:
        :param verbose: # TODO:verbose是用来干嘛的
        :param amp:
        :param amp_scaler:
        """
        # --- Init Process Recorders ---
        if statable is None:
            statable = {}
        self.statable = statable  # TODO:what is statable
        # --- Init Plugin ---
        if plugins is None:
            plugins = []
        self.plugins = plugins
        for plugin in self.plugins:
            plugin.set_miner(self)

        self.alchemistic_directory = alchemistic_directory  # working dir
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.experiment = experiment

        self.val_dataloader = val_dataloader
        self.gpu = gpu
        self.logger = logger
        self.in_notebook = in_notebook
        self.ignore_optimizer_resume = ignore_optimizer_resume

        self._create_dirs()
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_dir = os.path.join(alchemistic_directory, self.experiment)
        self.models_dir = os.path.join(alchemistic_directory, self.experiment, "models")
        if self.logger is None:
            self._set_logging_config(alchemistic_directory, self.experiment, logging_format)
            self.logger = logging
        # self._create_drawer(drawer)
        self.accumulated_iter = float(accumulated_iter)

        self.loss_func = loss_func

        self.resume = resume
        self.eval_stride = eval_epoch
        self.persist_stride = persist_epoch
        self.lowest_train_loss = float("inf")
        self.lowest_val_loss = float("inf")
        self.current_epoch = 0
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        self.max_epochs = max_epochs
        self.forward_fn = forward
        self.verbose = verbose
        self.amp = amp

        self.amp_scaler = amp_scaler
        if self.amp and self.amp_scaler:
            self.scaler = torch.cuda.amp.GradScaler()

        self._set_tqdm(in_notebook)
        # --- Before Init ---
        self._call_plugins("before_init")
        self._check_statable()
        self._init_model()
        if self.sheet:
            self.sheet_progress = dict(epoch=0, train_percentage="0%", val_percentage="0%")
            self.last_flushed_at = 0
            self.sheet.onready()
            self.sheet.flush()
        self.status = "init"
        # --- After Init ---
        self._call_plugins("after_init")

    def _call_plugins(self, name, **payload):
        """
        Call Hook Functions
        :param name: Hook Name
        :param payload: extra prams in specific Stage
        :return:
        """
        for plugin in self.plugins:
            getattr(plugin, name)(self, **payload)

    def _check_statable(self):
        for name, statable in self.statable.items():
            if not (
                    hasattr(statable, "state_dict") and hasattr(statable, "load_state_dict")
            ):
                raise Exception(f"The {name} is not a statable object")

    def _set_tqdm(self, in_notebook):
        if in_notebook:
            self.tqdm = tqdm.notebook.tqdm
        else:
            self.tqdm = tqdm.tqdm

    # def _init_sheet(self):
    #     self.sheet.set_miner(self)
    #     self.sheet.reset_index()
    #     self.sheet.create_column("code", "Code")
    #     self.sheet.create_column("progress", "Progress")
    #     self.sheet.create_column("loss", "Loss")
    #     self.sheet.update("code", self.experiment)

    # def create_sheet_column(self, key, title):
    #     if self.sheet is None:
    #         return
    #     self.sheet.create_column(key, title)
    #
    # def update_sheet(self, key, value):
    #     if self.sheet is None:
    #         return
    #     self.sheet.update(key, value)

    def _set_logging_config(self, alchemistic_directory, experiment, logging_format):
        self.log_dir = os.path.join(alchemistic_directory, experiment)
        log_file = os.path.join(self.log_dir, "log.txt")
        logging_format = (
            logging_format
            if logging_format is not None
            else "%(levelname)s %(asctime)s %(message)s"  # Default
        )
        logging.basicConfig(
            filename=log_file,
            format=logging_format,
            datefmt="%m-%d %H:%M:%S",
            level=logging.INFO,
        )

    # def _create_drawer(self, drawer):
    #     if drawer == "tensorboard":
    #         self.drawer = drawers.TensorboardDrawer(self)
    #     elif drawer == "matplotlib":
    #         self.drawer = drawers.MatplotlibDrawer(self)
    #     else:
    #         self.drawer = drawer

    # def _notebook_output(self, message, _type="info"):
    #     type_config = {
    #         "info": ["💬", "#6f818a"],
    #         "success": ["✅", "#7cb305"],
    #         "error": ["❌", "#cf1322"],
    #         "warning": ["⚠️", "#d46b08"],
    #     }[_type]
    #     if self.in_notebook:
    #         display(
    #             HTML(
    #                 f'<div style="font-size: 12px; color: {type_config[1]}">'
    #                 f'⏰ {time.strftime("%b %d - %H:%M:%S")} >>> '
    #                 f"{type_config[0]} {message}"
    #                 "</div>"
    #             )
    #         )

    # def _notebook_divide(self, message):
    #     if self.in_notebook:
    #         display(
    #             HTML(
    #                 '<div style="display: flex; justify-content: center;">'
    #                 f'<h3 style="color: #7cb305; border-bottom: 4px dashed #91d5ff; padding-bottom: 6px;">{message}</h3>'
    #                 "</div>"
    #             )
    #         )

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
                msg = "Could not find checkpoint to resume, " "train from scratch"
                self._notify(msg, "warning")
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
            msg = f"Start to load checkpoint {checkpoint_path}"
            self._notify(msg)
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
                msg = (
                    f"load checkpoint failed with {e}, the state in the "
                    "checkpoint is not matched with the model, "
                    "try to reload checkpoint with unstrict mode"
                )
                self._notify(msg, "warning")
                # UnStrict Mode
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            # load optimizer state
            if "optimizer" in checkpoint and not self.ignore_optimizer_resume:
                try:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    msg = (
                        f"load optimizer state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )
                    self._notify(msg, "warning")

            # load drawer state
            # if (self.drawer is not None) and ("drawer_state" in checkpoint):
            #     self.drawer.set_state(checkpoint["drawer_state"])

            # load scaler state
            if self.amp and self.amp_scaler:
                try:
                    self.scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    msg = (
                        f"load scaler state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )
                    self._notify(msg, "warning")

            # load other statable state
            # if "statable" in checkpoint:
            #     for name, statable in self.statable.items():
            #         if name not in checkpoint["statable"]:
            #             continue
            #         statable.load_state_dict(checkpoint["statable"][name])

            # load plugin states
            for plugin in self.plugins:
                key = f"__plugin.{plugin.__class__.__name__}__"
                plugin.load_state_dict(checkpoint.get(key, {}))

            msg = f"Checkpoint {checkpoint_path} Successfully Loaded"
            self._notify(msg, "success")
        self.model = self._parallel_model(self.model)

    def _parallel_model(self, model):
        # TODO:统一 miner.gpu 和 miner.device 的设置
        # TODO:探索模型平行的原理，如何完成，数据集需要吗
        if self.gpu:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                self._notify("no GPU detected, will train on CPU.")
            else:
                self._notify(f"found {gpu_count} GPUs, will use all of them to train")
                devices = list(map(lambda x: f"cuda:{x}", range(gpu_count)))
                model.cuda()
                model = torch.nn.DataParallel(model, devices)
        return model

    def _notify(self, message, _type="info"):
        getattr(self.logger, "info" if _type == "success" else _type)(message)
        print("[Dev Debug] notify:", _type, message)

    def train(self):
        """
        Core Function:start to train the model
        :return:
        """
        while True:
            self.current_epoch += 1
            self._call_plugins("before_epoch_start", epoch=self.current_epoch)
            # self._notebook_divide(f"Epoch {self.current_epoch}")
            self.model.train()  # Set Train Mode
            train_iters = len(self.train_dataloader)

            total_train_loss = 0
            # percentage = 0
            total = len(self.train_dataloader)
            self._notify(f"start to train epoch {self.current_epoch}")
            # self._update_progress(
            #     force=True,
            #     epoch=self.current_epoch,
            #     train_percentage="0%",
            #     val_percentage="0%",
            # )
            t = self.tqdm(self.train_dataloader)
            for index, data in enumerate(t):
                train_loss = self._run_train_iteration(index, data, train_iters)
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
                current_percentage = math.ceil(index / total * 100)
                # self._update_progress(train_percentage=f"{current_percentage}%")
            # DataLoader End
            if self.amp and self.amp_scaler:
                self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad(set_to_none=True)
            # self._update_progress(force=True, train_percentage=f"{current_percentage}%")

            total_train_loss = total_train_loss / train_iters
            self._notify(
                f"training of epoch {self.current_epoch} finished, "
                f"loss is {total_train_loss}"
            )

            # Begin eval
            self.model.eval()
            total_val_loss = 0
            total = len(self.val_dataloader)
            if self.val_dataloader:
                val_iters = len(self.val_dataloader)
                with torch.no_grad:
                    self._notify(f"validate epoch {self.current_epoch}")
                    t = self.tqdm(self.val_dataloader)
                    for index, data in enumerate(t):
                        val_loss = self._run_val_iteration(index, data, val_iters)
                        t.set_postfix({"val loss": val_loss})
                        total_val_loss += val_loss
                        current_percentage = math.ceil(index / total * 100)
                    #     self._update_progress(val_percentage=f"{current_percentage}%")
                    # self._update_progress(
                    #     force=True, val_percentage=f"{current_percentage}%"
                    # )

                total_val_loss = total_val_loss / val_iters
                self._notify(
                    f"validation of epoch {self.current_epoch} "
                    f"finished, loss is {total_val_loss}"
                )
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

            if total_val_loss < self.lowest_val_loss:
                message = (
                    "current val loss {} is lower than lowest {}, "
                    "persist this model as best one".format(
                        total_val_loss, self.lowest_val_loss
                    )
                )
                self._notify(message, "success")
                self.lowest_val_loss = total_val_loss
                self.persist("best")
            else:
                self.persist("latest")

            # self._call_hook_func("before_persist_checkpoint")

            if not self.current_epoch % self.persist_stride:
                self.persist("epoch_{}".format(self.current_epoch))

            if self.current_epoch >= self.max_epochs:
                self._call_plugins("before_quit")
                self._notify("exceed max epochs, quit!")
                break

            # if self.sheet:
            #     self.sheet.flush()
            self._call_plugins(
                "after_epoch_end",
                train_loss=total_train_loss,
                val_loss=total_val_loss,
                epoch=self.current_epoch,
            )

    def _run_train_iteration(self, index, data, train_iters):
        self.status = "train"
        self.current_train_iteration += 1
        self._call_plugins(
            "before_train_iteration_start",
            data=data,
            index=index,
            total_iters=train_iters,
            iteration=self.current_train_iteration,
        )
        if self.amp and self.amp_scaler:
            with torch.cuda.amp.autocast():
                _, loss = self._forward(data)
                seperate_loss = loss / self.accumulated_iter
            seperate_loss = self.scaler.scale(seperate_loss)
        else:
            _, loss = self._forward(data)
            seperate_loss = loss / self.accumulated_iter
        seperate_loss.backward()
        loss = loss.detach().cpu().item()
        # if self.verbose:
        #     self.logger.info(
        #         "[train {}/{}/{}] loss {}".format(
        #             self.current_epoch, index, train_iters, loss
        #         )
        #     )

        self._call_plugins(
            "after_train_iteration_end",
            loss=loss,
            data=data,
            index=index,
            total_iters=train_iters,
            iteration=self.current_train_iteration,
        )
        return loss

    def _forward(self, data):
        if self.forward_fn:
            return self.forward_fn(self, data)
        else:
            predict = self.model(data[0].to(self.devices))
            loss = self.loss_func(predict, data[1].to(self.devices))
            return predict, loss

    def _run_val_iteration(self, index, data, val_iters):
        self.status = "val"
        self.current_val_iteration += 1
        self._call_plugins(
            "before_val_iteration_start",
            data=data,
            index=index,
            total_iters=val_iters,
            iteration=self.current_val_iteration,
        )
        predict, loss = self._forward(data)
        loss = loss.detach().cpu().item()
        # if self.verbose:
        #     self.logger.info(
        #         "[val {}/{}/{}] loss {}".format(
        #             self.current_epoch, index, val_iters, loss
        #         )
        #     )
        self._call_plugins(
            "after_val_iteration_ended",
            predicts=predict,
            loss=loss,
            data=data,
            index=index,
            total_iters=val_iters,
            iteration=self.current_val_iteration,
        )
        return loss

    def persist(self, name):
        """save the model to disk"""
        self._call_plugins("before_checkpoint_persisted")
        # if self.drawer is not None:
        #     drawer_state = self.drawer.get_state()
        # else:
        #     drawer_state = {}

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
            # "drawer_state": drawer_state,
            "statable": {},
        }

        for statable_name, statable in self.statable.items():
            state["statable"][statable_name] = statable.state_dict()

        for plugin in self.plugins:
            key = f"__plugin.{plugin.__class__.__name__}__"
            state[key] = plugin.state_dict()

        if self.amp and self.amp_scaler:
            state["scaler"] = self.scaler.state_dict()

        modelpath = self._standard_model_path(name)
        torch.save(state, modelpath)
        message = f"save checkpoint to {self._standard_model_path(name)}"
        self._notify(message)
        self._call_plugins("after_checkpoint_persisted", modelpath=modelpath)

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

    # TODO: implement methods below
    def graceful_stop(self):
        """stop train and exist after this epoch"""
        pass

    def save_and_stop(self):
        """save the model immediately and stop training"""
        pass

    def _create_dirs(self):
        """Create directories"""
        utils.create_dir("")
        utils.create_dir(self.experiment)
        utils.create_dir(self.experiment, "models")

#     def periodly_flush(self, force=False):
#         if self.sheet is None:
#             return
#         now = int(datetime.now().timestamp())
#         # flush every 10 seconds
#         if not force and now - self.last_flushed_at < 10:
#             return
#         self.sheet.flush()
#         self.last_flushed_at = now
#
#     def _update_progress(self, force=False, **kwargs):
#         if self.sheet is None:
#             return
#
#         self.sheet_progress.update(kwargs)
#         progress = f"""
#          epoch:  {self.sheet_progress.get('epoch')}
# train progress:  {self.sheet_progress.get('train_percentage')}
#   val progress:  {self.sheet_progress.get('val_percentage')}
# """
#         self.sheet.update("progress", progress)
#         self.periodly_flush(force)
