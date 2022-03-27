import io
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm

from TorchMiner.Logger import ColoredLogger
from torch.optim import Optimizer


def seed_everything(seed):
    """
    Fix the seed for generating random numbers.
    :param seed:
    :return:
    """
    import torch
    from numpy import random

    torch.manual_seed(seed)
    # may lead to bad performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dir(directory, *args):
    """Create directory"""
    import os

    current_dir = directory
    for dir_name in args:
        current_dir = os.path.join(current_dir, dir_name)
    if not os.path.isdir(current_dir):
        os.mkdir(current_dir)


def figure2numpy(fig):
    # Solution to store Matplotlib Figure in TensorBoardX
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))  # Change to CHW
    return im


def find_resume_target(path: Path, index):
    """

    :param path: /
        Miner experiment path
    :param index: /
        True: Accept resume auto find result (Promise,default) Only best or latest
        string/Path: Will use the given checkpoint.
        int: Choose this epoch in auto find path.
    :return:
    """
    if index is True:
        search_paths = [
            path / "best.pth.tar",
            path / "latest.pth.tar",
        ]
    else:
        index = str(index)
        if Path(index).is_file():
            return Path(index)

        if (path / Path(index)).is_file():
            return path / Path(index)
        # The checkpoint is not given

        search_paths = [
            path,
            path / index,
            path / f"epoch_{index}.pth.tar",
            path / f"{index}.pth.tar",
            *path.glob("*.pth.tar"),
        ]

    for path in search_paths:
        if path.is_file():
            return path
    print(f"Tried to find Checkpoint in f{search_paths} but failed.")
    return None


class TorchMinerSettings:
    def __init__(
            self,
            alchemy_directory,
            experiment,
            model,
            optimizer,
            loss_func,
            train_dataloader=None,
            val_dataloader=None,
            resume=True,
            eval_epoch=1,
            persist_epoch=1,
            gpu=True,
            max_epochs=9999999,
            in_notebook=False,
            accumulated_iter=1,
            ignore_optimizer_resume=False,
            amp=False,
            amp_scaler=True,
    ):
        """
        Core Of TorchMiner
        :param alchemy_directory: The directory which TorchMiner will use to Store Everything in
        :param torch.nn.Module model: Target
        :param torch.optim.Optimizer: One should promise that Optimizer is inited on same device or
         a function that accepts model and returns an Optimizer, and TorchMiner will create the optimizer from it
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

        :param max_epochs:
        :param in_notebook:
        :param accumulated_iter:
        :param ignore_optimizer_resume:
        :param amp:
        :param amp_scaler:
        """
        self.miner = None
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Paths
        self.alchemy_directory = Path(alchemy_directory)  # working dir
        self.experiment = experiment
        self.experiment_dir = alchemy_directory / experiment
        self.models_dir = alchemy_directory / experiment / "models"
        # Model
        self.model = model.to(self.devices)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer if isinstance(optimizer, Optimizer) else optimizer(model)
        self.loss_func = loss_func.to(self.devices)
        self.amp = amp
        self.amp_scaler = amp_scaler
        self.scaler = None
        # setting
        self.resume = resume
        self.eval_epoch = eval_epoch
        self.persist_stride = persist_epoch
        self.train_only = False if val_dataloader else True
        self.accumulated_iter = float(accumulated_iter)
        self.gpu = gpu
        self.in_notebook = in_notebook
        self.ignore_optimizer_resume = ignore_optimizer_resume
        self.max_epochs = max_epochs
        if amp and amp_scaler:
            self.scaler = torch.cuda.amp.GradScaler()
        # Others
        self.logger_prototype = ColoredLogger
        self.tqdm = tqdm.tqdm
        self.logger = None
        # if self.train_only:
        #     self.logger.info("Running in Train Only Mode")

    def get_logger(self, name) -> logging.Logger:
        return self.logger_prototype(name)

    def prepare(self, miner):
        self.miner = miner
        self.logger = self.get_logger("Setting")
        self._parallel_model()

    def _parallel_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            # TODO: Cut GPU option, decide it by TorchMiner
            raise Exception(
                "Don't parallel the model yourself, instead, if the "
                "`gpu` option is true(default), TorchMiner will do this for you."
            )
        # TODO:统一 miner.gpu 和 miner.device 的设置
        # TODO:探索模型平行的原理，如何完成，数据集需要吗
        if self.gpu:
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                self.logger.warning("no GPU detected, will train on CPU.")
            else:
                self.logger.info(f"found {gpu_count} GPUs, will use all of them to train")
                devices = list(range(gpu_count))
                self.model.cuda()
                # TODO: 把Cuda移出去
                self.model = torch.nn.DataParallel(self.model, devices)


class TorchMinerMetrics:
    def __init__(self):
        self.miner = None
        self.logger = None

        self.lowest_train_loss = float("inf")
        self.lowest_val_loss = float("inf")
        self.current_epoch = 0
        self.current_train_iteration = 0
        self.current_val_iteration = 0
        #     In Epoch metrics
        self.train_iters = 0
        self.total_train_loss = 0
        self.data = None
        self.data_index = 0
        self.train_loss = 0

    def prepare(self, miner):
        self.logger = self.miner.settings.get_logger("Metrics")
        self.miner = miner
        self._resume()

    def _resume(self):
        setting: TorchMinerSettings = self.miner.settings
        check_point = find_resume_target(setting.models_dir, setting.resume)
        if check_point:
            # === Start Resume Procedure ===
            self.logger.info(f"Start to load checkpoint {check_point}")
            # TODO:After Loading Checkpoint, output basic information
            checkpoint = torch.load(check_point)
            # load model state
            try:
                setting.model.load_state_dict(checkpoint["state_dict"], strict=True)
            except Exception as e:
                self.logger.warning(e)
                self.logger.critical(
                    f"load checkpoint failed, the state in the "
                    "checkpoint is not matched with the model, "
                    "try to reload checkpoint with unstrict mode"
                )
                # UnStrict Mode
                setting.model.load_state_dict(checkpoint["state_dict"], strict=False)

            # load optimizer state
            if "optimizer" in checkpoint and not setting.ignore_optimizer_resume:
                try:
                    setting.optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception as e:
                    self.logger.warning(e)
                    self.logger.critical(
                        f"load optimizer state failed, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )
            # Read Train Process From Resumed Data
            self.current_epoch = checkpoint.get("epoch", 0)
            self.current_train_iteration = checkpoint.get("train_iteration", 0)
            self.current_val_iteration = checkpoint.get("val_iteration", 0)
            self.lowest_train_loss = checkpoint.get("lowest_train_loss", 9999)
            self.lowest_val_loss = checkpoint.get("lowest_val_loss", 9999)

            # load scaler state
            if setting.amp and setting.amp_scaler:
                try:
                    setting.scaler.load_state_dict(checkpoint["scaler"])
                except Exception as e:
                    self.logger.warning(
                        f"load scaler state failed with {e}, will skip this error and continue, "
                        "stop the process if it is not expected"
                    )

            self.miner.plugins.load(checkpoint)

            self.logger.info(f"Checkpoint {checkpoint} Successfully Loaded")
        else:
            self.logger.warning("Could not find checkpoint to resume, " "train from scratch")
