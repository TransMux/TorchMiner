# -*- coding:utf-8 -*-
import time

import tqdm
from IPython.core.display import HTML, display

from TorchMiner import BasePlugin


class JupyterLogger(BasePlugin):
    config = {
        "info": ["💬", "#6f818a"],
        "success": ["✅", "#7cb305"],  # TODO:How to implement success method
        "error": ["❌", "#f3715c"],
        "critical": ["❗", "#cf1322"],
        "warning": ["⚠️", "#d46b08"],
    }

    def __init__(self, name):
        super(JupyterLogger, self).__init__()
        self.name = name
        # self.log_dir = ""
        # TODO:Do you need to redirect the original log into a file when using Jupyter Logger?

    def prepare(self, miner, *args, **kwargs):
        super(JupyterLogger, self).prepare(miner)
        if not miner.in_notebook:
            self.logger.critical("Miner.in_notebook was set to False, but JupyterLogger will still patch on it.")
        self.miner.logger_prototype = JupyterLogger

    def _output(self, message, mode: str):
        display(
            HTML(
                f'<div style="font-size: 12px; color: {self.config[mode][1]}">'
                f'⏰ {time.strftime("%b %d - %H:%M:%S")} >>> '
                f"{self.config[mode][0]} [{self.name}]{message}"
                "</div>"
            )
        )

    def debug(self, message):
        raise NotImplementedError("JupyterLogger does Not Support 'debug' Logging.")

    def info(self, message):
        self._output(message, "info")

    def warning(self, message):
        self._output(message, "warning")

    def error(self, message):
        self._output(message, "error")

    def critical(self, message):
        self._output(message, "critical")

    def before_epoch_start(self, *args, **kwargs):
        display(
            HTML(
                '<div style="display: flex; justify-content: center;">'
                f'<h3 style="color: #7cb305; border-bottom: 4px dashed #91d5ff; padding-bottom: 6px;">Epoch {kwargs["epoch"]}</h3>'
                "</div>"
            )
        )


class JupyterTqdm(BasePlugin):
    def prepare(self, miner, *args, **kwargs):
        if not miner.in_notebook:
            self.logger.critical("Miner.in_notebook was set to False, but JupyterTqdm will still patch on it.")
        self.miner.tqdm = tqdm.notebook.tqdm
