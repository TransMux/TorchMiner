# -*- coding:utf-8 -*-
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

from TorchMiner.plugin import Plugin

pool = ThreadPoolExecutor(1)
logger = logging.getLogger(__name__)


# TODO:Should I make this function a util function of the TorchMiner Package?
def _async(fn):
    @functools.wraps(fn)
    def _func(*args, **kwargs):
        def _inner(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.warn(f"error occured while handle task {e}")
                raise e

        return pool.submit(_inner, *args, **kwargs)

    return _func


class ColumnNotExistsError(Exception):
    pass


class Sheet(Plugin):
    def __init__(self):
        super().__init__()
        self.columns = []
        self.cached_row_data = {}

    def _create_experiment_row(self):
        """Create a row for this experiment"""
        raise NotImplementedError

    def _create_end_column_divider(self):
        raise NotImplementedError

    @_async
    def update(self, key, value):
        """Update value for a column"""
        if key not in self.columns:
            raise ColumnNotExistsError
        if not isinstance(value, dict):
            value = {"raw": value}
        self.cached_row_data[key] = value

    def flush(self):
        raise NotImplementedError

    def onready(self):
        """Called after all the columns are created"""
        raise NotImplementedError

    def create_column(self, key, title):
        """Create a column on the sheet."""
        self.columns.append(key)

    @property
    def experiment_row_name(self):
        return f"_row_{self.code}"

    @property
    def title_row_name(self):
        return "_row_title"

    @property
    def end_column_name(self):
        return "_column_end_divider"

    @property
    def banner_row_name(self):
        return "_row_banner"
