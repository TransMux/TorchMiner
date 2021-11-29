# -*- coding:utf-8 -*-
import functools
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from TorchMiner.plugins import BasePlugin

pool = ThreadPoolExecutor(1)


# TODO:Should I make this function a util function of the TorchMiner Package?
# TODO:Find a better way to accomplish async function
def _async(fn):
    @functools.wraps(fn)
    def _func(*args, **kwargs):
        def _inner(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print(e)
                raise e

        return pool.submit(_inner, *args, **kwargs)

    return _func


class ColumnNotExistsError(Exception):
    pass


class Sheet(BasePlugin):
    def __init__(self):
        super().__init__()
        self.columns = []
        self.cached_row_data = {}
        self.last_flushed_at = None

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

    def periodly_flush(self, force=False):
        now = int(datetime.now().timestamp())
        # flush every 10 seconds
        if not force and now - self.last_flushed_at < 10:
            return
        self.flush()
        self.last_flushed_at = now

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
