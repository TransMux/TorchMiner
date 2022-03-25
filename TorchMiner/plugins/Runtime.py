from TorchMiner import BasePlugin


class GarbageCollect(BasePlugin):
    def __init__(self):
        import gc
        self.gc = gc
        super(GarbageCollect, self).__init__()

    def after_train_epoch_end(self, *args, **kwargs):
        self._collect()

    def after_val_epoch_end(self, *args, **kwargs):
        self._collect()

    def _collect(self):
        self.logger.debug("Garbage Collected!")
        self.gc.collect()
