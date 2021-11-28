from TorchMiner.Logger import ColoredLogger


class BasePlugin:
    def __init__(self):
        self.name = self.__class__.__name__
        self.logger = ColoredLogger(self.name)
        self.miner = None

    def prepare(self, miner, *args, **kwargs):
        # Can be used for Monkey Patch Operations
        self.miner = miner

    # Plugin Data Begin
    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}

    # Plugin Data End

    def set_miner(self, miner):
        """
        This function is deprecated.
        :param miner:
        :return:
        """
        self.miner = miner

    # Hook Functions Begin
    def before_init(self, *args, **kwargs):
        pass

    def after_init(self, *args, **kwargs):
        pass

    def before_epoch_start(self, *args, **kwargs):
        pass

    def before_quit(self, *args, **kwargs):
        pass

    def after_epoch_end(self, *args, **kwargs):
        pass

    def before_train_iteration_start(self, *args, **kwargs):
        pass

    def after_train_iteration_end(self, *args, **kwargs):
        pass

    def before_val_iteration_start(self, *args, **kwargs):
        pass

    def after_val_iteration_ended(self, *args, **kwargs):
        pass

    def before_checkpoint_persisted(self, *args, **kwargs):
        pass

    def after_checkpoint_persisted(self, *args, **kwargs):
        pass
    # Hook Functions end

    # def print_txt(self, printable, name):
    #     with open(self.plugin_file(f"{name}.txt"), "a") as f:
    #         print(
    #             f"================ Epoch {self.current_epoch} ================\n",
    #             file=f,
    #         )
    #         print(printable, file=f)
    #         print("\n\n", file=f)

    # @property
    # def plugin_dir(self):
    #     if hasattr(self, "_plugin_dir"):
    #         return getattr(self, "_plugin_dir")
    #
    #     plugin_dir = os.path.join(self.code_dir, self.__class__.__name__)
    #     try:
    #         os.mkdir(plugin_dir)
    #     except FileExistsError:
    #         pass
    #     self._plugin_dir = plugin_dir
    #     return self._plugin_dir
    #
    # def plugin_file(self, name):
    #     return os.path.join(self.plugin_dir, name)

    # def create_sheet_column(self, key, name):
    #     self.miner.create_sheet_column(f"{self.prefix}{key}", f"{self.prefix}{name}")
    #
    # def update_sheet(self, key, value):
    #     self.miner.update_sheet(f"{self.prefix}{key}", value)
    #
    # def scalars(self, values, graph):
    #     return self.miner.drawer.scalars(
    #         self.miner.current_epoch, values, f"{self.prefix}{graph}"
    #     )
