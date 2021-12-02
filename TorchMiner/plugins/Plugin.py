class BasePlugin:
    def __init__(self):
        self.name = self.__class__.__name__
        self.miner = None
        self.logger = None  # Plugin Logger will be inited in prepare stage

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
        self.logger = self.miner.get_logger(self.name)

    def after_init(self, *args, **kwargs):
        pass

    def before_train_epoch_start(self, *args, **kwargs):
        pass

    def before_train_iteration_start(self, *args, **kwargs):
        pass

    def after_train_iteration_end(self, *args, **kwargs):
        pass

    def after_train_epoch_end(self, *args, **kwargs):
        pass

    def before_val_epoch_start(self, *args, **kwargs):
        pass

    def before_val_iteration_start(self, *args, **kwargs):
        pass

    def after_val_iteration_ended(self, *args, **kwargs):
        pass

    def after_val_epoch_end(self, *args, **kwargs):
        pass

    def before_checkpoint_persisted(self, *args, **kwargs):
        pass

    def after_checkpoint_persisted(self, *args, **kwargs):
        pass

    def before_quit(self, *args, **kwargs):
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


class PluginManager:
    def __init__(self, miner, plugins):
        self.miner = miner
        self.logger = miner.get_logger("PluginManager")
        if plugins:
            self.plugins = plugins
            self.plugin_names = [i.__class__.name for i in self.plugins]
        else:
            self.plugins = []
        self.check_requirements()
        self.prepare()

    def check_requirements(self):
        """
        Check Requirements of each Plugins
         - Requirements are set in restrict string
        :return:
        """
        for p in self.plugins:
            self.logger.debug(f"Checking Requirements of {p}.")
            for r in p:
                if r not in self.plugin_names:
                    self.logger.error(f"Requirement {r} of {p} is not Find.")
            else:
                self.logger.info("Successfully Passed Plugin Requirements Check with no Errors.")
        # TODO: Try to import Unmet needs

    def status(self):
        """
        Print the Status of all registered Plugins
        :return:
        """
        self.logger.info(f"Registered Plugins:{self.plugins}")

    def register(self):
        """
        Register Plugins from a given module list
        :return:
        """

    def call(self, hook, **payload):
        """
        Call Hook Functions
        :return:
        """
        for plugin in self.plugins:
            getattr(plugin, hook)(miner=self, **payload)  # !!! `miner=self` is totally different with just `self`

    def prepare(self):
        """
        prepare a given Plugin
        :return:
        """
        for p in self.plugins:
            p.prepare(self.miner)

    def load(self, checkpoint):
        # load plugin states
        for plugin in self.plugins:
            key = f"__plugin.{plugin.__class__.__name__}__"
            plugin.load_state_dict(checkpoint.get(key, {}))

    def save(self):
        temp = {}
        for p in self.plugins:
            temp[f"__plugin.{p.__class__.__name__}__"] = p.state_dict()
        return temp
