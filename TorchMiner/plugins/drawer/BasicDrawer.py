# -*- coding:utf-8 -*-
from TorchMiner.plugins import Plugin


class Drawer(Plugin):
    """To vistualize everything in training process"""

    def __init__(self, miner, state=None):
        """Constructor

        Args:
            miner (Miner):
                Miner instance
            graph (state, optional):
                Defaults to None. Since we could draw multiple graph during
                training, the state records the current position of each graph
                The keys are the name of the graphs and values are current
                positions.
        """
        super().__init__()
        self.step_file = os.path.join(
            miner.alchemistic_directory, miner.experiment, ".drawer_step"
        )
        self.miner = miner

        if state is None:
            self.state = {}
        else:
            self.state = state

    def scalars(self, x, value, graph):
        """Plot different scalars on a graph

        Args:
            value (dict):
                scalar to plot
            graph (string):
                graph name
        """
        raise NotImplementedError()

    def scalar(self, x, value, graph):
        """Plot one scalar on a graph

        Args:
            value (float):
                scalar to plot
            graph (string):
                graph name
        """
        self.scalars(x, {graph: value}, graph)

    def get_state(self):
        """Return current state(counter) of the Drawer"""
        return self.state

    def set_state(self, state):
        """Set current state(counter) to state"""
        self.state = state
