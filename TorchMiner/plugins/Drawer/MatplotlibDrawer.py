# -*- coding:utf-8 -*-
import _pickle as pickle
import os

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from TorchMiner.plugins.Drawer import BasicDrawer


class MatplotlibDrawer(BasicDrawer):
    def __init__(self, miner):
        super(MatplotlibDrawer, self).__init__(miner)
        self.graph_dir = os.path.join(
            self.miner.alchemistic_directory, self.miner.experiment, "graphs"
        )
        self.data_file = os.path.join(self.graph_dir, ".graphs.pickle")
        self.colors = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        if not os.path.isdir(self.graph_dir):
            os.mkdir(self.graph_dir)
        # Load from previous
        # TODO: Need to add into state_dict?
        if os.path.isfile(self.data_file):
            with open(self.data_file, "rb") as f:
                self.graph_data = pickle.load(f)

    def _update_state(self, x, values, graph):
        if graph not in self.state or not isinstance(self.state[graph], dict):
            self.state[graph] = {}
        for key in values:
            if key not in self.state[graph]:
                self.state[graph][key] = {}
            self.state[graph][key][x] = values[key]
        with open(self.data_file, "wb") as f:
            pickle.dump(self.state, f)

    def _save_png(self, graph):
        png_file = os.path.join(self.graph_dir, graph + ".png")
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        for index, curve in enumerate(self.state[graph]):
            ax.plot(
                *zip(*sorted(self.state[graph][curve].items())),
                label=curve,
                color=self.colors[index]
            )

        ax.legend(loc="upper left")
        fig.savefig(png_file, facecolor="#F0FFFC")
        return png_file

    def scalars(self, x, values, graph):
        """Add a scalar on a graph

        Args:
            value (dict):
                scalars to put on the graph
            graph (string):
                graph name
        """
        self._update_state(x, values, graph)
        return self._save_png(graph)
