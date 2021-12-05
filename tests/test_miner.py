import unittest

from TorchMiner import Miner


class Plugins(unittest.TestCase):
    def test_empty_plugin_list(self):  # (#11)
        miner = Miner("", None, None, None, "", plugins=None)
        self.assertEqual(miner.plugins.plugins, [])
