import unittest
from pathlib import Path

from TorchMiner.utils import find_resume_target


class Find_Resume_Target(unittest.TestCase):
    normal_model_path = Path("models")

    def test_auto_find(self):
        self.assertEqual(
            Path('models/best/best.pth.tar'),
            find_resume_target(self.normal_model_path / "best", True)
        )
        self.assertEqual(
            None,
            find_resume_target(self.normal_model_path / "epoch", True)
        )
        self.assertEqual(
            Path('models/latest/latest.pth.tar'),
            find_resume_target(self.normal_model_path / "latest", True)
        )

    def test_given_path(self):
        self.assertEqual(
            Path('models/best/epoch_2.pth.tar'),
            find_resume_target(self.normal_model_path / "best", Path("epoch_2.pth.tar"))
        )
        self.assertEqual(
            Path('models/best/epoch_2.pth.tar'),
            find_resume_target(self.normal_model_path / "best", "epoch_2.pth.tar")
        )
        self.assertEqual(
            Path('models/best/epoch_1.pth.tar'),
            find_resume_target(self.normal_model_path / "best", 1)
        )


if __name__ == '__main__':
    unittest.main()
