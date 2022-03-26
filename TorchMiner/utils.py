import io
from pathlib import Path

import numpy as np


def seed_everything(seed):
    """
    Fix the seed for generating random numbers.
    :param seed:
    :return:
    """
    import torch
    from numpy import random

    torch.manual_seed(seed)
    # may lead to bad performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dir(directory, *args):
    """Create directory"""
    import os

    current_dir = directory
    for dir_name in args:
        current_dir = os.path.join(current_dir, dir_name)
    if not os.path.isdir(current_dir):
        os.mkdir(current_dir)


def figure2numpy(fig):
    # Solution to store Matplotlib Figure in TensorBoardX
    # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    im = im.transpose((2, 0, 1))  # Change to CHW
    return im


def find_resume_target(path: Path, index):
    """

    :param path: /
        Miner experiment path
    :param index: /
        True: Accept resume auto find result (Promise,default) Only best or latest
        string/Path: Will use the given checkpoint.
        int: Choose this epoch in auto find path.
    :return:
    """
    if index is True:
        search_paths = [
            path / "best.pth.tar",
            path / "latest.pth.tar",
        ]
    else:
        index = str(index)
        if Path(index).is_file():
            return Path(index)

        if (path / Path(index)).is_file():
            return path / Path(index)
        # The checkpoint is not given

        search_paths = [
            path,
            path / index,
            path / f"epoch_{index}.pth.tar",
            path / f"{index}.pth.tar",
            *path.glob("*.pth.tar"),
        ]

    for path in search_paths:
        if path.is_file():
            return path
    print(f"Tried to find Checkpoint in f{search_paths} but failed.")
    return None
