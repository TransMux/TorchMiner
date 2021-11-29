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
        # TODO:os.path.join
        current_dir = os.path.join(current_dir, dir_name)
    if not os.path.isdir(current_dir):
        os.mkdir(current_dir)
