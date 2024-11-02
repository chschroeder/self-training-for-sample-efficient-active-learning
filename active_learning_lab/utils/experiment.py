import os
import torch

import numpy as np

from pathlib import Path


def set_random_seed(seed, pytorch=True):
    # PYTHONHASHSEED and numpy seed have the smaller range (2**32-1)
    assert 0 <= seed <= 2**32-1

    os.environ['PYTHONHASHSEED'] = str(seed)
    if pytorch:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_data_dir():
    if 'DATA_DIR' in os.environ:
        base_dir = os.environ['DATA_DIR']
    else:
        base_dir = '.data/'
    return base_dir


def get_tmp_dir(tmp_base, folder_name):
    path = Path(tmp_base).joinpath(folder_name)
    if not path.exists():
        path.mkdir(parents=True)

    return str(path.absolute())
