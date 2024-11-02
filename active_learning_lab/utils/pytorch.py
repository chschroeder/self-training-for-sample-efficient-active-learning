import gc
import logging
import torch

from contextlib import contextmanager


# inspired by:
#     https://discuss.pytorch.org
#     /t/is-there-anything-wrong-with-setting-default-tensor-type-to-cuda/27949/4
@contextmanager
def default_tensor_type(tensor_type):
    default_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(default_type)


def free_resources_fix(full=False):
    """This should never be necessary in theory but has been a successful remedy for some situations in which
    the GPU memory was full."""
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if full:
            torch.cuda.ipc_collect()
    except RuntimeError as e:
        logging.info(f'Exception during free_resources_fix(): {e}')

    if full:
        gc.collect()
