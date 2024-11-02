import time
from contextlib import contextmanager


# https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
@contextmanager
def measure_time_context():
    start_time = end_time = time.perf_counter()
    yield lambda: end_time - start_time
    end_time = time.perf_counter()


def measure_time(func, has_return_value=True):
    start_time = time.perf_counter()
    return_values = func()
    end_time = time.perf_counter()

    if has_return_value:
        return (end_time-start_time), return_values
    else:
        return end_time-start_time
 
