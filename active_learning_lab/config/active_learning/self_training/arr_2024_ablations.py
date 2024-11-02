from active_learning_lab.config.active_learning.self_training.arr_2024 import (
    DEFAULT_CONFIG,
    TMP_BASE,
    update_config,
    set_defaults,
    Container
)


EXPERIMENT_NAME = 'arr24-self-training-ablations'

_unused = DEFAULT_CONFIG, TMP_BASE, Container, update_config, set_defaults
