import numpy as np


def bald(p, eps=1e-8):
    p_mean = np.mean(p, axis=1)
    model_prediction_entropy = -np.sum(p_mean * np.log2(p_mean + eps), axis=-1)
    expected_prediction_entropy = -np.mean(np.sum(p * np.log2(p + eps), axis=-1), axis=1)
    return model_prediction_entropy - expected_prediction_entropy
