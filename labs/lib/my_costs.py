# -*- coding: utf-8 -*-
import numpy as np


def compute_error(y, tx, w):
    return y - np.dot(tx, w)


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e ** 2)


def calculate_rmse(e):
    """Calculate the root mse for vector e."""
    mse = calculate_mse(e)
    return np.sqrt(2 * mse)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(abs(e))
