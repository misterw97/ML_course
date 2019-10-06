# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def compute_error(y, tx, w):
    return y - (tx @ w)