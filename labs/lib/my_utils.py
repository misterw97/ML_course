# -*- coding: utf-8 -*-
import numpy as np

def arrayMap(f, *x):
    return np.array(list(map(f,*x)))