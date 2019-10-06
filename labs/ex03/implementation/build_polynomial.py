# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np
import random as rand


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    ones = np.ones(x.shape[0])
    polys = np.array([x**(n+1) for n in range(degree)]).T
    tx = np.column_stack((ones, polys))
    
    #randoms = np.array([rand.random() for n in range(x.shape[0])])
    #tx = np.column_stack((tx, randoms))

    return tx
