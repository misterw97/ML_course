# -*- coding: utf-8 -*-
import numpy as np

def unzip(zipped):
    return [ i for i, _ in zipped ], [ j for _, j in zipped ]

def arrayMap(f, *x):
    return np.array(list(map(f,*x)))

def array_map(f, *x):
    return np.array(list(map(f,*x)))


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    ones = np.ones(x.shape[0])
    polys = np.array([x**(n+1) for n in range(degree)]).T
    tx = np.column_stack((ones, polys))
    
    #randoms = np.array([rand.random() for n in range(x.shape[0])])
    #tx = np.column_stack((tx, randoms))

    return tx


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # source: https://stackoverflow.com/a/3677283
    indices = np.random.permutation(x.shape[0])
    rate = int(np.floor(indices.shape[0] * ratio))
    training_idx, test_idx = indices[:rate], indices[rate:]
    training = (x[training_idx], y[training_idx])
    test = (x[test_idx], y[test_idx])
    return training, test