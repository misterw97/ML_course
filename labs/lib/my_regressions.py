# -*- coding: utf-8 -*-
from my_costs import *
from my_utils import batch_iter

# TODO: unit tests


def least_squares(y, tx, loss_fn=calculate_mse):
    """Least squares normal equations."""
    w = np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def compute_gradient(y, tx, w):
    error = compute_error(y, tx, w)
    return -1 / y.shape[0] * np.dot(tx.T, error)


def least_squares_GD(y, tx, initial_w, max_iters, gamma, loss_fn=calculate_mse):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_fn=calculate_mse):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    i = 0
    batch_size = 1
    for mini_y, mini_x in batch_iter(y, tx, batch_size, max_iters):
        gradient = compute_gradient(mini_y, mini_x, w)
        w = w - gamma * gradient
        i = i + 1
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def ridge_regression(y, tx, lambda_, loss_fn=calculate_mse):
    """Ridge regression equations."""
    N = y.shape[0]
    D = tx.shape[1]
    a = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    loss = loss_fn(compute_error(y, tx, w))
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma, loss_fn=calculate_mse):
    # TODO
    raise NotImplementedError


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, loss_fn=calculate_mse):
    # TODO
    raise NotImplementedError
