# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(lambds, mse_tr, mse_te, tr_label='train error', te_label='test error', show_error=False, std_tr=[], std_te=[], best_lambda=None):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", label=tr_label)
    plt.semilogx(lambds, mse_te, marker=".", label=te_label)
    if show_error:
        plt.errorbar(lambds, mse_tr, yerr=std_tr)
        plt.errorbar(lambds, mse_te, yerr=std_te)
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    
    if best_lambda is not None:
        plt.axvline(x = best_lambda)
    
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")
