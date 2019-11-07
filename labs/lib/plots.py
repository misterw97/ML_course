# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from labs.lib.utils import array_map, unzip


def plot_features(y, x, col_names=None, N_COLS=3):
    N_FEATURES = x.shape[1]
    N_ROWS = int(np.ceil(N_FEATURES / N_COLS))
    fig, ax = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 10, N_ROWS * 7))
    for i in range(N_FEATURES):
        index = (int(i / N_COLS), int(i % N_COLS))
        ax[index].scatter(x[:, i], y)
        ax[index].set_title(f'F[{i}] {(col_names[i] if col_names is not None else "")}')


def plot_weights(weights, col_names):
    wl = zip(weights, col_names)
    wl = sorted(wl, key=lambda r: -abs(r[0]))
    values, labels = unzip(wl)
    colors = array_map(lambda w: "b" if w > 0 else "r", values)
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(8, 8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, abs(np.array(values)), height=0.8, align="center", color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Feature")
    ax.set_title("Features absolute values (red: negative, blue: positive)")
    plt.show()


def plot_binary_distribution(y):
    positives = np.count_nonzero(y == 1)
    negatives = np.count_nonzero(y == -1)
    total = positives + negatives
    middle = total / 2
    fig, ax = plt.subplots(figsize=(10, 1))
    p1 = ax.barh(1, positives, left=-middle + negatives, color="b")
    p2 = ax.barh(1, negatives, left=-middle, color="r")
    ax.legend((p1[0], p2[0]), ("Positives", "Negatives"))
    plt.show()
