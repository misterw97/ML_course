# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def f_affine(w0, w1, x):
    return x*w1+w0

def plot_fit(ws, x, y, f = f_affine):
    xmin = min(x)
    xmax = max(x)
    w1 = ws[len(ws)-1][1]
    w0 = ws[len(ws)-1][0]
    print('w0', w0, 'w1', w1, 'pts', ([0.0,2.0], [w0, 2.0*w1+w0]))
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(x, y, s = 40, c = 'b', marker = '+')
    ax1.plot([xmin,xmax], [f(w0, w1, xmin), f(w0, w1, xmax)], color = 'red', linestyle = 'solid')
    plt.show()