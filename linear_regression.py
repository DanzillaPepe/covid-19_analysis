import pandas as pd
from math import ceil

import consts

regression_line = lambda k, b: lambda x: k * x + b


def slope(xs, ys):
    return xs.cov(ys) / xs.var()


def intercept(xs, ys):
    return ys.mean() - (xs.mean() * slope(xs, ys))


def plot_regression_line(xs, ys, ax, color):
    k = slope(xs, ys)
    b = intercept(xs, ys)
    left_border, righht_border = xs.min() * 0.9, xs.max() * 1.1
    step = (righht_border - left_border) / consts.BINS
    s = pd.Series(range(int(left_border), int(righht_border), ceil(step)), dtype='float64')

    df = pd.DataFrame({0: s, 1: s.map(regression_line(k, b))})
    df.plot(0, 1,
            grid=True,
            figsize=consts.FIG_SIZE,
            color=consts.COLORS[color],
            ax=ax,
            linewidth=consts.LINE_WIDTH,
            legend=False
            )
    r_2 = r_squared(xs, ys, k, b)
    return k, r_2


def residuals(xs, ys, k, b):
    estimate = regression_line(k, b)
    return pd.Series(map(lambda x, y: y - estimate(x), xs, ys))


def r_squared(xs, ys, k, b):
    r_var = residuals(xs, ys, k, b).var()
    y_var = ys.var()
    return 1 - (r_var / y_var)
