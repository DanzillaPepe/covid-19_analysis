import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime as dt

import consts
import tools
import data_loading
import linear_regression


def plot(df_list, x_axis, y_axis, labels, mode, **params):
    fig, ax = plt.subplots()

    df_list = tools.make_list(df_list)
    labels = tools.make_labels(labels, df_list)

    logy = params.get('logy')

    for i in range(len(df_list)):
        df = df_list[i]
        color = i % len(consts.COLORS)

        if x_axis == 'date':
            dates = [dt.datetime.strptime(d, '%Y-%m-%d') for d in df[x_axis]]
            xs = pd.Series(['{0}.{1}\n{2}'.format(date.day, date.month, date.year) for date in dates], dtype='str')
        else:
            xs = df[x_axis]

        if y_axis == 'frequency':
            new_xs, new_ys = list(), list()
            step = (xs.max() - xs.min()) / consts.BINS
            if tools.is_nan(step):
                continue
            for bin_id in range(consts.BINS):
                x_value = xs.min() + bin_id * step + 0.5 * step
                new_xs.append(x_value)
                y_series = df[(x_value - 0.5 * step <= df[x_axis]) & (df[x_axis] <= x_value + 0.5 * step)]
                new_ys.append(len(y_series))
            xs, ys = pd.Series(new_xs), pd.Series(new_ys)
        else:
            ys = df[y_axis]
        label = labels[i]

        if params.get('make_bins'):
            values = df[[x_axis, y_axis]]

            new_xs, new_ys = list(), list()
            step = (xs.max() - xs.min()) / consts.BINS
            for bin_id in range(consts.BINS):
                x_value = 0.5 * step + bin_id * step
                y_series = values[(x_value - 0.5 * step <= values[x_axis]) & (values[x_axis] <= x_value + 0.5 * step)][
                    y_axis]
                y_value = y_series.mean()

                new_xs.append(x_value)
                new_ys.append(y_value)

            xs, ys = pd.Series(new_xs), pd.Series(new_ys)

        if params.get('world_delta'):
            df_global = data_loading.load_preprocess_scrub(x_axis)
            new_ys = list()
            values = df[[y_axis, 'date']]
            for y_value, date in values.to_numpy():
                world_data = df_global[df_global['date'] == date][y_axis]
                world_mean = world_data.mean()
                delta = world_mean - y_value
                new_ys.append(delta)
            ys = pd.Series(new_ys)

        if logy:
            ys = ys.apply(np.log).apply(lambda x: x if not tools.is_inf(x) and not tools.is_nan(x) else math.nan)

        r = ys.corr(xs)
        label += '\nr = {:0.2f}'.format(r)
        label = label.strip()

        if params.get('regression'):
            k, r_2 = linear_regression.plot_regression_line(xs, ys, ax, (color + 1) % len(consts.COLORS))
            label += '\nk = {:0.2f}\nR^2 = {:0.2f}'.format(k, r_2)
            label = label.lstrip()

        df_rendered = pd.DataFrame(np.array([xs, ys]).T)
        df_rendered = data_loading.scrub_data(df_rendered, [0, 1])
        if mode == 'scatter':
            df_rendered.plot.scatter(0, 1,
                                     grid=True,
                                     color=consts.COLORS[color],
                                     label=label,
                                     ax=ax,
                                     figsize=consts.FIG_SIZE,
                                     fontsize=consts.TICKS_SIZE,
                                     s=consts.DOT_SIZE,
                                     legend=None if label == '' else True,
                                     )
        elif mode == 'line':
            df_rendered.plot.line(0, 1,
                                  grid=True,
                                  color=consts.COLORS[color],
                                  label=label,
                                  ax=ax,
                                  figsize=consts.FIG_SIZE,
                                  fontsize=consts.TICKS_SIZE,
                                  linewidth=consts.LINE_WIDTH,
                                  legend=None if label == '' else True
                                  )

    x_label = x_axis
    plt.xlabel(x_label, fontdict=consts.LABELS_STYLE)
    y_label = y_axis
    if params.get('world_delta'):
        y_label = 'Δ' + y_label
    if logy:
        y_label = 'log(' + y_label + ')'
    plt.ylabel(y_label, fontdict=consts.LABELS_STYLE)

    plt.subplots_adjust(left=consts.LEFT_MARGIN, right=consts.RIGHT_MARGIN, top=consts.TOP_MARGIN,
                        bottom=consts.BOTTOM_MARGIN)

    plt.locator_params(axis='x', tight=True, nbins=consts.X_TICKS)
    if not logy:
        plt.locator_params(axis='y', tight=True, nbins=consts.Y_TICKS)

    plt.show()


def hist(df_list, x_axis, bins, labels=None, **params):
    df_list = tools.make_list(df_list)

    labels = tools.make_labels(labels, df_list)

    logy = params.get('logy')

    for i in range(len(df_list)):
        df = df_list[i]
        color = i % len(consts.COLORS)

        xs = df[x_axis]

        label = labels[i]

        df_rendered = pd.DataFrame(xs)
        df_rendered.plot.hist(bins=bins,
                              grid=True,
                              color=consts.COLORS[color],
                              label=label,
                              figsize=consts.FIG_SIZE,
                              fontsize=consts.TICKS_SIZE,
                              legend=None if label == '' else True,
                              logy=logy
                              )

        plt.xlabel(x_axis, fontdict=consts.LABELS_STYLE)
        y_label = 'frequency'
        if logy:
            y_label = 'log(frequency)'
        plt.ylabel(y_label, fontdict=consts.LABELS_STYLE)

        plt.subplots_adjust(left=consts.LEFT_MARGIN, right=consts.RIGHT_MARGIN, top=consts.TOP_MARGIN,
                            bottom=consts.BOTTOM_MARGIN)

        patch = mpatches.Patch(color=consts.COLORS[color], label=label)
        plt.legend(handles=[patch])

        plt.locator_params(axis='x', tight=True, nbins=consts.X_TICKS)
        if not logy:
            plt.locator_params(axis='y', tight=True, nbins=consts.Y_TICKS)

        plt.show()


def countries_histogram(x_axis, countries, **params):
    scrub = x_axis
    df = data_loading.load_preprocess_scrub(scrub, **params)
    countries = tools.make_list(countries)

    if params.get('date'):
        df = data_loading.date_truncation(df, params['date'])

    df_list = list()
    for country in countries:
        df_list.append(
            data_loading.country_truncation(
                df,
                country))

    hist(df_list, x_axis, consts.BINS, countries, **params)


def countries_plot(x_axis, y_axis, countries, mode, **params):
    scrub = [x_axis, y_axis]
    df = data_loading.load_preprocess_scrub(scrub, **params)
    countries = tools.make_list(countries)

    df_list = list()
    for country in countries:
        df_truncated = data_loading.country_truncation(df, country)
        if params.get('date'):
            df_truncated = data_loading.date_truncation(df_truncated, params['date'])

        df_list.append(df_truncated)

    plot(df_list, x_axis, y_axis, countries, mode=mode, **params)


def linear_rate_corr(corr1, corr2, corr_name, y_axis, **params):
    scrub = [corr1, corr2]
    df = data_loading.load_preprocess_scrub(scrub, **params)

    for country in consts.COUNTRIES_LIST:
        df_c = data_loading.country_truncation(df, country)
        xs = df_c[corr1]
        ys = df_c[corr2]

        corr = linear_regression.slope(xs, ys)
        df.loc[df['location'] == country, corr_name] = corr

    if params.get('date'):
        df = data_loading.date_truncation(df, params['date'])

    plot(df, corr_name, y_axis, None, mode='scatter', **params)


def inter_countries_plot(x_axis, y_axis, label=None, mode='scatter', **params):
    scrub = [x_axis, y_axis]
    df = data_loading.load_preprocess_scrub(scrub, **params)
    df = df[df[y_axis] < 1000]

    plot(df, x_axis, y_axis, label, mode=mode, **params)
