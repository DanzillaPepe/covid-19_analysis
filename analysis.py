import pandas
import scipy as sp
from scipy import stats
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import numpy as np
import datetime as dt
from math import sqrt, ceil, log, isnan, isinf, e
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale, normalize, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_percentage_error, \
    accuracy_score
from functools import reduce

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

FIG_SIZE = (21, 14)
DOT_SIZE = 24
LINE_WIDTH = 3
X_TICKS = 30
Y_TICKS = 30
CUSTOM_DATE = dt.datetime.strptime('9.08.2021', '%d.%m.%Y')
BINS = 60

LEFT_MARGIN = 0.08
RIGHT_MARGIN = 0.98
TOP_MARGIN = 0.95
BOTTOM_MARGIN = 0.1

LABELS_SIZE = 30
LABELS_STYLE = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': LABELS_SIZE}

TICKS_SIZE = 22
TICKS_STYLE = {'family': 'DejaVu Sans',
               'size': TICKS_SIZE}

matplotlib.rc('font', **TICKS_STYLE)

averageScoreDelta = 0.1

tMax = 1.0 * averageScoreDelta
tMin = 0.1 * averageScoreDelta
tMult = 0.9


def load_data(scrub=None, **params):
    df = pd.read_csv('data/owid-covid-data.csv')
    if scrub:
        scrub = make_list(scrub)
        for column in scrub:
            df = df[df[column].notna()]
    if params.get('countries'):
        df = country_truncation(df, params['countries'])
    if params.get('except_countries'):
        df = country_exclusion(df, params['except_countries'])
    if params.get('date'):
        df = date_truncation(df, params['date'])
    return df


COUNTRIES_LIST = list(set(load_data()['location']))


def make_list(lst):
    if type(lst) is not list:
        return [lst]
    return lst


def make_labels(labels, df_list):
    if labels is None:
        labels = [''] * len(df_list)
    return labels


def is_nan(x):
    try:
        return isnan(x)
    except:
        return False


def is_inf(x):
    try:
        return isinf(x)
    except:
        return False


def country_truncation(df, countries):
    df_truncated = list()
    countries = make_list(countries)
    for country in countries:
        df_c = df[df['location'] == country]
        df_truncated.append(df_c)
    return pd.concat(df_truncated)


def country_exclusion(df, countries):
    df_truncated = list()
    countries = make_list(countries)
    for country in countries:
        df_c = df[df['location'] != country]
        df_truncated.append(df_c)
    return reduce(lambda left, right: pd.merge(left, right, how="inner"),
                  df_truncated)


def get_date_str(date):
    date_str = '{year}-{zero1}{month}-{zero2}{day}'.format(day=date.day,
                                                           month=date.month,
                                                           year=date.year,
                                                           zero1="0" if date.day < 10 else "",
                                                           zero2="0" if date.month < 10 else "",
                                                           )
    return date_str


def date_truncation(df, date):
    return df[df['date'] == get_date_str(date)]


def slope(xs, ys):
    return xs.cov(ys) / xs.var()


def intercept(xs, ys):
    return ys.mean() - (xs.mean() * slope(xs, ys))


regression_line = lambda k, b: lambda x: k * x + b


def plot_regression_line(xs, ys, ax, color):
    k = slope(xs, ys)
    b = intercept(xs, ys)
    left_border, righht_border = xs.min() * 0.9, xs.max() * 1.1
    step = (righht_border - left_border) / BINS
    s = pd.Series(range(int(left_border), int(righht_border), ceil(step)), dtype='float64')

    df = pd.DataFrame({0: s, 1: s.map(regression_line(k, b))})
    df.plot(0, 1,
            grid=True,
            figsize=FIG_SIZE,
            color=COLORS[color],
            ax=ax,
            linewidth=LINE_WIDTH,
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


def plot(df_list, x_axis, y_axis, labels, mode, **params):
    fig, ax = plt.subplots()

    df_list = make_list(df_list)
    labels = make_labels(labels, df_list)

    logy = params.get('logy')

    n = len(df_list[0][x_axis])
    for i in range(len(df_list)):
        df = df_list[i]
        color = i % len(COLORS)

        if x_axis == 'date':
            dates = [dt.datetime.strptime(d, '%Y-%m-%d') for d in df[x_axis]]
            xs = pd.Series(['{0}.{1}\n{2}'.format(date.day, date.month, date.year) for date in dates], dtype='str')
        else:
            xs = df[x_axis]

        if y_axis == 'frequency':
            new_xs, new_ys = list(), list()
            step = (xs.max() - xs.min()) / BINS
            if is_nan(step):
                continue
            for bin_id in range(BINS):
                x_value = xs.min() + bin_id * step + 0.5 * step
                new_xs.append(x_value)
                y_series = df[(x_value - 0.5 * step <= df[x_axis]) & (df[x_axis] <= x_value + 0.5 * step)]
                new_ys.append(len(y_series))
            xs, ys = pd.Series(new_xs), pd.Series(new_ys)
        else:
            ys = df[y_axis]

        label = labels[i]

        if params.get('mean'):
            values = df[[x_axis, y_axis]]
            new_ys = list()
            for x_value in xs.to_numpy():
                y_series = values[values[x_axis] == x_value][y_axis]
                new_ys.append(y_series.mean())
            ys = pd.Series(new_ys)

        if params.get('make_bins'):
            values = df[[x_axis, y_axis]]

            new_xs, new_ys = list(), list()
            step = (xs.max() - xs.min()) / BINS
            for bin_id in range(BINS):
                x_value = 0.5 * step + bin_id * step
                new_xs.append(x_value)
                y_series = values[(x_value - 0.5 * step <= values[x_axis]) & (values[x_axis] <= x_value + 0.5 * step)][
                    y_axis]
                new_ys.append(y_series.mean())
            xs, ys = pd.Series(new_xs), pd.Series(new_ys)

        if params.get('world_delta'):
            df_global = load_data(scrub=x_axis)
            new_ys = list()
            values = df[[y_axis, 'date']]
            for y_value, date in values.to_numpy():
                world_data = df_global[df_global['date'] == date][y_axis]
                world_mean = world_data.mean()
                delta = world_mean - y_value
                new_ys.append(delta)
            ys = pd.Series(new_ys)

        if logy:
            ys = ys.apply(np.log).apply(lambda x: x if not is_inf(x) and not is_nan(x) else 0)

        r = ys.corr(xs)
        label += '\nr = {:0.2f}'.format(r)
        label = label.strip()

        if params.get('regression'):
            k, r_2 = plot_regression_line(xs, ys, ax, (color + 1) % len(COLORS))
            label += '\nk = {:0.2f}\nR^2 = {:0.2f}'.format(k, r_2)
            label = label.lstrip()

        df_rendered = pd.DataFrame(np.array([xs, ys]).T)
        if mode == 'scatter':
            df_rendered.plot.scatter(0, 1,
                                     grid=True,
                                     color=COLORS[color],
                                     label=label,
                                     ax=ax,
                                     figsize=FIG_SIZE,
                                     fontsize=TICKS_SIZE,
                                     s=DOT_SIZE,
                                     legend=None if label == '' else True,
                                     )
        elif mode == 'line':
            df_rendered.plot.line(0, 1,
                                  grid=True,
                                  color=COLORS[color],
                                  label=label,
                                  ax=ax,
                                  figsize=FIG_SIZE,
                                  fontsize=TICKS_SIZE,
                                  linewidth=LINE_WIDTH,
                                  legend=None if label == '' else True
                                  )

    x_label = x_axis
    plt.xlabel(x_label, fontdict=LABELS_STYLE)
    y_label = y_axis
    if params.get('world_delta'):
        y_label = 'Δ' + y_label
    if logy:
        y_label = 'log(' + y_label + ')'
    plt.ylabel(y_label, fontdict=LABELS_STYLE)

    plt.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=BOTTOM_MARGIN)

    plt.locator_params(axis='x', tight=True, nbins=X_TICKS)
    if not logy:
        plt.locator_params(axis='y', tight=True, nbins=Y_TICKS)

    plt.show()


def hist(df_list, x_axis, bins, labels=None, **params):
    df_list = make_list(df_list)

    labels = make_labels(labels, df_list)

    logy = params.get('logy')

    for i in range(len(df_list)):
        df = df_list[i]
        color = i % len(COLORS)

        xs = df[x_axis]

        label = labels[i]

        df_rendered = pd.DataFrame(xs)
        df_rendered.plot.hist(bins=bins,
                              grid=True,
                              color=COLORS[color],
                              label=label,
                              figsize=FIG_SIZE,
                              fontsize=TICKS_SIZE,
                              legend=None if label == '' else True,
                              logy=logy
                              )

        plt.xlabel(x_axis, fontdict=LABELS_STYLE)
        y_label = 'frequency'
        if logy:
            y_label = 'log(frequency)'
        plt.ylabel(y_label, fontdict=LABELS_STYLE)

        plt.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, top=TOP_MARGIN, bottom=BOTTOM_MARGIN)

        patch = mpatches.Patch(color=COLORS[color], label=label)
        plt.legend(handles=[patch])

        plt.locator_params(axis='x', tight=True, nbins=X_TICKS)
        if not logy:
            plt.locator_params(axis='y', tight=True, nbins=Y_TICKS)

        plt.show()


def countries_histogram(x_axis, countries, **params):
    df = load_data(scrub=x_axis, **params)

    if params.get('date'):
        df = date_truncation(df, params['date'])

    df_list = list()
    for country in countries:
        df_list.append(
            country_truncation(
                df,
                country))

    hist(df_list, x_axis, BINS, countries, **params)


def countries_plot(x_axis, y_axis, countries_to_plot, mode, **params):
    df = load_data(scrub=[x_axis, y_axis], **params)

    df_list = list()
    for country in countries_to_plot:
        df_truncated = country_truncation(df, country)
        if params.get('date'):
            df_truncated = date_truncation(df_truncated, params['date'])

        df_list.append(df_truncated)

    plot(df_list, x_axis, y_axis, countries_to_plot, mode=mode, **params)


def linear_rate_corr(corr1, corr2, corr_name, y_axis, **params):
    df = load_data(scrub=[corr1, corr2], **params)

    for country in COUNTRIES_LIST:
        df_c = country_truncation(df, country)
        xs = df_c[corr1]
        ys = df_c[corr2]

        corr = slope(xs, ys)
        df.loc[df['location'] == country, corr_name] = corr

    if params.get('date'):
        df = date_truncation(df, params['date'])

    plot(df, corr_name, y_axis, None, mode='scatter', **params)


def inter_countries_plot(x_axis, y_axis, mode, **params):
    df = load_data(scrub=[x_axis, y_axis], **params)

    plot(df, x_axis, y_axis, None, mode=mode, **params)


def kNN(corr_list, y_axis, k=5, **params):
    df = load_data(scrub=corr_list + [y_axis], **params)

    X = df.loc[:, corr_list].values
    y = df.loc[:, y_axis].values

    countries_shuffled = COUNTRIES_LIST.copy()
    random.shuffle(countries_shuffled)
    countries_list = list()

    X_test, y_test = list(), list()
    i = 0
    while len(X_test) < len(df.index) * 0.1:
        new_country = countries_shuffled[i]
        countries_list.append(new_country)

        X_test += list(country_truncation(df, new_country).loc[:, corr_list].values)
        y_test += list(country_truncation(df, new_country).loc[:, y_axis].values)

        i += 1
    X_test, y_test = np.array(X_test), np.array(y_test)

    df_exluded = country_exclusion(df, countries_list)
    X_train = df_exluded.loc[:, corr_list].values
    y_train = df_exluded.loc[:, y_axis].values

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    data = [X_train, X_test, y_train, y_test]
    annealing(data)

    """
    y_pred = classifier.predict(X_test)
    lr = LinearRegression().fit(X_train, y_train)
    """


def local_change(weights):
    new_weights = list()
    signs = [1, -1]
    for w in weights:
        coef = -w - 1
        while w + coef < 0:
            coef = random.choice(signs) * random.normalvariate(0, 0.5)
        new_weights.append(w + coef)
    return new_weights


def count_score(data, weights):
    X_train, X_test, y_train, y_test = data
    classifier = KNeighborsRegressor(n_neighbors=5, metric='wminkowski', p=2, metric_params={'w': weights})
    classifier.fit(X_train, y_train)

    return classifier.score(X_test, y_test)


def annealing(data):
    X_train, X_test, y_train, y_test = data

    old_weights = [1] * X_train.shape[1]
    old_score = count_score(data, old_weights)
    best_score = old_score
    best_weights = list()

    t = tMax
    deltas = list()
    while t >= tMin:
        t *= tMult
        new_weights = local_change(old_weights)
        new_score = count_score(data, new_weights)
        deltas.append(new_score - old_score)
        if not (new_score > old_score or random.random() <= e ** ((new_score - old_score) / t)):
            continue
        old_weights = new_weights
        if new_score > best_score:
            best_score = new_score
            best_weights = new_weights.copy()
    print('deltas mean = {:0.2f}'.format(pd.Series(deltas).mean()))
    print('best score = {:0.2f}'.format(best_score))
    print('best weights:', best_weights)


countries_entry = ['United States', 'China', 'Russia', 'Spain', 'Ukraine', 'Germany', 'Georgia', 'Germany'][2]
countries_entry = make_list(countries_entry)

x_axis = 'date'
y_axis = 'total_deaths_per_million'

corr1 = 'total_cases'
corr2 = 'total_vaccinations'
corr_name = 'k_v/c'

corr_list = ['population_density', 'gdp_per_capita', 'median_age', 'people_vaccinated_per_hundred',
             'human_development_index']

"""
countries_histogram('new_deaths_per_million', countries_entry, logy=True)
countries_plot(x_axis, y_axis, countries_entry, mode='line', regression=False, logy=False, world_delta=False)
linear_rate_corr(corr1, corr2, corr_name, y_axis='total_cases_per_million', make_bins=False, regression=True, logy=True, date=CUSTOM_DATE)
inter_countries_plot(x_axis, y_axis, mode='scatter', mean=True, make_bins=True, regression=True)
"""
kNN(corr_list, y_axis)
