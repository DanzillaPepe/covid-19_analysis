import sys
import random
import numpy as np
from math import e

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import time

import consts
import tools
import data_loading


def kNN(corr_list, y_axis, k=5, **params):
    scrub = corr_list + [y_axis]
    df = data_loading.load_preprocess_scrub(scrub, **params)

    if params.get('file'):
        file = open(params['file'], 'w')
    else:
        file = sys.stdout

    X = df.loc[:, corr_list].values
    y = df.loc[:, y_axis].values

    countries_shuffled = consts.COUNTRIES_LIST.copy()
    random.shuffle(countries_shuffled)
    countries_list = list()

    X_test, y_test = list(), list()
    i = 0
    while len(X_test) < len(df.index) * 0.1:
        new_country = countries_shuffled[i]
        countries_list.append(new_country)

        X_test += list(data_loading.country_truncation(df, new_country).loc[:, corr_list].values)
        y_test += list(data_loading.country_truncation(df, new_country).loc[:, y_axis].values)
        i += 1

    X_test, y_test = np.array(X_test), np.array(y_test)

    df_exluded = data_loading.country_exclusion(df, countries_list)
    X_train = df_exluded.loc[:, corr_list].values
    y_train = df_exluded.loc[:, y_axis].values

    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    data = [X_train, X_test, y_train, y_test]
    results, labels = annealing(data, k, corr_list)
    results, labels = [y_axis] + results, ['Guessed parameter'] + labels

    max_len = max(tools.max_str(corr_list), tools.max_str(labels)) + len(consts.TAB) + len(
        str(len(corr_list))) + 2 + len(
        consts.TAB)
    label_format = '{:' + str(max_len) + 's}'
    float_format = ':{:s}'.format(2 * consts.TAB) + '{:0.2f}\n'
    int_format = ':{:s}'.format(2 * consts.TAB) + '{:d}\n'
    str_format = ':{:s}'.format(2 * consts.TAB) + '{:s}\n'
    formates = {np.float64: float_format, float: float_format, int: int_format, str: str_format}

    for i in range(len(results)):
        result = results[i]
        label = labels[i]

        if type(result) in formates.keys():
            tools.write_formatted(file, label_format, formates, label, result)
        elif type(result) is list:
            file.write((label_format + '\n').format(label))

            lst = [(result[0][j], result[1][j]) for j in range(len(result[0]))]
            lst.sort(key=lambda w: -w[0])

            for j in range(len(lst)):
                label = '{:s}{:d}. '.format(consts.TAB, j + 1) + lst[j][1]
                tools.write_formatted(file, label_format, formates, label, lst[j][0])
            file.write('\n')


def local_change(weights):
    ind = random.randint(0, len(weights) - 1)
    weight = weights[ind]

    signs = [1, -1]
    new_weight = -1
    while new_weight < 0:
        rnd = random.random()
        coef = random.choice(signs) * rnd
        new_weight = weight + coef

    new_weights = weights.copy()
    new_weights[ind] = new_weight
    return new_weights


def count_score(data, weights, k):
    X_train, X_test, y_train, y_test = data
    classifier = KNeighborsRegressor(n_neighbors=k, metric='minkowski', p=2, metric_params={'w': weights})
    classifier.fit(X_train, y_train)

    return classifier.score(X_test, y_test)


def annealing(data, k, corr_list):
    start_time = time.time()
    X_train, X_test, y_train, y_test = data

    old_weights = [1] * X_train.shape[1]
    old_score = count_score(data, old_weights, k)
    best_score = old_score
    best_weights = list()

    t = consts.tMax
    iterations = 0
    while t >= consts.tMin:
        iterations += 1
        t *= consts.tMult

        new_weights = local_change(old_weights)
        new_score = count_score(data, new_weights, k)
        if not (new_score > old_score or random.random() <= e ** ((new_score - old_score) / t)):
            continue
        old_weights = new_weights
        old_score = new_score

        if new_score > best_score:
            best_score = new_score
            best_weights = new_weights.copy()

    results = [best_score,
               [best_weights, corr_list],
               iterations,
               time.time() - start_time,
               y_train.size + y_test.size,
               [[y_train.size, y_test.size], ['Training samples', 'Testing samples']]
               ]

    labels = ['Best score',
              'Optimal weights',
              'Iterations performed',
              'Annealing time (seconds)',
              'All samples',
              'Including']

    return results, labels
