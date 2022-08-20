import pandas as pd
from functools import reduce

import consts
import tools
import linear_regression


def load_data():
    df = pd.read_csv('data/owid-covid-data.csv')
    return df


def preprocess(df, pre_scrub):
    df_prepr = scrub_data(df, pre_scrub)
    for country in consts.COUNTRIES_LIST:
        df_c = country_truncation(df_prepr, country)
        xs = df_c['total_cases']

        ys = df_c['total_vaccinations']
        k_vc = linear_regression.slope(xs, ys)
        df_prepr.loc[df_prepr['location'] == country, 'k_v/c'] = k_vc

        ys = df_c['total_deaths']
        k_dc = linear_regression.slope(xs, ys)
        df_prepr.loc[df_prepr['location'] == country, 'k_d/c'] = k_dc

        male_smokers = df_c['male_smokers'].mean()
        female_smokers = df_c['female_smokers'].mean()
        df_prepr.loc[df_prepr['location'] == country, 'smokers'] = (male_smokers + female_smokers) / 2

    return df_prepr


def scrub_data(df, scrub, **params):
    df_scrubbed = df.copy()
    for column in scrub:
        df_scrubbed = df_scrubbed[df_scrubbed[column].notna()]

    if params.get('countries'):
        df_scrubbed = country_truncation(df_scrubbed, params['countries'])
    if params.get('except_countries'):
        df_scrubbed = country_exclusion(df_scrubbed, params['except_countries'])
    if params.get('date'):
        df_scrubbed = date_truncation(df_scrubbed, params['date'])
    if params.get('one_sample_per_country'):
        new_df = dict([('location', [])])
        for country in consts.COUNTRIES_LIST:
            df_c = country_truncation(df_scrubbed, country)

            if df_c.empty:
                continue
            new_df['location'].append(country)
            for column in scrub:
                if column == 'total_cases_per_million':
                    to_add = df_c[column].max()
                else:
                    to_add = df_c[column].mean(skipna=True)

                if not new_df.get(column):
                    new_df[column] = [to_add]
                else:
                    new_df[column].append(to_add)
        df_scrubbed = pd.DataFrame(new_df)
    return df_scrubbed


def load_preprocess_scrub(scrub=None, **params):
    df = load_data()
    to_scrub = tools.make_list(scrub)
    pre_scrub = [column for column in to_scrub if column not in consts.MY_COLUMNS.keys()] + \
                list(set().union(*[consts.MY_COLUMNS[my_column] for my_column in consts.MY_COLUMNS.keys() if
                                   my_column in scrub]))
    df = preprocess(df, pre_scrub)
    df = scrub_data(df, to_scrub, **params)

    return df


def country_truncation(df, countries):
    df_truncated = list()
    countries = tools.make_list(countries)
    for country in countries:
        df_c = df[df['location'] == country]
        df_truncated.append(df_c)
    return pd.concat(df_truncated)


def country_exclusion(df, countries):
    df_truncated = list()
    countries = tools.make_list(countries)
    for country in countries:
        df_c = df[df['location'] != country]
        df_truncated.append(df_c)
    return reduce(lambda left, right: pd.merge(left, right, how="inner"),
                  df_truncated)


def date_truncation(df, date):
    return df[df['date'] == tools.get_date_str(date)]
