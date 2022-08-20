import consts
import data_loading
import plotting
import kNN

countries_entry = ['United States',
                   'China',
                   'Russia',
                   'Spain',
                   'Ukraine',
                   'Germany',
                   'Georgia',
                   'Germany',
                   'Zimbabwe']

corr_list = [
    'population_density',
    'gdp_per_capita',
    'hospital_beds_per_thousand',

    'diabetes_prevalence',
    'cardiovasc_death_rate',

    'median_age',
    'aged_65_older',
    'aged_70_older',

    'k_v/c',
    'stringency_index'
]

kNN.kNN(corr_list=corr_list, y_axis='k_d/c', file='kNN_results.txt', one_sample_per_country=True)

"""
countries_plot(x_axis, y_axis, countries_entry, mode='line', regression=False, logy=False, world_delta=False)


plotting.inter_countries_plot(x_axis='k_v/c', y_axis='total_cases_per_million',
                              label=None,
                              mode='scatter',
                              mean=False,
                              make_bins=False,
                              logy=True,
                              regression=True,
                              date=consts.CUSTOM_DATE,
                              one_sample_per_country=False)
"""