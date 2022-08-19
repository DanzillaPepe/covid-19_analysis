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

kNN.kNN(corr_list=corr_list, y_axis='total_cases_per_million', file='kNN_results.txt', one_sample_per_country=True)

"""
countries_plot(x_axis, y_axis, countries_entry, mode='line', regression=False, logy=False, world_delta=False)
inter_countries_plot(x_axis='k_v/c', y_axis='k_d/c', mode='scatter', mean=False, make_bins=True, regression=False, date=CUSTOM_DATE)
"""
