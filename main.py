import consts
import plotting
import kNN

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

kNN.kNN(corr_list=corr_list, y_axis='total_deaths_per_million', file='kNN_results.txt', one_sample_per_country=True)

"""
plotting.inter_countries_plot(x_axis='new_deaths_per_million', y_axis='frequency',
                              label=None,
                              mode='scatter',
                              mean=False,
                              make_bins=True,
                              logy=False,
                              regression=True,
                              date=None,
                              one_sample_per_country=False)

plotting.countries_histogram(x_axis='new_cases',
                             countries=countries_entry[2],
                             label=None,
                             logy=False,
                             date=None,
                             one_sample_per_country=False,
                             center=True)
"""
