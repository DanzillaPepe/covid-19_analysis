import matplotlib
import datetime as dt

import data_loading

COUNTRIES_LIST = list(set(data_loading.load_data()['location']))

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
TAB = '    '

FIG_SIZE = (21, 14)
DOT_SIZE = 42
LINE_WIDTH = 3
X_TICKS = 25
Y_TICKS = 20
CUSTOM_DATE = dt.datetime.strptime('18.08.2022', '%d.%m.%Y')
BINS = 120

LEFT_MARGIN = 0.1
RIGHT_MARGIN = 0.95
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

MY_COLUMNS = {
    'k_v/c': ['total_cases', 'total_vaccinations'],
    'k_d/c': ['total_cases', 'total_deaths'],
    'smokers': ['male_smokers', 'female_smokers']
}

averageScoreDelta = 0.05
tMax = 1.0 * averageScoreDelta
tMin = 0.1 * averageScoreDelta
tMult = 0.9999999

# tMult = 0.9999999
