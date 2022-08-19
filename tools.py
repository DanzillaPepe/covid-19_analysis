from math import isnan, isinf


def make_list(lst):
    if type(lst) is not list and type(lst) is not None:
        return [lst]
    elif type(lst) is None:
        return list()
    return lst


def make_labels(labels, df_list):
    if labels is None:
        labels = [''] * len(df_list)
    return labels


def max_str(arr):
    arr = list(map(len, arr))
    arr.sort()
    arr.reverse()
    return arr[0]


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


def get_date_str(date):
    date_str = '{year}-{zero1}{month}-{zero2}{day}'.format(day=date.day,
                                                           month=date.month,
                                                           year=date.year,
                                                           zero1="0" if date.day < 10 else "",
                                                           zero2="0" if date.month < 10 else "",
                                                           )
    return date_str

def write_formatted(file, label_format, formats, label, value):
    file.write((label_format + formats[type(value)]).format(label, value))