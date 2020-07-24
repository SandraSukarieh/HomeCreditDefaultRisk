import numpy as np


def check_outliers(data, feature):
    """

    :param data: the data set we want to check outliers for
    :param feature: the specific feature to check whether it contains outliers
    :return: two series of lower and upper outliers for further checking
    """
    column = data[feature]
    column_mean = column.mean()
    column_std = column.std()

    column_lower_bound = column_mean - 2 * column_std
    column_upper_bound = column_mean + 2 * column_std

    lower_outliers = column[column < column_lower_bound]
    upper_outliers = column[column > column_upper_bound]

    return lower_outliers, upper_outliers


def replace_outliers_with_nan_for_feature(data, feature, is_max):
    """

    :param data: the data set we want to deal with outliers in
    :param feature: the feature we want to remove its outlier values
    :param is_max: if True then outliers are in the high tail of the feature distribution
    :return: the outlier value to perform unit test
    """
    if is_max:
        outlier_value = data[feature].max()
    else:
        outlier_value = data[feature].min()

    anomaly_column = feature + '_ANOMALY'
    data[anomaly_column] = False
    data.loc[data[feature] == outlier_value, anomaly_column] = True
    data[feature].replace({outlier_value: np.nan}, inplace=True)

    return outlier_value


