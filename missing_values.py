import pandas as pd


def calculate_missing_values_total_and_percentage(data):
    """

    :param data: the data set we want to calculate the count and percentage of missing values in each of its columns
    """
    missing_values_per_column = (data.isnull().sum())
    missing_values_per_column_percentage = ((data.isnull().sum() * 100) / len(data))
    missing_data_information = pd.concat([missing_values_per_column, missing_values_per_column_percentage], axis=1, sort=True)  # here we added the sort due to a warning!
    missing_data_information.rename(columns={0: 'Missing_Values_Count', 1: 'Percentage'}, inplace=True)
    missing_data_information = missing_data_information[missing_data_information['Missing_Values_Count'] != 0]
    if not missing_data_information.empty:
        missing_data_information.sort_values(by='Percentage', ascending=False, inplace=True)
        print('This data set contains {} columns with missing values'.format(missing_data_information.shape[0]))
        print(missing_data_information)
    else:
        print('No missing values in this data set')
