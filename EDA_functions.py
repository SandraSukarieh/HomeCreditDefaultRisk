import matplotlib.pyplot as plt
import seaborn as sns
import unittest

from missing_values_functions import calculate_missing_values_total_and_percentage


def get_basic_statistics(data):
    """

    :param data: the data set we want to apply basic statistics for
    """
    print('dimensions:')
    print(data.shape)
    print('\n')
    print('columns:')
    print(data.columns.values)
    print('\n')
    print('data types count:')
    print(data.dtypes.value_counts())
    print('\n')
    print('data types details:')
    print(data.dtypes)


def analyse_target_label(application_train):
    """

    :param application_train: the training data set
    """
    # get the percentages of different labels of the target
    print('Target details in training data:')
    print(application_train['TARGET'].value_counts(normalize=True))
    print('\n')

    # plot the counts of different labels of the target
    print('Plotting \"Target Analysis in Training data\"')
    plt.figure(figsize=(5, 5))
    plt.title('Target Analysis in Training data')
    sns.barplot(x=application_train['TARGET'].value_counts().index, y=application_train['TARGET'].value_counts().values)
    plt.xlabel('Target label')
    plt.ylabel('Count')
    plt.savefig('Target_Analysis_in_Training_data.png', bbox_inches='tight')


def analyse_training_data(application_train, application_test):
    """

    :param application_train: the training data set
    :param application_test: the testing data set
    """

    class EDAUnitTest(unittest.TestCase):

        def train_test_consistency(self):
            self.assertEqual(len(set(application_train.columns).difference(set(application_test.columns))), 1)

    unit_test_object = EDAUnitTest()

    print('Training data basic statistics:')
    print('\n')
    get_basic_statistics(application_train)
    print('\n')
    print('Testing data basic statistics:')
    print('\n')
    get_basic_statistics(application_test)
    print('\n')
    unit_test_object.train_test_consistency()  # to make sure that train and test data columns are consistent except for the target column
    print('Snapshot of the training data:')
    print(application_train.head())
    print('\n')
    analyse_target_label(application_train)
    print('\n')
    print('Checking missing values in training data:')
    calculate_missing_values_total_and_percentage(application_train)


def plot_default_percentage(data, feature, path):
    """

    :param data: the data set
    :param feature: the feature we want to check against the target
    :param path: path to save the plots
    """
    # taking the mean gives the same percentage of people defaulting as it's taking the ratio of the ones
    temp = data[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    temp.describe()
    temp.sort_values(by='TARGET', ascending=False, inplace=True)
    plt.clf()
    if feature == 'ORGANIZATION_TYPE':
        plt.figure(figsize=(15, 5))
    else:
        plt.figure(figsize=(5, 5))
    sns.barplot(x=feature, y='TARGET', data=temp, order=temp[feature])  # order for plotting as categories, not as int
    title = 'Target vs. ' + feature
    plt.title(title, weight='bold', size=14)
    plt.xticks(rotation=90)
    plt.savefig(path + title.replace(' ', '_') + '.png', bbox_inches='tight')


def analyse_categorical_feature_against_target(data, feature, path):
    """

    :param data: the data set
    :param feature: the feature we want to check against the target
    :param path: path to save the plots
    """

    print(data[feature].value_counts(normalize=True))
    plot_default_percentage(data, feature, path)


def plot_numerical_feature_distribution(data, feature, path):
    """

    :param data: the data set
    :param feature: the feature we want to check against the target
    :param path: path to save the plots
    """

    plt.clf()
    data.describe()
    sns.distplot(data[feature].dropna(), kde=False)
    plt.xticks(rotation=90)
    title = 'Density of ' + feature
    plt.title(title, weight='bold', size=14)
    plt.xticks(rotation=90)
    plt.savefig(path + title.replace(' ', '_') + '.png', bbox_inches='tight')
