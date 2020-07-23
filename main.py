import pandas as pd
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from argparse import ArgumentParser

from EDA_functions import analyse_training_data, analyse_categorical_feature_against_target, plot_numerical_feature_distribution, get_basic_statistics
from encoding_categorical_data_functions import encode_categorical_features
from correlation_functions import check_correlation_with_target_train
from checking_outliers_functions import check_outliers, replace_outliers_with_nan_for_feature
from model_fitting_functions import get_train_test_data_for_model, fitting_baselines_cross_validation, fitting_rf_classifier, validating_features_selection
from features_engineering_functions import get_interesting_info_from_bureau, get_interesting_info_from_previous_applications, feature_selection_anova_f_value


class GeneralUnitTest(unittest.TestCase):

    def check_removing_outliers_employment_train(self):
        self.assertTrue(application_train['DAYS_EMPLOYED'].max() < outlier_value)

    def check_removing_outliers_employment_test(self):
        self.assertTrue(application_test['DAYS_EMPLOYED'].max() < outlier_value)

    def check_alignment_success(self):
        self.assertTrue(application_test.shape[1] + 1 == application_train.shape[1])

    def test_if_no_missing_values(self):
        self.assertEqual(sum_of_missing_values, 0)


unit_test_object = GeneralUnitTest()


# create folders to save the plots -----------------------------------------------------

if not os.path.isdir('correlation_plots'):
    print('creating a folder for correlation plots...')
    os.mkdir('correlation_plots')

if not os.path.isdir('training_data_categorical_features'):
    print('creating a folder for categorical features against target plots...')
    os.mkdir('training_data_categorical_features')

if not os.path.isdir('training_data_numerical_features'):
    print('creating a folder for densities of numerical features...')
    os.mkdir('training_data_numerical_features')

print('\n')

# parse arguments -----------------------------------------------------------------------


def parse_config():
    my_args = ArgumentParser('HomeCreditDefaultRisk')
    my_args.add_argument('-p', '--path', dest='path', help='path to data files')
    return my_args.parse_args(sys.argv[1:])


conf = parse_config()


# load data into data frames ------------------------------------------------------------

path = conf.path
application_train = pd.read_csv(path+'application_train.csv')
application_test = pd.read_csv(path+'application_test.csv')
bureau = pd.read_csv(path+'bureau.csv')
previous_application = pd.read_csv(path+'previous_application.csv')


print('---------------------------- General Train Data Analysis ------------------------------------------------------')

print('\n')
analyse_training_data(application_train, application_test)
print('\n')

print('From the first analysis of the training data, we notice the following:')
print('======================================================================')
print('1- training and testing data are identical, except for the target column which is the output label.')
print('2- the majority of clients repaid the load (around 92%), while around 8% defaulted.')
print('3- there are missing values in 67 columns in the training data that needs to be processed later.')
print('\n')

print('------------------ Train Data Categorical and Integer Features Analysis ---------------------------------------')

print('\n')
print('categorical features of the training data set:')
print(application_train.select_dtypes('object').apply(pd.Series.nunique, axis=0).index.tolist())
print('\n')
# here we add the features we want to analyse. We chose this sample of features just for simplicity
features_list = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
                 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'HOUSETYPE_MODE', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
                 'FLAG_PHONE', 'REGION_RATING_CLIENT', 'FLAG_DOCUMENT_2']

print('Here we print the percentage og each value of the feature')
print('check plots in the corresponding folder')
print('\n')
for feature in features_list:
    analyse_categorical_feature_against_target(application_train, feature, 'training_data_categorical_features/')
    print('\n')
print('\n')

print('We notice the following insights:')
print('=================================')
print('1- Clients with cash loans have a higher chance to default than revolving loans.')
print('2- The majority of clients are females.')
print('3- Males have a higher chance to default (around 10%).')
print('4- Car owners are more likely to repay the loans.')
print('5- Businessmen and students always repaid the loan.')
print('7- Unemployed people or the ones who in maternity leaves are more likely to default.')
print('8- Lower secondary are the most who default.')
print('9- Single people and civil married people are the most people to default.')
print('10- Clients with more children/family members are more likely to default.')
print('11- The majority of clients did not provide document 2 (around 99%).')
print('12- Clients who did provide document 2 tend more to default (Unfortunately, we lack more info about it).')
print('\n')

print('-------------------------------------- Numerical Features Analysis --------------------------------------------')

print('\n')
print('numerical features of the training data set:')
print(application_train.select_dtypes('float').apply(pd.Series.nunique, axis=0).index.tolist())
print('\n')
print('check plots in the corresponding folder')
features_list = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE',
                 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR',
                 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE']
for feature in features_list:
    plot_numerical_feature_distribution(application_train, feature, 'training_data_numerical_features/')
print('\n')

print('We notice that densities are reasonable for the selected features, except for \'DAYS_EMPLOYED\'.')
print('We check \'DAYS_EMPLOYED\' further in case it contains outliers.')
print('\n')

print('------------------------------------------- Checking Anomalies ------------------------------------------------')

print('\n')
print((application_train['DAYS_BIRTH'].apply(lambda x: x / -365)).describe())  # values are negative in the data set
print('\n')
print((application_train['DAYS_REGISTRATION'].apply(lambda x: x / -365)).describe())
print('\n')
print((application_train['DAYS_EMPLOYED'].apply(
    lambda x: x / 365)).describe())  # values are both positive and negative in the data set
print('\n')

print('The max value of days of employment do not make sense as it equals 1000 years.')
print('We check the target labels of those entities to decide if we drop those outliers or fix them in some way.')
print('\n')

outliers_lower, outliers_upper = check_outliers(application_train, 'DAYS_EMPLOYED')

print('Lower outliers for DAYS_EMPLOYED')
print(outliers_lower)
print('\n')
print('Higher outliers for DAYS_EMPLOYED')
print(outliers_upper)

print('\n')
print('Outliers checker returned the same suspicious value as it is further than 2 std from the mean.')
print('\n')

anomalies_entities = application_train[application_train['DAYS_EMPLOYED'] == application_train['DAYS_EMPLOYED'].max()]
print(anomalies_entities['TARGET'].mean())
print('\n')

print('5% of the anomalies entities default, which is very interesting, so we will not remove them.')
print('Instead, we will assign a NAN value to be filled later when dealing with missing values.')
print('We can, however, add an additional column to indicate whether the value was originally an anomaly or not.')
print('\n')

outlier_value = replace_outliers_with_nan_for_feature(application_train, 'DAYS_EMPLOYED', True)
unit_test_object.check_removing_outliers_employment_train()  # to make sure that the outlier value is removes

# check if the problem is also available in the test data
print(application_test['DAYS_EMPLOYED'].apply(lambda x: x / 365).describe())
print('\n')

print('We notice that the same issue appears in the test data, so we fix it in the same way.')
print('It is important to keep train and test data consistent and get the same changes done on both of them.')
print('\n')

outlier_value = replace_outliers_with_nan_for_feature(application_test, 'DAYS_EMPLOYED', True)
unit_test_object.check_removing_outliers_employment_test()

print('We can use the outlier checker for all the features and analyse further if we get some results.')
print('Here we only mention the suspicious cases we found for simplicity.')

outliers_lower, outliers_upper = check_outliers(application_train, 'AMT_INCOME_TOTAL')

print('Lower outliers for AMT_INCOME_TOTAL')
print(outliers_lower)
print('\n')
print('Higher outliers for AMT_INCOME_TOTAL')
print(outliers_upper)

print('\n')
print('We also notice some weird results for AMT_INCOME_TOTAL, so we check them further.')
print('\n')

print(application_train['AMT_INCOME_TOTAL'].describe())
print(application_train['AMT_INCOME_TOTAL'].max())
print('\n')
print('We notice that the max value is really big and a bit suspicious, so we deal with it as what is previously done.')

outlier_value = replace_outliers_with_nan_for_feature(application_train, 'AMT_INCOME_TOTAL', True)
unit_test_object.check_removing_outliers_employment_train()  # to make sure that the outlier value is removed

print('We notice that the problem is not available in the test data:')
print(application_test['AMT_INCOME_TOTAL'].max())
print('\n')

print('------------------------------------------- Correlation -------------------------------------------------------')

print('\n')
check_correlation_with_target_train(application_train)
print('\n')

print('We notice the following insights:')
print('=================================')
print('1- The density of defaulting drops when the client is older and this confirms the positive correlation we found.')
print('2- EXT_SOURCE_3 (the feature with the highest negative correlation with target) has the biggest difference.')
print('3- In general, the client\'s chance to default decreases when the value of the feature EXT_SOURCE increases.')
print('\n')

print('-------------------------------------- Categorical Features Encoding ------------------------------------------')

print('\n')
print('Classification models cannot deal with categorical data, so we need to encode them.')
print('We use \'Label encoding\' for binary features, and \'One-hot encoding\' for features with more than 2 labels.')
print('\n')

application_train, application_test = encode_categorical_features(application_train, application_test)
print('\n')
print('training data dimensions after encoding: ', application_train.shape)
print('testing data dimensions after encoding: ', application_test.shape)
print('\n')

print('The difference between test and train columns should be 1 (due to the target), but the difference now is 3.')
print('This means that some columns have more labels in the training data than in the test data.')
print('We align the two data sets with \'inner\' type to only keep values present in both of them.')
print('It is important to save the target column and re-add it to the train data, as the alignment would remove it.')
print('\n')

target_column = application_train['TARGET']
application_train, application_test = application_train.align(application_test, join='inner', axis=1)
application_train['TARGET'] = target_column

print('train data dimensions after encoding and aligning: ', application_train.shape)
print('test data dimensions after encoding aligning: ', application_test.shape)

unit_test_object.check_alignment_success()  # to make sure that the alignment process is successful
print('\n')

plt.close('all')

print('--------------------------------------- Baseline Model Fitting ------------------------------------------------')

print('\n')

train_data, labels, features, test_data = get_train_test_data_for_model(application_train, application_test)

sum_of_missing_values = np.isnan(train_data).sum().sum()
unit_test_object.test_if_no_missing_values  # making sure there are no missing values left after imputation

sum_of_missing_values = np.isnan(test_data).sum().sum()
unit_test_object.test_if_no_missing_values

print('Training data shape: ', train_data.shape)
print('Testing data shape: ', test_data.shape)

fitting_baselines_cross_validation(train_data, labels)
print('\n')
print('We notice that the random forest classifier is the best (as expected), so we choose it for predictions.')
print('\n')

fitting_rf_classifier(train_data, labels, test_data, features, application_test, 'RF_Baseline_Predictions', 'RF_Baseline_Feature_Importance', 20)
print('We save the results of prediction in \'RF_Baseline_Predictions.csv\'')
print('\n')

print('----------------------------- Features Engineering 1 - Adding Features ----------------------------------------')

print('\n')
print('checking features of other data sets')
print('\n')
print('Bureau: All client\'s previous credits provided by other financial institutions and reported to Credit Bureau.')
get_basic_statistics(bureau)

print('\n')
print('check the values of active credit as it seems interesting')
print(bureau['CREDIT_ACTIVE'].value_counts())

print('\n')
print('taking the number of previous loans per client and checking how many of them is active might be interesting')
bureau_info = get_interesting_info_from_bureau(bureau)
print(bureau_info.head())

print('-----------------------------------------')

print('Previous applications: All previous applications for Home Credit loans of clients who have loans in our sample.')
get_basic_statistics(previous_application)

print('\n')
print('check the values of previous application status as it seems interesting')
print(previous_application['NAME_CONTRACT_STATUS'].value_counts())

print('\n')
print('taking the count of previous applications per client at Home Credit and checking how many were approved/refused.')
previous_application_info = get_interesting_info_from_previous_applications(previous_application)
print(previous_application_info.head())

print('\n')

print('------------------------------------- Improved Model Fitting - 1 ----------------------------------------------')

print('\n')

application_train_improved = application_train.merge(bureau_info, on='SK_ID_CURR', how='left')
application_train_improved = application_train_improved.merge(previous_application_info, on='SK_ID_CURR', how='left')

application_test_improved = application_test.merge(bureau_info, on='SK_ID_CURR', how='left')
application_test_improved = application_test_improved.merge(previous_application_info, on='SK_ID_CURR', how='left')

application_train_improved = application_train_improved.fillna(0)
application_test_improved = application_test_improved.fillna(0)

train_data_improved, labels, features, test_data_improved = get_train_test_data_for_model(application_train_improved, application_test_improved )

sum_of_missing_values = np.isnan(train_data_improved).sum().sum()
unit_test_object.test_if_no_missing_values  # making sure there are no missing values left after imputation

sum_of_missing_values = np.isnan(test_data_improved).sum().sum()
unit_test_object.test_if_no_missing_values


print('Training data shape: ', train_data_improved.shape)
print('Testing data shape: ', test_data_improved.shape)

print('\n')
validating_features_selection(train_data_improved, labels)
print('We notice that the AUC score is a bit better than the first classifier without the additional features')
print('\n')

fitting_rf_classifier(train_data_improved, labels, test_data_improved, features, application_test_improved, 'RF_Improved_Predictions', 'RF_Improved_Feature_Importance', 20)
print('We save the results of prediction in \'RF_Improved_Predictions.csv\'.')
print('\n')

print('----------------------------- Features Engineering 2 - Features Selection--------------------------------------')

print('\n')
print('Here we check the best n features to use only them in predicting.')
print('As a simple first solution, we use Anova F-test and we only keep 30 features since the features importance plots show a decrease.')
print('\n')

selected_columns = feature_selection_anova_f_value(application_train_improved, 30).tolist()

application_train_selected_columns = application_train_improved.copy()
application_test_selected_columns = application_test_improved.copy()


for column in application_train_selected_columns.columns:
    if column not in selected_columns and column != 'SK_ID_CURR' and column != 'TARGET':
        application_train_selected_columns.drop(column, axis=1, inplace=True)

for column in application_test_selected_columns.columns:
    if column not in selected_columns and column != 'SK_ID_CURR':
        application_test_selected_columns.drop(column, axis=1, inplace=True)

print('------------------------------------- Improved Model Fitting - 2 ----------------------------------------------')

train_data_selected_columns, labels, features, test_data_selected_columns = get_train_test_data_for_model(application_train_selected_columns, application_test_selected_columns)

sum_of_missing_values = np.isnan(train_data_selected_columns).sum().sum()
unit_test_object.test_if_no_missing_values  # making sure there are no missing values left after imputation

sum_of_missing_values = np.isnan(test_data_selected_columns).sum().sum()
unit_test_object.test_if_no_missing_values

print('Training data shape: ', train_data_selected_columns.shape)
print('Testing data shape: ', test_data_selected_columns.shape)

print('\n')
validating_features_selection(train_data_selected_columns, labels)
print('We notice that the AUC score is better than the first improved model that had a lot of features.')
print('\n')

print('We fit our final model that has a set of selected features from multiple data sets')
print('\n')
fitting_rf_classifier(train_data_selected_columns, labels, test_data_selected_columns, features, application_test_selected_columns, 'RF_Improved_Selected_Features_Predictions', 'RF_Improved_Selected_Features_Feature_Importance', 20)
print('We save the results of prediction in \'RF_Improved_Selected_Features_Predictions.csv\'.')
print('\n')