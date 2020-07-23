import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def get_interesting_info_from_bureau(bureau):
    """

    :param bureau: the bureau data frame
    :return: data frame with the extracted important info from bureau
    """
    bureau_ids = bureau['SK_ID_CURR'].unique()  # to get one entity per client

    previous_loans = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count()
    previous_loans.rename(columns={'SK_ID_BUREAU': 'PREVIOUS_LOANS_COUNT'}, inplace=True)

    active_loans = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR', as_index=False)[
        'SK_ID_BUREAU'].count()
    active_loans.rename(columns={'SK_ID_BUREAU': 'ACTIVE_LOANS_COUNT'}, inplace=True)

    bureau_info = pd.DataFrame(bureau_ids, columns=['SK_ID_CURR'])

    bureau_info = bureau_info.merge(previous_loans, on='SK_ID_CURR', how='left')
    bureau_info = bureau_info.merge(active_loans, on='SK_ID_CURR', how='left')

    if bureau_info.isna().sum().sum() != 0:
        bureau_info = bureau_info.fillna(0)

    bureau_info = bureau_info.astype(int)

    return bureau_info


def get_interesting_info_from_previous_applications(previous_application):
    """

    :param previous_application: the previous applications data frame
    :return: data frame with the extracted important info from previous applications
    """
    previous_ids = previous_application['SK_ID_CURR'].unique()  # to get one entity per client

    previous_applications_count = previous_application.groupby('SK_ID_CURR', as_index=False)['SK_ID_PREV'].count()
    previous_applications_count.rename(columns={'SK_ID_PREV': 'PREVIOUS_APPLICATIONS_COUNT'}, inplace=True)

    approved_previous_applications = \
    previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR',
                                                                                             as_index=False)[
        'SK_ID_PREV'].count()
    approved_previous_applications.rename(columns={'SK_ID_PREV': 'APPROVED_PREVIOUS_APPLICATIONS'}, inplace=True)

    refused_previous_applications = \
    previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR',
                                                                                            as_index=False)[
        'SK_ID_PREV'].count()
    refused_previous_applications.rename(columns={'SK_ID_PREV': 'REFUSED_PREVIOUS_APPLICATIONS'}, inplace=True)

    previous_application_info = pd.DataFrame(previous_ids, columns=['SK_ID_CURR'])

    previous_application_info = previous_application_info.merge(previous_applications_count, on='SK_ID_CURR',
                                                                how='left')
    previous_application_info = previous_application_info.merge(approved_previous_applications, on='SK_ID_CURR',
                                                                how='left')
    previous_application_info = previous_application_info.merge(refused_previous_applications, on='SK_ID_CURR',
                                                                how='left')

    if previous_application_info.isna().sum().sum() != 0:
        previous_application_info = previous_application_info.fillna(0)

    previous_application_info = previous_application_info.astype(int)

    return previous_application_info


def feature_selection_anova_f_value(data, features_count):
    """

    :param data: the data set we want to check its top best features
    :param features_count: the number of top features we want to consider
    :return: the top features
    """

    feature_cols = data.columns.drop('TARGET')

    selector = SelectKBest(f_classif, k=features_count)
    X_new = selector.fit_transform(data[feature_cols], data['TARGET'])
    selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                                     index=data.index,
                                     columns=feature_cols)
    selected_columns = selected_features.columns[selected_features.var() != 0]
    return selected_columns
