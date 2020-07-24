import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_features(application_train, application_test):
    """

    :param application_train: the training data set we want to encode its categorical features
    :param application_test: the testing data set we want to encode its categorical features
    """
    # use label encoding for binary features
    le = LabelEncoder()
    for column in application_train:
        if application_train[column].dtype == 'object':
            if len(application_train[column].unique()) <= 2:
                print('Encoding {} with label encoding'.format(column))
                le.fit(application_train[column])
                application_train[column] = le.transform(application_train[column])
                application_test[column] = le.transform(application_test[column])

    # use One-hot encoding for the rest
    application_train = pd.get_dummies(application_train)
    application_test = pd.get_dummies(application_test)

    return application_train, application_test

