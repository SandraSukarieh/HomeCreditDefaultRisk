import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

pd.options.mode.chained_assignment = None  # default='warn'


def get_train_test_data_for_model(original_training, original_testing):
    """

    :param original_training: the original training data
    :param original_testing: the original testing data
    :return:
    """
    train_data = original_training.drop('TARGET', axis=1)
    train_data = train_data.drop('SK_ID_CURR', axis=1)
    labels = original_training['TARGET']
    features = train_data.columns.tolist()
    test_data = original_testing.drop('SK_ID_CURR', axis=1)
    train_data, test_data = impute_and_normalize_data(train_data, test_data)
    return train_data, labels, features, test_data


def impute_and_normalize_data(train_data, test_data):
    """

    :param train_data: the train data to fit and apply imputation and scaling
    :param test_data: the test data to be imputed and scaled
    :return:
    """
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_data)
    train_data = imputer.transform(train_data)
    test_data = imputer.transform(test_data)

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_data)
    train_data = feature_scaler.transform(train_data)
    test_data = feature_scaler.transform(test_data)

    return train_data, test_data



def fitting_baselines_cross_validation(train_data, labels):
    """

    :param train_data: the train data after imputation and normalization
    :param labels: the labels (the target column labels)
    """
    classification_models = {'Logistic': LogisticRegression(C=0.0001, max_iter=1000),
                             'DecisionTree': DecisionTreeClassifier(), 'RandomForest': RandomForestClassifier()}
    # we change the C value in the logistic regression to prevent it from overfitting
    results = dict()

    for model in classification_models:
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        model_result = cross_val_score(classification_models[model], train_data, labels, cv=kfold, scoring='roc_auc')
        results[model] = model_result
        print(model + ": = ", mean(model_result))


def fitting_rf_classifier(train_data, labels, test_data, features, application_test, output_name, plot_name, features_plot):
    """

    :param train_data: the train data to fit the model
    :param labels: the label of the train data
    :param test_data: the test data we need to predict for
    :param features: the columns name of the data set
    :param application_test: the original test data
    :param output_name: name of the csv file to save the predictions
    :param plot_name: the name of the features importance plot
    :param features_plot: number of features we want to plot the importance of
    """

    print('Retraining the RF model')
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
    rf_classifier.fit(train_data, labels)

    plotting_features_importance(rf_classifier, features, plot_name, features_plot)

    print('\n')
    print('Predicting for the test data')
    test_probabilities = rf_classifier.predict_proba(test_data)[:, 1]

    # as mentioned in the sample submission of kaggle
    prediction_result = application_test[['SK_ID_CURR']]
    prediction_result['TARGET'] = test_probabilities

    prediction_result.to_csv(output_name+'.csv', index=False)


def plotting_features_importance(classifier, features, plot_name, features_plot):
    """

    :param classifier: the classifier for which we want to check the importance of features
    :param features: the features of the data set used to train the classifier
    :param plot_name: the name of the features importance plot
    :param features_plot: number of features we want to plot the importance of
    """

    feature_importance_values = classifier.feature_importances_
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': feature_importance_values})

    feature_importances['Normalized_Importance'] = feature_importances['Importance'] / feature_importances[
        'Importance'].sum()

    feature_importances.sort_values(by='Normalized_Importance', ascending=False, inplace=True)

    plotting = feature_importances.head(features_plot)

    print('\n')
    print('Plotting ' + plot_name)
    plt.clf()
    plt.figure(figsize=(30, 30))
    title = plot_name + '_Top_' + str(features_plot)
    plt.title(title)
    plotting.plot(x='Feature', y='Normalized_Importance', kind='bar')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.xticks(rotation=90)
    plt.savefig(title+'.png', bbox_inches='tight')


def validating_features_selection(train_data, labels):
    """

    :param train_data: the edited data set we want to validate the classifier performance with
    :param labels: the labels of the edited data set
    """
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    model_result = cross_val_score(RandomForestClassifier(), train_data, labels, cv=kfold, scoring='roc_auc')
    print("current model auc score = ", mean(model_result))





