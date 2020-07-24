import matplotlib.pyplot as plt
import seaborn as sns


def check_correlation_with_target_train(application_train):
    """

    :param application_train: training data set to check its features correlation with the target
    """
    # corr gives a dataframe and we only care about the correlation with the target, so we only take that column
    target_correlation = application_train.corr()['TARGET'].sort_values(ascending=False)
    print('The top 10 features positively correlated with the target are:')
    print('\n')
    print(target_correlation.head(11))  # target is included with itself as the highest value (1)
    print('\n')
    print('The top 10 features negatively correlated with the target are:')
    print('\n')
    print(target_correlation.tail(10))
    print('\n')

    print('Plotting \"Correlation of Age and Target\"')
    plt.clf()
    sns.kdeplot(application_train[application_train['TARGET'] == 1]['DAYS_BIRTH'] / (-365), label="Default",
                color="red")
    sns.kdeplot(application_train[application_train['TARGET'] == 0]['DAYS_BIRTH'] / (-365), label="Repaid",
                color="green")
    plt.xlabel('Client Age')
    plt.ylabel('Density')
    plt.savefig('correlation_plots/Correlation_Age_Target.png', bbox_inches='tight')
    print('\n')

    print('Plotting \"Correlation of EXT_SOURCE and Target\"')

    plt.clf()
    sns.kdeplot(application_train[application_train['TARGET'] == 1]['EXT_SOURCE_1'], label="Default", color="red")
    sns.kdeplot(application_train[application_train['TARGET'] == 0]['EXT_SOURCE_1'], label="Repaid", color="green")
    plt.xlabel('EXT_SOURCE_1')
    plt.ylabel('Density')
    plt.savefig('correlation_plots/Correlation_EXT_SOURCE_1_Target.png', bbox_inches='tight')

    plt.clf()
    sns.kdeplot(application_train[application_train['TARGET'] == 1]['EXT_SOURCE_2'], label="Default", color="red")
    sns.kdeplot(application_train[application_train['TARGET'] == 0]['EXT_SOURCE_2'], label="Repaid", color="green")
    plt.xlabel('EXT_SOURCE_2')
    plt.ylabel('Density')
    plt.savefig('correlation_plots/Correlation_EXT_SOURCE_2_Target.png', bbox_inches='tight')

    plt.clf()
    sns.kdeplot(application_train[application_train['TARGET'] == 1]['EXT_SOURCE_3'], label="Default", color="red")
    sns.kdeplot(application_train[application_train['TARGET'] == 0]['EXT_SOURCE_3'], label="Repaid", color="green")
    plt.xlabel('EXT_SOURCE_3')
    plt.ylabel('Density')
    plt.savefig('correlation_plots/Correlation_EXT_SOURCE_3_Target.png', bbox_inches='tight')



