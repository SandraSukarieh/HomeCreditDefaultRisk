# HomeCreditDefaultRisk
an initial solution to "Home Credit Default Risk Challenge" by Kaggle, implemented with Python 3.7.


## Requirements
- Python To install the dependencies used in the code, you can use the **requirements.txt** file as follows:


```sh
$ pip3 install -r requirements.txt
```

## Running the code
- Run the ``` main.py ``` as follows:

```sh
$ python3 main.py
```

The mandatory arguments it takes is:
-  ``` --path ```: path to the CSV data files of the challenge (string format).

How I ran the code and saved the output into ``` output.txt ```:

```sh
$ python3 main.py -p 'path\to\csv\data\files' > output.txt
```

## Solution highlights

1- The solution starts with exploratory data analysis step, focusing on the training (with the TARGET as the output) and testing data:

- Getting basic statistics about the training and testing data, e.g. dimensions, data types, etc.
- Checking the categorical features of the training data and analyzing their values against the TARGET, and plotting those relationships as bar plots.
- Checking the numerical features of the training data and plotting their densities as histograms.
- Checking how many columns in the training data contains missing values, and what are those percentages.
- Checking for outliers using the mean and standard deviation, then replace those values with NAN to be filled later.
- Checking the correlation between the features and the TARGET, and plotting corresponding information.

2- We encode the categorical features depending on:
- Label encoding for binary features.
- One-hot encoding for features with more than two labels.

3- We fit the baseline model as following:
- We impute missing values using the "median" strategy.
- We normalize the values to be in [0, 1].
- We use cross-validation with 5 folds and the AUC metric to choose the best model among: Logistic Regression/ Decision Tree/ Random Forest (the selected model).
- We fit the random forest model on the train data and then use it to predict the TARGET labels of the test data.
- We plot the top 20 important features for the classifier.
- We save the predictions into 'RF_Baseline_Predictions.csv'.

4- We improve our model using features engineering, and we start with including features from the other data sets, such as Bureau and Previous_Applications:
- We get features that seem interesting from those two data sets.
- We merge those features into the training and testing data.
- We impute and normalize the improved data sets.
- We check if the model's performance gets better with cross-validation.
- We re-fit the random forest classifier using the improved training data and predict the TARGET for the testing data.
- We plot the top 20 important features for the classifier.
- We save the predictions into 'RF_Improved_Predictions.csv'.

5- We continue with features engineering by features selection:
- We use Anova F-test to get the most n important features for our classifier (here we use 30 features).
- We update the training and testing data sets based on the selected features.
- We impute and normalize the improved data sets.
- We check if the model's performance gets better with cross-validation.
- We re-fit the random forest classifier using the improved training data and predict the TARGET for the testing data.
- We plot the top 20 important features for the classifier.
- We save the predictions into 'RF_Improved_Selected_Features_Predictions.csv'.

The output of this solution (including the found insights and discussions) is available in ``` output.txt ``` in addition to the multiple plots and the final CSV files of predictions.


## Future improvements
When having more resources and time, the following can be done:
- Check 'Domain Knowledge' features by getting a deeper understanding of financial data and use that to extract new features from the ones we already have.
- Add interaction terms between features especially the most relevant ones.
- Raise some features to some power to emphasize their effect in the fitted model.
- Perform a deeper parameter tunning for the selected model.
