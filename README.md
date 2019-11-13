# Applying Machine-Learning Models to Predict Credit Risk 

In this assignment, several machine-learning models were used to predict credit risk using free data from LendingClub. Credit risk poses an inherently imbalalanced classification problem. This was first addressed by deploying various resampling algorithms to the data, and then using `LogisticRegression` from Scikit-learn library to build logistic regression classifiers. In the second part of the analysis, ensemble classifiers were used to predict loan risk. 

##### Resampling Algorithms: 
- `Naive Random Oversampler`
- `SMOTE`
- `Cluster Centroids`
- `SMOTEENN` 

##### Ensemble Learning Methods: 
- `Balanced Random Forest Classifier`
- `Easy Ensemble AdaBoost Classifier`


LendingClub Data Analysis 
------

The LendingClub data included 
Before running the algorithms, the data from LendingClub was cleaned and prepared for analysis.  This included dropping null values and columns, converting integer values to numerical data types, and encoding categorical data using `get_dummies()`.  

The target data  to `low_risk` and `high_risk` based on their values

`x = {'Current': 'low_risk'}   
df = df.replace(x)`

`x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)`
  
Once the data was in proper form, the features and target outcome data were placed in new dataframes  

The target values, low risk and high risk, returned an expected imbalanced problem. Within the dataset, there were 68,470 high risk values verse 347 low risk values. 

![imbalance](images/imbalance.png) 

To address this issue, multiple resampling models were tested and compared to determine which algorithm results in the best performance based on balanced accuracy score, recall score, and geometric mean score.  Below is a summary of results. 

1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.
2. Calculate the `balanced accuracy score` from `sklearn.metrics`.
3. Calculate the `confusion matrix` from `sklearn.metrics`.
4. Print the `imbalanced classification report` from `imblearn.metrics`.

```from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)```

- Split the Data into Training and Testing 


model->fit->predict
### Naive Random Oversampler and SMOTE algorithms to oversample the data 

##### Naive Random Oversampling Balanced Accuracy Score: 70.0%

![naive](images/naive.png)

##### SMOTE Oversampling Balanced Accuracy Score: 72.0%

![smote](images/smote.png)


### Cluster Centroids algorithm to undersample the data 

##### Cluster Centroids Undersampling Balanced Accuracy Score: 65.0%

![cluster](images/cluster.png)


### SMOTEENN algorithm to over and under-sample the data 

##### SMOTEENN Combination Balanced Accuracy Score: 69.0%

![smoteenn](images/smoteenn.png)

|  Algorithms               | Balanced Accuracy Score | Recall Score| Geometric Mean Score |
| -------------             |:-------------:          | -----:      |    ---               |
| Naive Random Oversampler  | 70.0%                   |             |                      |
| SMOTE Oversampling        | 72.0%                   |        |                      |
| Cluster Centroids         | 65.0%                   |          |                      |
| SMOTEENN                  | 69.0%                   |           |                      |



In conclusion, the SMOTE oversampler model had the best balanced accuracy score, recall score, and geometric mean score.


## Ensemble Learning

For the ensemble learners, use 100 estimators for both models

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.
2. Calculate the balanced accuracy score from `sklearn.metrics`.
3. Print the confusion matrix from `sklearn.metrics`.
4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.
5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.

##### Balanced Random Forest Classifier Accuracy Score: 78.0%

![forest](images/forester.png)

The top three features: 

![features](images/features.png)

##### Easy Ensemble Classifier Accuracy Score: 93.0%

![easy](images/easy.png)

In conclusion, the Easy Ensemble Classifier had the best balanced accuracy score, recall score, and geometric score. 



