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
 
Before running the algorithms, the data from LendingClub was cleaned and prepared for analysis.  This included dropping null values and columns, and encoding categorical data using `get_dummies()`.  

The target outcome column data, that would be used in the analysis, was converted  to `low_risk` if the loan status was flagged as `Current`  and `high_risk` if the loan status was `Late (31-120 days)`, `Late (16-30 days)`, `Default`, or `In Grace Period`. 
  
The features and target outcome data were placed in new separate dataframes that would be further divided into training and testing data.  

The target values, low risk and high risk, returned an expected imbalanced problem. Within the dataset, there were 68,470 low risk values verse 347 high risk values. 

![imbalance](images/imbalance.png) 

From there, the data was split into training and testing data using `train_test_split` from Scikit-learn library. 

```X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)```

The y_train data further indicates an imbalanced classification. 

![targetimbalance](images/targetimbalance.png)

The training data was resampled: 

- ###### Naive Random, Oversampler

![naive_plot](images/naive_plot.png)


- ###### SMOTE, Oversampler

![smote_plot](images/smote_plot.png)


- ###### Cluster Centroids, Undersampler

![cluster_plot](images/cluster_plot.png)


- ###### SMOTEENN, Combination

![smoteenn_plot](images/smoteenn_plot.png)



1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.
2. Calculate the `balanced accuracy score` from `sklearn.metrics`.
3. Calculate the `confusion matrix` from `sklearn.metrics`.
4. Print the `imbalanced classification report` from `imblearn.metrics`.

model->fit->predict

##### Naive Random Oversampling Balanced Accuracy Score: 70.0%

![naive](images/naive.png)

##### SMOTE Oversampling Balanced Accuracy Score: 72.0%

![smote](images/smote.png)


##### Cluster Centroids Undersampling Balanced Accuracy Score: 65.0%

![cluster](images/cluster.png)


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



