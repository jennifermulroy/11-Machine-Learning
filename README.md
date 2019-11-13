# Applying Machine-Learning Models to Predict Credit Risk 

In this assignment, several machine-learning models were used to predict credit risk using free data from LendingClub. Credit risk poses an inherently imbalalanced classification problem that was addressed by deploying various resampling methods to the data. Once the data was resampled, `LogisticRegression` from Scikit-learn library was used to compare the performance of each model.  

LendingClub Data Analysis 
------

Before running the algorithms, the data from LendingClub was cleaned and prepared for analysis. This included dropping null values and columns, converting integer values to numerical data types, and encoding strings using get_dummies(). Once the data was in proper form, the features and target outcome data were placed in new dataframes and further split between testing and training. The target values, low risk and high risk, returned an expected imbalanced problem. Within the dataset, there were 68,470 high risk values verse 347 low risk values. To address this issue, multiple resampling models were tested and compared to determine which algorithm results in the best performance based on balanced accuracy score, recall score, and geometric mean score.  Below is a summary of results. 

Convert the target column values to `low_risk` and `high_risk` based on their values

`x = {'Current': 'low_risk'}   
df = df.replace(x)`

`x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)`

```from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)```

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

##### Balanced Random Forest Classifier Accuracy Score: 78.0%

![forest](images/forester.png)

The top three features: 

![features](images/features.png)

##### Easy Ensemble Classifier Accuracy Score: 93.0%

![easy](images/easy.png)

In conclusion, the Easy Ensemble Classifier had the best balanced accuracy score, recall score, and geometric score. 



