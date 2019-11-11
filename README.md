# Applying Machine-Learning Models to Predict Credit Risk 

In this assignment, several machine-learning models were used to predict credit risk using free data from LendingClub. The algorithms used in the analysis were selected to address the inherently imbalanced classification problem of credit risk. For each algorithm, a logistic regression classifier from sklearn.linear_model was used to resample the data. A balanced accuracy score, confusion matrix, and imbalanced classification report was generated to compare results. 

Before running the algorithms, the data from LendingClub was cleaned and prepared for analysis. This included dropping null values and columns, converting integer values to numerical data types, and encoding strings using get_dummies(). Once the data was in proper form, the features and target outcome data were placed in new dataframes, and further split between testing and training. The target values, low risk and high risk, returned an expected imbalanced problem. Within the dataset, there were 68,470 high risk values verse 347 low risk values. TO address this issue, multiple resampling models were tested and compared to determine which algorithm results in the best performance based on balanced accuracy score, recall score, and geometric mean score.  Below is the summary of results. 

> Which model had the best balanced accuracy score?
> Which model had the best recall score?
> Which model had the best geometric mean score?

### Naive Random Oversampler and SMOTE algorithms to oversample the data 


### Cluster Centroids algorithm to undersample the data 


### SMOTEENN algorithm to over and under-sample the data 




### Naive Random Oversample Summary of Results 
Accuracy Score:
Confusion Matrix:
Imbalanced Classification Report: 
