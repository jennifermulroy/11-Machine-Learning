# Applying Machine-Learning Models to Predict Credit Risk 

In this assignment, several machine-learning models were used to predict credit risk using free data from LendingClub. 

Credit risk poses an inherently imbalanced classification problem, the number of observations belonging to a low credit risk class tends to be significantly higher compared to a high credit risk class. Conventional machine learning algorithms do not perform well with class disruption and therefore, may produce biased and inaccurate results with imbalanced data.   

In the first part of the analysis, resampling algorithms are used to address the class imbalance problem by oversampling, undersampling, and using a combination approach on the training data. With the resampled data, `LogisticRegression` from Scikit-learn library was used to build logistic regression classifier models. The performance of each model was then evaluated. 

In the second part of the analysis, two ensemble learning methods were analyzed. Ensemble methods are meta-algorithms combining several machine learning techniques into one predictive model. 


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

The target outcome column data, credit risk, was set on loan status. The data within this column was converted  to `low_risk` if the loan status was flagged as `Current`  and `high_risk` if the loan status was `Late (31-120 days)`, `Late (16-30 days)`, `Default`, or `In Grace Period`. 
  
The features and target outcome data were placed in new separate dataframes, to be used as X and y inputs into the regression model.  

The target values, low risk and high risk, returned an expected imbalanced problem. Within the dataset, there were 68,470 low risk values verse 347 high risk values. 

![imbalance](images/imbalance.png) 

From there, the featues and target outcome data was split into training and testing data using `train_test_split` from Scikit-learn library. 

```X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)```

The y_train data again shows imbalanced classification. 

![targetimbalance](images/targetimbalance.png)

Resampling Algorithms, Training the Logistic Regression Model 
------

The `X_train` and `y_train` data was resampled before using as inputs into the regression model. 

After running the Naive Random and SMOTE Oversampling algorithms, the number of `high_risk` and `low_risk` were equal at 51366 observations. The Cluster Centroids undersampler produced an equal number of 246 observations. And the SMOTEENN combination resampler produced 51359 `high_risk` and 46660 `low_risk` observations. 

Using these resampled training datasets, the models were fit with the `logistic regression classifier` and used to predicte outcomes using the test data. 

- ##### Naive Random, Oversampler

![naive_plot](images/naive_plot.png)


- ##### SMOTE, Oversampler

![smote_plot](images/smote_plot.png)


- ##### Cluster Centroids, Undersampler

![cluster_plot](images/cluster_plot.png)


- ##### SMOTEENN, Combination

![smoteenn_plot](images/smoteenn_plot.png)



Model Performance Analysis 
----

To evaluate the performance of the models, a balanced accuracy score, confusion matrix, and imbalanced classification report were generated to analyze. 

A confusion matrix illustrates the performance of a classification model by comparing actual results with the model's predicted results. From the confusion matrix, precision and recall ratios can be measured. 

Precision score is the ratio of correctly predicted positive observations to the total predicted positive observations. Recall is the ratio of correctly predicted positive observations to all observations in the positive class. F1 score is the weighted average of Precision and Recall. 

Total scores are weighted by each class and can be misleading with imbalanced classification data as the overdominate class will carry a higher weight.  Given the imbalanced classification in the test data, more heavily weighted towards low credit risk observations, breaking down how well the model predicted each class provides additional insight.   

From the results summarized below, all four resampling alogorithms had very low precision rates classifying observations as high credit risk. Each model scored fairly well in identifying above 70% of actual high credit risk observations based on the high risk recall scores, but misclassified low credit risk observations as high credit risk, resulting in a low precision rate for high risk. 

On the flip side, precision scores were very high for low risk as the models all performed well in correctly classifying low credit risk out of the total low credit risk the model identified. 

The recall scores were all similar, except for Cluster Centroids. Undersampling the data improved the recall score of high risk but offset by a lower recall of low risk. 

#### *In conclusion, all models performed fairly well in correctly classifying high credit risk and low credit risk observations, but all models had high error rates misclassifying low credit risk as high credit risk. SMOTE marginally performed best across all metrics. 


|  Resampling Algorithms    | Precision Score / High Risk       | Precision Score / Low Risk | Recall / High Risk | Recall / Low Risk
| -------------             |:-------------:                    | -----:                     |   ----             | ---- 
| Naive Random              | 0.01                              |1.00                        | 0.70               | 0.69      
| SMOTE                     | 0.02                              |1.00                        | 0.71               | 0.73              
| Cluster Centroids         | 0.01                              |1.00                        | 0.82               | 0.47
| SMOTEENN                  | 0.01                              |1.00                        | 0.74               | 0.65               



#### Total Score Summary

|  Resampling Algorithms    | Balanced Accuracy       | Recall Score|F1 Score     |   Geometric Mean Score |
| -------------             |:-------------:          | -----:      | -----:      |   ---                  |
| Naive Random              | 70.0%                   |0.69         |0.81         | 0.70                   |
| SMOTE                     | 72.0%                   |0.73         |0.84         | 0.72                   |
| Cluster Centroids         | 65.0%                   |0.48         |0.64         | 0.62                   |
| SMOTEENN                  | 69.0%                   |0.65         |0.78         | 0.69                   |



Imbalanced Classification Reports: 

![naive](images/naive.png)


![smote](images/smote.png)


![cluster](images/cluster.png)


![smoteenn](images/smoteenn.png)



## Ensemble Learning

For the Ensemble learning analysis, `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` with 100 estimators used to build out both models using the same training and testing data. 

The Balanced Random Forest and Easy Ensemble had similar low precision rates for high risk but Easy Ensemble performed very well across all other metrics.  

|  Ensemble Classifiers     | Precision Score / High Risk       | Precision Score / Low Risk | Recall / High Risk | Recall / Low Risk
| -------------             |:-------------:                    | -----:                     |   ----             | ---- 
| Balanced Random Forest    | 0.09                              |1.00                        | 0.66               | 0.89      
| Easy Ensemble             | 0.09                              |1.00                        | 0.92               | 0.94 

#### Total Score Summary

|  Resampling Algorithms    | Balanced Accuracy       | Recall Score|F1 Score     |   Geometric Mean Score |
| -------------             |:-------------:          | -----:      | -----:      |   ---                  |
| Balanced Random Forest    | 78.0%                   |0.89         |0.93         | 0.77                   |
| Easy Ensemble             | 93.0%                   |0.94         |0.97         | 0.93                   |


The top three features from the Balanced Random Forest:

Total principal received, payments received to date for portion of total amount funded by investors, and interest received to date

![features](images/features.png)


![forest](images/forester.png)

![easy](images/easy.png)




