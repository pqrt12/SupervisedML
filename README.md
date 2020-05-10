# SupervisedML
# Challenge
This challenge uses a given dataset from LendingClub, which is huge and can not be uploaded, to assess credit risks.

In notebook [Resampling.ipynb](https://github.com/pqrt12/SupervisedML/blob/master/Notebook/Resampling.ipynb), different data sampling algorithms are used against the same dataset, then trained with the same logistic regression classifier (from Scikit-learn), and their results are compared. 

All data are fit_trnasform with sklearn.preprocessing LabelEncoder. Three features  
&nbsp;&nbsp;&nbsp;&nbsp;['pymnt_plan', 'hardship_flag', 'debt_settlement_flag']  
are constant, they are removed. The final data have 82 features. Binary coding is also tried, it is not activated in the notebook, however.

The four sampling algorithms studied here are:
+ Naive random oversampling
+ SMOTE oversampling
+ Cluster Centroids undersampling
+ SMOTEENN combined sampling
 
Their trained models' performance are tabulated together at the end of the notebook for easier review. There are three tables:
* High Risk detection results of "precision", "sensitivity", "f1".
* Low Risk detection results of "precision", "sensitivity", "f1". 
* Balanced Accuracy score

It is important to detect the high risk credits. We see Cluster Centroids undersampling yields a better high risk sensitivity in this dataset. We may need more data to confirm the significance.

Notebook [Ensemble.ipynb](https://github.com/pqrt12/SupervisedML/blob/master/Notebook/Ensemble.ipynb) studied two ensemble classifiers. With the same data, after training, prediction is generated, their results are compared. Easy Ensemble AdaBoost Classifier offers much better results. Its high risk detection precision is 9%, sensitivity is 92%, and f1 is 16%. Its low risk detection precision is 100%, sensitivity is 94%, and f1 is 97%. Its accuracy is 93%. This classifier is recommended.

