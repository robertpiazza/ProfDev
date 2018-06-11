# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:45:12 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

"Datacamp-Supervised Learning with Scikit-learn: Fine Tuning Your Model

#In \[([1-9]*)\]: 
#[0-9]* XP
#%%How good is your model?

##Classification metrics
#● Measuring model performance with accuracy:
#● Fraction of correctly classified samples
#● Not always a useful metric

##Class imbalance example: Emails
#● Spam classification
#● 99% of emails are real; 1% of emails are spam
#● Could build a classifier that predicts ALL emails as real
#● 99% accurate!
#● But horrible at actually classifying spam
#● Fails at its original purpose
#● Need more nuanced metrics

#Diagnosing classification predictions
#● Confusion matrix 

#confusion matrix
#true positive-tp
#true negative-tn
#false positive-fp
#false negative-fn
#Accuracy = (tp+tn)/(tp+tn+fp+fn)
#Precision (positive predictive value/PPV)= tp/(tp+fp)
#Recall (sensitivity, hit rate or true positive rate) = tp/(tp+fn)
#F1 Score (Harmonic Mean of precision and recall) = 2* (Precision*Recall)/(Precision + Recall)
#high precision = low false positive rate
#high recall = classifier predicted most positive or spam emails correctly

##Confusion matrix in scikit-learn

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y,
 ...: test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))
[[52 7]
 [ 3 112]]
print(classification_report(y_test, y_pred))
 precision recall f1-score support
 0 0.95 0.88 0.91 59
 1 0.94 0.97 0.96 115
avg / total 0.94 0.94 0.94 174

#%%Metrics for classification

#PIMA indians data set

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%Logistic regression and the ROC curve - used for classification- not regression

#Logistic regression for binary classification
#● Logistic regression outputs probabilities
#● If the probability ‘p’ is greater than 0.5:
#● The data is labeled ‘1’
#● If the probability ‘p’ is less than 0.5:
#● The data is labeled ‘0’ 

##Linear decision boundary

#Logistic regression in scikit-learn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

##Probability thresholds
#● By default, logistic regression threshold = 0.5
#● Not specific to logistic regression
#● k-NN classifiers also have thresholds
#● What happens if we vary the threshold?

##The ROC curve- teh receiver operating characteristics curve

##Plotting the ROC curve

from sklearn.metrics import roc_curve
y_pred_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate’)
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show();

logreg.predict_proba(X_test)[:,1]


#%%Building a logistic regression model

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#%%Plotting an ROC curve

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#%%Precision-recall Curve

#Precision and recall do not take true negatives into consideration

#%%Area under the ROC curve

##Area under the ROC curve (AUC)
#● Larger area under the ROC curve = better model

#AUC in scikit-learn

from sklearn.metrics import roc_auc_score
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y,
 ...: test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score(y_test, y_pred_prob)
#Out[6]: 0.997466216216

#AUC using cross-validation

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(logreg, X, y, cv=5,
 ...: scoring='roc_auc')
print(cv_scores)
[ 0.99673203 0.99183007 0.99583796 1. 0.96140652]


#%%AUC computation

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5,scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


#%%Hyperparameter tuning

#Hyperparameter tuning
#● Linear regression: Choosing parameters
#● Ridge/lasso regression: Choosing alpha
#● k-Nearest Neighbors: Choosing n_neighbors
#● Parameters like alpha and k: Hyperparameters
#● Hyperparameters cannot be learned by fi!ing the model

##Choosing the correct hyperparameter

#Try a bunch of different hyperparameter values
#● Fit all of them separately
#● See how well each performs
#● Choose the best performing one
#● It is essential to use cross-validation

##Grid search cross-validation

##GridSearchCV in scikit-learn

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)
knn_cv.best_params_
#Out[6]: {'n_neighbors': 12}
knn_cv.best_score_
#Out[7]: 0.933216168717
#%%Hyperparameter tuning with GridSearchCV

#here we're not splitting into test and train set to focus on grid search

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


#%%Hyperparameter tuning with RandomizedSearchCV

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

#Note that randomized only saves computational time, does not give a better answer than GridCV



#%%Hold-out set for final evaluation

##Hold-out set reasoning
#● How well can the model perform on never before seen data?
#● Using ALL data for cross-validation is not ideal
#● Split data into training and hold-out set at the beginning
#● Perform grid search cross-validation on training set
#● Choose best hyperparameters and evaluate on hold-out set


#%%Hold-out set reasoning

#You want to be absolutely certain about your model's ability to generalize to unseen data.

#%%Hold-out set in practice I: Classification

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


#%%Hold-out set in practice II: Regression

#use elastic net: a*L1 + b*L2 or l1_ratio parameter

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split


# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train,y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

