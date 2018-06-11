# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:22:18 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

"Datacamp-Supervised Learning with Scikit-learn: Regression

#%%Introduction to regression

#Boston housing data

X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values

##Predicting house value from a single feature

X_rooms = X[:,5]
type(X_rooms), type(y)
#Out[6]: (numpy.ndarray, numpy.ndarray)
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1)

##Plotting house value vs. number of rooms

plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show();

#Plotting house value vs. number of rooms

##Fitting a regression model

import numpy as np
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms),
 ...: max(X_rooms)).reshape(-1, 1)
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space),
 ...: color='black', linewidth=3)
plt.show()

#%%Which of the following is a regression problem?

#A bike share company using time and weather data to predict the number of bikes being rented at any given hour.

#%%Importing data for supervised learning

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

#%%Exploring the Gapminder data

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

df.info()
df.describe()
df.head()


#%%The basics of linear regression

#Regression mechanics
#● y = ax + b
#● y = target
#● x = single feature
#● a, b = parameters of model
#● How do we choose a and b?
#● Define an error function for any given line
#● Choose the line that minimizes the error function

#The loss function
#● Ordinary least squares (OLS): Minimize sum of squares of residuals

#Linear regression in higher dimensions
#● y=a1x1+a2x2+b
#● To fit a linear regression model here:
#● Need to specify 3 variables
#● In higher dimensions:
#    y = a1x1 + a2x2 + a3x3 + anxn + b
#● To fit a linear regression model here:
#● Must specify coefficient for each feature and the variable b
#● Scikit-learn API works exactly the same way:
#● Pass two arrays: Features, and target

##Linear regression on all features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
 ...: test_size = 0.3, random_state=42)
reg_all = linear_model.LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
#.score will apply the R-squared metric
reg_all.score(X_test, y_test)
#Out[6]: 0.71122600574849526

#Usually never use it out of the box like here, we normally apply regularization




#%%Fit & predict for regression

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))
#scoring on training data

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()


#%%Train/test split for regression

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))


#%%Cross-validation

#Cross-validation motivation
#● Model performance is dependent on way the data is split
#● Not representative of the model’s ability to generalize
#● Solution: Cross-validation!

##Cross-validation basics

Cross-validation and model performance
#● 5 folds = 5-fold CV
#● 10 folds = 10-fold CV
#● k folds = k-fold CV
#● More folds = More computationally expensive

##Cross-validation in scikit-learn

from sklearn.model_selection import cross_val_score
reg = linear_model.LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)
print(cv_results)
#[ 0.63919994 0.71386698 0.58702344 0.07923081 -0.25294154]
np.mean(cv_results)
#Out[5]: 0.35327592439587058


#%%5-fold cross-validation

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

#%%K-Fold CV comparison

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))

%timeit cross_val_score(reg, X, y, cv = 3)

%timeit cross_val_score(reg, X, y, cv = 10)

#%%Regularized regression

##Why regularize?
#● Recall: Linear regression minimizes a loss function
#● It chooses a coefficient for each feature variable
#● Large coefficients can lead to overfitting
#● Penalizing large coefficients: Regularization

##Ridge regression
#● Loss function = OLS loss function +α ∗summation(from i=1 to n)ai^2
#● Alpha: Parameter we need to choose
#● Picking alpha here is similar to picking k in k-NN
#● Hyperparameter tuning (More in Chapter 3)
#● Alpha controls model complexity
#● Alpha = 0: We get back OLS (Can lead to overfitting)
#● Very high alpha: Can lead to underfitting

##Ridge regression in scikit-learn

from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X, y,
 ...: test_size = 0.3, random_state=42)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
#Out[6]: 0.69969382751273179

#Lasso regression
#● Loss function = OLS loss function + α ∗summation(from i=1 to n)|ai|

##Lasso regression in scikit-learn

from sklearn.linear_model import Lasso
X_train, X_test, y_train, y_test = train_test_split(X, y,
 ...: test_size = 0.3, random_state=42)
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)
#Out[6]: 0.59502295353285506

#Lasso regression for feature selection
#● Can be used to select important features of a dataset
#● Shrinks the coefficients of less important features to exactly 0

##Lasso for feature selection in scikit-learn

from sklearn.linear_model import Lasso
names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

#Lasso for feature selection in scikit-learn



#%%Regularization I: Lasso

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()



#%%Regularization II: Ridge

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
    
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

#should choose an alpha of about 0.5