# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 10:53:21 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Tree Based Regression Section 4.2.6

#Your task is to identify the relationship between body temperature and babel 
#fish translator bacteria count. Your Data Science Officer received a data set 
#from our data-gathering nanobots. You must also model babel fish fishery costs 
#using random forests to estimate the costs of a babel fish fishery to provide 
#all space explorers with universal translators.

#We use random forests to calculate and analyze the relationship between 
#predictor and responder variables to determine how fish body temperature 
#effect the number of translator bacteria in our babel fish. First, we create 
#two random forest models with the same number of trees, but different depths 
#to examine the trade-off between depth and overfitting. Next, we predict and 
#plot the regression results for temperature and translator bacteria to 
#understand the relationship between the two variables and tree depth on random 
#forest regressor results. We then analyze the plot to better understand how 
#to control the temperature in a way that maximizes translator capabilities by 
#increasing bacteria counts.

#Finally, we analyze which model, random forests or linear regression, is a 
#better fit for the babel fish fishery data set. In this case, we use the 
#coefficient of determination as a metric of performance.

#%%Create and Build Two Random Forests
#In this challenge, you fit two random forest regressor models. These models 
#will be used in other missions to demonstrate the effects of specifying 
#different tree depths. The code provided reads the data set. You now create 
#two random forest regressors, each with a forest of 10 trees. 

#Like all trees, random forests start at a root and have branches. The depth 
#of the branches impacts how well a random forest regressor model fits the data 
#set: underfitting, overfitting, or just right. In our case, we try two depths 
#of 2 and 5 branches. In general, as you add more branches, the model begins to 
#overfit the data.

import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
#data set
# Get Variable Names
with open('4-2-6 babelfishreadings.csv', 'r') as csvfile:
    _reader = csv.reader( csvfile, delimiter =',',quotechar ='"')
    variable_names = _reader.next()
    variable_names = np.array(variable_names)
# Get Variables
data = np.genfromtxt('4-2-6 babelfishreadings.csv',dtype=float, delimiter=',', skip_header=1)
X = data[:, [0]]
y = np.ravel(data[:,[1]])
# Random Forest Regressors
rfr_1 = RandomForestRegressor(n_estimators=10, max_depth = 2)
rfr_2 = RandomForestRegressor(n_estimators=10, max_depth = 5)
# Fit/Build the Model
rfr_1.fit(X,y)
rfr_2.fit(X,y)

#%%Plot Model Results

#First, create a set of evenly-spaced intervals from 0.0 to 10.0 in increments 
#of 0.01 that you can use to plot model prediction across the range of 0 to 10. 

#Next, feed the points to the models to see both models' predictions across the 
#input space.

import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as pl
# data set
# Get Variable Names
with open('4-2-6 BabelFishReadings.csv', 'r') as csvfile:
    _reader = csv.reader( csvfile, delimiter =',',quotechar ='"')
    variable_names = next(_reader)
    variable_names = np.array(variable_names)
# Get Variables
data = np.genfromtxt('4-2-6 BabelFishReadings.csv',dtype=float, delimiter=',', skip_header=1)
X = data[:, [0]]
y = np.ravel(data[:,[1]])
# Random Forest Regressors
rfr_1 = RandomForestRegressor(n_estimators=10, max_depth=2)
rfr_2 = RandomForestRegressor(n_estimators=10, max_depth=5)
# Fit/Build the Model
rfr_1.fit(X, y)
rfr_2.fit(X, y)
# Input data
X_ = np.arange(0.0, 10.0, 0.01)[:, np.newaxis]
# Predicting
y_1 = rfr_1.predict(X_)### INSERT CODE
y_2 = rfr_2.predict(X_)### INSERT CODE
# Plot the results
pl.figure()
pl.scatter(X, y, c="k", label="data")
pl.plot(X_, y_1, c="#5D5166", label="max_depth=2", linewidth=2)
pl.plot(X_, y_2, c="#FF971C", label="max_depth=5", linewidth=2)
pl.xlabel("Temperature 10 degree Units")
pl.ylabel("Translator bacteria (in thousands) ")
pl.title("Regression")
pl.legend()
pl.xlim((0,10))
pl.ylim((0,10))
pl.show()

#%%Random Forests for High Dimensional Models
#In this challenge, we compare the coefficient of determination scores (R^2) 
#between a linear regression and a random forest model to predict median value 
#of alien-occupied fisheries from the Babel Fishery data set. The R^2 is the 
#percent of variation that can be explained by the regression model. The higher 
#the number, the more variation is accounted for and the closer the model.

#We have already fit a linear regression model to the data, which scored 
#R^2 = 0.67. Now, see what you get using a random forest.

import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
# Get Variables
data = np.genfromtxt('4-2-6 BabelFishery.data',dtype=float, skip_header=1)
X_ = data[:, :13] # Get the 13 variables
y_ = np.ravel(data[:,[13]]).reshape(-1,1) # get the 14th variable is the expected result, the Median value of alien-occupied fisheries

# Normalize
scalerX = StandardScaler().fit(X_)
scalery = StandardScaler().fit(y_)
X_ = scalerX.transform(X_)
y_ = scalery.transform(y_).reshape(506,)
# Random Forest Regression
rfr = RandomForestRegressor(n_estimators=10,random_state=42)
rfr.fit(X_,y_)
print(rfr.predict(X_).shape)
print(y_.shape)
print(np.corrcoef(rfr.predict(X_),y_))