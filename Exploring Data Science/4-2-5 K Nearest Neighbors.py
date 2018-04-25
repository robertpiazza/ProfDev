# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:51:57 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-K Nearest Neighbors Section 4.2.5

#Suppose we have have a large collection of data on home properties. This 
#includes information like the number of bathrooms, the square footage, the sale 
#price, etc. We want to determine the price of our home by referencing 
#comparable houses. What do real estate agents do? They typically find similar 
#homes and look at the prices. For example, our house has 2 bathrooms and the 
#one that sold down the street has 2 as well, so we can say that these houses 
#share a similar feature. We go through all of our pre-existing examples, 
#finding the best overall matches. If we are looking for the 5 closest examples, 
#we have a k of 5. We then average the sale prices of the k most similar houses. 
#This average sale price is one estimate of what we expect for our home. But we 
#can do even better by weighting the 5 neighbors, by how similar they are to our 
#home. This means that data points that are more similar to the data point we 
#are attempting to predict are weighted higher than more distant examples. If, 
#for example, the first 4 examples are very similar compared to the 5th neighbor, 
#the first 4 examples account for more of the average value than the 5th.

#%% Transormation

#Normalization helps eliminate bias in distance metrics by putting all values 
#on the same scale. When calculating the Euclidean distance between instances, 
#if one variable is price and is measured in thousands of dollars, and another 
#variable is average temperature, a 100 unit shift in price is likely not very 
#significant, whereas a 100 unit shift in temperature is very significant. If 
#you have not previously completed the Hypothesis Testing mission, please do so 
#for further detail.

#Normalize the zaploid data to be between 0 and 1 using the MinMaxScaler class 
#from scikit-learn. The function that you need to use is fit_transform(). This 
#returns the data, but transformed to be between 0 and 1.


from sklearn import neighbors
from sklearn.datasets import load_boston
import math
import sklearn.preprocessing
#load the data
zaploid = load_boston()
#load MinMaxScaler
#the feature_range option specifies the scale of the features to use
mmScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
zaploidData = zaploid.data
scaledData = mmScaler.fit_transform(zaploidData) 
print(scaledData)

#%% Regression

#Now that we transformed the data, we need to perform KNN regression. We split 
#the data into training and testing sets, and then evaluate its performance.

from sklearn import neighbors
from sklearn.datasets import load_boston
import math
import sklearn.preprocessing
#setup the scaler
mmScaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
#load the data
zaploid = load_boston()
zaploidData=zaploid.data
#scale the data
scaledData = mmScaler.fit_transform(zaploidData)
#scale the target variable
scaledTarget = zaploid.target
#take first 350 as training
X = scaledData.tolist()[:350]
Y = scaledTarget.tolist()[:350]
#take remaining as test
testX = scaledData.tolist()[351:]
testY = scaledTarget.tolist()[351:]
#setup regression model
knn = neighbors.KNeighborsRegressor(n_neighbors=5,weights='distance')
y1 = knn.fit(X,Y).predict([testX[0]]) ### fit the model and predict the test value
print y1