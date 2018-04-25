# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:44:07 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-SciKit-Learn Section 1.1.4

#Scikit, imported as sklearn, is a machine learning library that is built on 
#top of NumPy, SciPy, and matplotlib

import numpy as np
#import sklearn as sk
#commented out due to not being used
from sklearn import preprocessing
data = np.genfromtxt('1-1-4 Sample.data', delimiter=',', dtype='f8,f8,f8,f8',names='a,b,c,d')
#this method of import creates a flexible type which does not allow `mean.()` 
#to be called
X = np.array([data[x] for x in data.dtype.names])
mean = X.mean(axis=0) #mean of each column
print(mean)
stdev = X.std(axis=0)  #stdev of each column
print(stdev)
scaledData = preprocessing.scale(X)
print(scaledData)

print(scaledData.mean(axis=0)) #mean of each column of scaled data (0 across)
print(scaledData.std(axis=0)) #std of each column of scaled data (1 across)
#We can see the data is normalized

#Handbook Entry:
##Machine Learning Overview
#Machine Learning is the name for the type of techniques that use algorithms 
#to essentially learn from data. There are two types of learning: supervised 
#and unsupervised. Supervised is where you give the algorithm training data 
#and the answers from which it learns. For example, classifying email as 
#"inbox" or "spam" is where a machine learning algorithm is given examples 
#of both kinds of email message, and then asked to check each incoming 
#message, and sort appropriately. Conversely, unsupervised machine learning 
#is where the algorithm is given data and is asked to learn natural 
#relationships or patterns that exist in the data. For example, given an 
#overhead shot of a carnival you can assign each person a point in #
#2-dimensional space. Now, you don't know who is with whom, so giving 
#training data is impossible, but you could cluster the points (people) 
#into groups that are likely to be together.

#vWhen scaling training data, it is important 
#that you perform the same scaling on your test data

X = np.loadtxt('1-1-4 sample.data',delimiter=',')
mean = X.mean(axis=0)
stdev = X.std(axis=0)
scaler = preprocessing.StandardScaler().fit(X)
print('scaler.mean =', scaler.mean_)
Xscaled = scaler.transform(X)
print('Xscaled mean=', Xscaled.mean(axis=0))
#Here's what happens when we apply the scaler to secondary data
#just like we would from training data to test data
newdata = np.loadtxt('1-1-4 sample2.data', delimiter=',')
print(newdata)
newscaled = scaler.transform(newdata)
#mean is not 0
print(newscaled.mean(axis=0))
#and std is not 1 
print(newscaled.std(axis=0))



