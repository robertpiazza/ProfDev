# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:44:48 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Random Forests Section 4.1.3

#Utilize multiple decision trees to classify human activity data and thus, 
#accomplish the Captain's orders.

##Bootstrapping

#To build a forest of trees we need to start with a single tree. To create a 
#tree, we need to have a bootstrapped data set from which to learn. To create a 
#bootstrap sample from this full data set, we sample a training point from the 
#data set. Then, we make a copy of it and place the copy into our training set. 
#The original remains in our complete data set. This is called sampling with 
#replacement. We repeat this until we have filled our training data set. If we 
#desired a training set of 10 data points, we would repeat 10 times. At this 
#point you may be wondering what is stopping us from picking the same point 
#twice. Nothing! The distribution of a single training set may vary from the 
#original distribution from which it was sampled. Put another way, the original 
#data set had 50% 7's and 50% 5's. A bootstrapped sample may have a distribution 
#of 10% 7's and 90% 5's. This is caused by random chance and sampling with 
#replacement.

#Another important feature of random forests is that they only work on a 
#random set of the available features at each tree. This can be thought of as 
#choosing a random subset of features and learning from them, which forces the 
#algorithm to not always choose the same starting condition on which to split 
#the data

#A decision tree works by dividing the data according to the strongest features
#To get more diverse trees, we need to provide only a subset of our features to 
#the tree. This way a lower-ranking feature gets a chance to root the tree. 
#This means that each tree receives a random subset of the overall image

#Combining Models

#To determine what model output to use, we take the mode of all the decision 
#trees output. This means that the majority vote is taken as the label for the 
#data. For example, if we have 5 decision trees and they output 7,7,7,7,5 as 
#class labels, we have 4 votes for '7' and 1 vote for '5'. The mode of this is 
#7 and this is the output value that we use for the random forest.

#We get a higher classification performance when we use this. If we build a 
#single model from the data, the model performs well on that data. However, if 
#we provide new data to it that it had not learned from, it will likely not 
#perform as well. By bootstrapping the data, we are creating trees that learn 
#different concepts about the data (i.e., the tree that thinks everything is a 
#7). By combining multiple versions of this, we can exploit characteristics 
#from each of them.

#Determine activity based on accelerometer readings distributed on a person

#Goal is based on this data to distinguish activity with 80% accuracy

#%%Single tree:
import numpy as np
from sklearn import tree
from sklearn import cross_validation
#load data from file
data = np.loadtxt('4-1-3 accelData.txt',delimiter=',')
DT = data.transpose()
#split data into arrays
X = np.array(DT[0:-1]).transpose()
Y = np.array(DT[-1]).transpose()
#create training and test sets, 70% in training 30% in testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.30, random_state=0)
dt = tree.DecisionTreeClassifier(random_state=0)
# pass in the right arguments
dt.fit(X_train,y_train)
print(dt.score(X_test,y_test))
#0.7 but can't address variability

#%%Now an ensemble
from sklearn import ensemble
from sklearn.utils import shuffle
import numpy as np
from sklearn import cross_validation
#load data from file
data = np.loadtxt('4-1-3 accelData.txt',delimiter=',')
DT = data.transpose()
#split data into arrays
X = np.array(DT[0:-1]).transpose()
Y = np.array(DT[-1]).transpose()
#create training and test sets, 70% in training 30% in testing set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.30, random_state=0)
#construct random forest with 10 trees, creating bootstrap samples
rf = ensemble.RandomForestClassifier(n_estimators=10,random_state=0,bootstrap=True)
rf.fit(X_train,y_train) ### fit and score the model
print(rf.score(X_test,y_test))

#%% Final thoughts

#An ensemble of trees performs better than a single tree. This is because the 
#trees learn different concepts from the data. Some trees are better than 
#others on certain examples. By combining trees we can leverage this strength.

