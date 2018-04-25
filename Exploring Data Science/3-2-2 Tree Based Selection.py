# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:51:05 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Discover-Tree Based Selection Section 3.2.2

#use random forest classification and the feature importance score to identify 
#strong predictor variables

#The consumers of data science analytics often want to understand the main 
#drivers of a process and how to influence the outcome in their favor. As data 
#scientists attempt to characterize a process or system, it is necessary to 
#identify noisy variables and remove them from the models. Identifying and 
#eliminating sources of noise helps a model to make useful predictions. One way 
#to accomplish this is to use random forests and importance coefficients

#A random forest is a collection of decision trees

#an individual decision tree is a set of if-then branching statements on 
#variables that end with a model prediction

#Decision trees are simple, and usually quite effective, models that split the 
#input space into non-overlapping subsets of the original data. The tree fits a 
#unique model to each subset, which may be as simple as a value (e.g., True or 
#False) or may be more complex, such as a regression model over the subset. 

#Decision trees are also useful in regression applications. Instead of a leaf 
#node providing a fixed value, the leaf node could provide a more complex model 
#over the subset of data, such as a linear regression model

#CART and C4.5 are most common

#At the root node, the algorithm considers all variables and thresholds, and 
#determines which variable and which threshold results in the "best" split of 
#the data. "Best" is detemined by a measure of accuracy at the child nodes, 
#which is usually based on information theoretic measures. The recursion 
#continues until it hits termination criteria, such as tree depth, or the 
#number of remaining observations at the leaf node. Some implementations 
#include branch pruning criteria to avoid overfitting.

#A random forest is a collection of decision trees.Each of the trees is 
#constructed from a different subset of the data, and often the depth of the 
#tree is limited to a small value. Different randomizations may occur where an 
#individual tree may split on a random subset of variable attributes and the 
#thresholds selected for the splits can be randomly selected. For example, a 
#random forest might consist of 100 decision trees, each trained on a randomly 
#selected 80% of the data. For each tree, only a subset of the attributes will 
#be used to split nodes in the tree.

#You would think that this results in poorly performing trees, which it does. 
#However, when you combine the results of several ok-but-not-great models, the 
#average over all trees produces very good results. Random forests generate the 
#prediction by aggregating predictions over all trees by majority vote or by 
#weighting the predictions of each tree. The randomization and aggregation 
#present in random forests are quite effective at preventing overfitting and 
#dealing with noisy data.

#Random forests can estimate variable importance in a couple of different ways
#The implementation we employ uses the depth that each variable appears in the 
#decision trees comprising the random forest. Variables that tend to be near 
#the top of trees influence the prediction of more observations than variables 
#that tend to be near the bottom. The fraction of observations affected by a 
#variable is used as an estimate of variable importance. You can read more 
#details about this approach in the SciKit documentation

#what makes food edible by using random forests to determine variable importance. 
#Examine the model output first, then interpret variable importance from the model.

import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Get variables names ['Number_of_spines' ' Colored' ' Toothed' ' Has_flower' ' Lobed']
#with open('3-2-2 plants.csv', 'rb') as csvfile:
#    plants_reader = csv.reader( csvfile, delimiter =',',quotechar ='"')
#    variable_names = plants_reader.next()
#    variable_names = (np.array(variable_names))[[ 0,1,2,3,4]] #only need 5 (0-4)

#Commented out due to variable names not properly extracting

# Read in the samples for variable names into X and edible flags for y

data = np.genfromtxt('3-2-2 plants.csv',dtype=float, delimiter=',', skip_header=1)
plants_X = data[:, [0,1,2,3,4]] # get all the samples for the variables
plants_y = np.ravel(data[:,[5]]) # Flattened array required for fit's 2nd argument y
clf = RandomForestClassifier( n_estimators = 10, random_state = 33)
clf = clf.fit(plants_X, plants_y)
#print(variable_names)
#['Number_of_spines', ' Colored', ' Toothed', ' Has_flower', ' Lobed']
print(clf.feature_importances_)
#[ 0.32794357  0.42173935  0.13083123  0.04243847  0.07704738]
#'Colored' is the most important
#Has flower is least important



