# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:23:05 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

# Data Science Examples from ExploringDataScience.com

###Foundations-SciPy Section 1.1.3

# SciPy, another Python package for statistics

import scipy as sp
#Use the Python help function to open the help for open. Normally, q would
#quit back to your shell, but the web console works a little differently
#help(open)

#SciPy also has an enhanced help function called info. You can search SciPy 
#information for the word "mean". This searches the SciPy and NumPy for the 
#word mean
sp.info('mean')
sp.info(sp.mean)
#not working

##Dictionaries

#Dictionaries are a useful structure that Python uses to store key value pairs 
#very similar to a hash map, map, or key-value pair in other languages

#D = {'key1': 'value1'} 
#You would access the value for key1 using D['key1']

#SciPy can calculate many of the statistics as NumPy, but is more 
#computationally efficient and has a higher degree of accuracy

import numpy as np
from scipy import stats
data = np.genfromtxt('iris.data', delimiter=',', dtype='f8,f8,f8,f8,S15',names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type_string'])
#We have entered a method from the SciPy library to calculate the min, max, 
#mean, standard deviation, and variance. 
#For more information, use sp.info(stats.describe).
#given:
sizeofdata, (minval,maxval), mean, variance, skew, kurtosis = stats.describe(data['petal_width'])
#assignment
sizeofdata, (minval,maxval), mean, variance, skew, kurtosis = stats.describe(data['petal_length'])
#The median is not included as the output of the describe function. For median, you can use sp.median( ). 
#median of the petal_length vector-
sp.median(data['petal_length'])
