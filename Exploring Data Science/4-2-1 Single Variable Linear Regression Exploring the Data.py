# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:56:32 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Single Variable Linear Regression: Exploring the Data Section 4.2.1

#%%Objective:

#predict ship Nautical Miles Per Gallon (NMPG) using single variable linear 
#regression

#%%

#Data exploration allows for a basic understanding of your data. This is key 
#when determining a method for analysis, choosing preprocessing and cleaning 
#methods, interpreting model results, and performing gut checks

#Data set is a modified auto data set

#%% Data exploration I

#The initial format the data is provided in, is not ideal. The column variables 
#are separated by inconsistent amounts of whitespace. We will start by 
#reformatting the data into a more usable format


import csv
datafile = open('4-2-1 ship-nmpg.data','r')
outfile = open('4-2-1 ship-nmpg.csv','w')
out = csv.writer(outfile)
# write out the csv headers
out.writerow(['nmpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin','name'])
for line in datafile:
    line = line.strip()
# add code to split the numeric data from the string name
    numbers, name = line.split('\t')
# add code to split the numbers into a list
    numbers = numbers.split()
    newline = numbers + [name]
# add code to writerow each newline of data as csv
    out.writerow(newline)
datafile.close()
outfile.close()

#%% Data Exploration II

import numpy as np
import scipy as sp
from scipy import stats
data = np.genfromtxt('4-2-1 ship-nmpg.csv', delimiter=",", names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,S5")
print(data.size) #398 rows
#First entry in cyl column
print('First entry', data['cyl'][0])
#Minimum entry in cyl column
cylNobs, (cylMin,cylMax), cylMean, cylVariance,cylSkewness, cylKurtosis = stats.describe(data['cyl'])
#DescribeResult(#observations, minmax=(), mean, variance, skewness, kurtosis)
print('Cyl min entry', cylMin)
#Mean of cyl
print('Cyl Mean', sp.mean(data['cyl']))
#Median of cyl
print('Cyl median', sp.median(data['cyl']))
#Mode of cyl
cylNumber, cylAmount = sp.stats.mode(data['cyl'])
print('Cyl median', cylNumber)
#hp column is missing numbers:
print('hp mean', sp.mean(data['hp'], 'not a number'))

#%% How to deal with missing values

#There's a lot of methods for imputation

#We'll use (median+mode)/2 rounded to nearest integer

#Be cautious when using this imputation method, as it can bias results when the 
#number of missing data points is large

import numpy as np
from scipy import stats
data = np.genfromtxt('4-2-1 ship-nmpg.csv', delimiter=",", names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,U35")
names = data.dtype.names
print(names)
hpmean =np.nanmean(data['hp'])
print(hpmean)
hpmedian =np.nanmedian(data['hp'])
print(hpmedian)
imputeHP = np.round((hpmean+hpmedian)/2.0,0)

for i in range(len(data['hp'])):
    if np.isnan(data['hp'][i]):
        data['hp'][i] = imputeHP #assign value here
        
np.savetxt('4-2-1 ship-nmpg-imp.csv', data, delimiter=',', newline='\n',fmt="%f,%i,%f,%f,%f,%f,%i,%i,%s", header= "'nmpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin', 'name'")