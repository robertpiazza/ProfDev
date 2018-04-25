# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 16:08:28 2018

@author: Robert Piazza - Booz Allen Hamilton
Describe 
"""

## Data Science Examples from ExploringDataScience.com

#Describe-Summary Statistics Section 2.1.1
import numpy as np
import pandas as pd
from scipy import stats
iris = pd.read_csv('iris.data', names = ['sepalL', 'sepalW', 'petalL', 'petalW', 'class'])
irisData = iris['petalL']
x=irisData
#most common measure of central tendency
arithmeticMean = np.mean(x)

#geometric mean--the nth root of the product of all of our data points, 
#where n is the number of observations
iris['sepalArea']=iris['sepalL']*iris['sepalW']
sepalMean = np.mean(iris['sepalArea'])

geometricMean = stats.gmean(x)

#harmonic mean can be used to generate a valid estimate of the average
# or rates or ratios

#The Enterprise travels for a total of 50,000 miles. 
#It travels the first 25,000 miles at a rate of 18,000 miles per hour 
#and the remaining 25,000 at a rate of 20,000 miles per hour. 
#What is its average speed? 
print(stats.hmean([18000.0,20000.0]))

harmonicMean = stats.hmean(x)

#Median
median = stats.scoreatpercentile(x, 50)
median = stats.scoreatpercentile(iris['sepalArea'], 50)

#Mode (tricky- first number is mode, second is number of occurences)
mode = stats.mode(x)
mode = stats.mode(iris['sepalL'])

#Measures of dispersal
var = np.var(x)
std = np.std(x)

#Median Absolute Deviation-average absolute distance between a set 
#of observations and some fixed-value
def medianAbsoluteDeviation(x):
    m = np.median(x)
    return np.median([np.abs(xi - m) for xi in x])

#mean absolute deviation, which considers the average deviation 
#rather than the 50th percentile value
def meanAbsoluteDeviation(x):
    m = np.mean(x)
    return np.mean([np.abs(xi - m) for xi in x])

#Variance of Sepal Area:
print(np.var(iris['sepalArea']))
#standard deviation of the sepal area
print(np.std(iris['sepalArea']))
#mean absolute deviation of the sepal area
m = np.mean(iris['sepalArea'])
print(np.mean([np.abs(xi - m) for xi in iris['sepalArea']]))

#Categorical Data Metrics

#M1 Index quantifies the level of dispersion 
#M1 Index is maximized when the information entropy is at its highest;
#that is, when our observations are evenly distributed across all categories.

m1Data = iris.groupby('class').size() / float(iris.shape[0])
def m1Index(x):
    return 1 - sum([xi ** 2 for xi in x])
print(m1Index(m1Data))

#Distribution shape
#Skewness-how symmetrical a distribution is about its mean
#(-) means median to left of mean and vice versa
#>1 highly skewed, .5-1 moderately skewed, <.5 approx symmetric
stats.skew(x) #bias-corrected skewness (for samples) can be obtained by setting bias = false

#bias-corrected skewness of the petal rectangular area
iris['petalArea']=iris['petalW']*iris['petalL']	
print(stats.skew(iris['petalArea'], bias = False))

#Kurtosis- measures how 'peaked' a distribution is- how much density is in the tails
#Excess kurtosis (EK)- sample kurtosis - 3x bias correction- good for multiple indpendent random variables
#EK classified in 3 categories: 
#1. Mesokurtic - 0 EK - Gaussian
#2. Leptokurtic: Rare or extremem outliers occur more often- A distribution with positive excess kurtosis (e.g., a logistic distribution). Leptokurtic distributions can be thought of as having "fatter" tails compared to a normal distribution. 
#3. Platykutric: A distribution with negative excess kurtosis (e.g., a Bernoulli distribution). Platykurtic distributions are "spread out" relative to a normal distribution and have "thinner" tails.
stats.kurtosis(x)
#This is EK, get sample Kurtosis with bias = false.

#bias-corrected kurtosis of the petal rectangular area
print(stats.kurtosis(iris['petalArea'], bias = False))