# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 11:30:10 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

## Data Science Examples from ExploringDataScience.com

#Describe-Correlation Section 2.1.3

#Correlation is not causation
#Pearson Product-Moment Correlation Coefficient-r-
#Measures the strength of the linear relationship between two variables
#Assumes linearity, highly susceptible to outliers (remove or use Spearman's rank), 
#Homoscedasticity (data has same spread across variables)-will cause underreporting
#Bivariate Normal-both variables have approx normal distributions
#Checklist- looks roughly linear, no massive outliers, 
#spread is roughly constant, bivariate normal distribution

#scatter plot of data
import numpy as np
from numpy import loadtxt
from numpy import transpose
from matplotlib import pyplot as plt
#Import Data
data0 = loadtxt("2-1-3 VOLCANICDATA.txt")
data = transpose(data0)
#Define Variables
prob = data[0,]
length = data[1,]
#Use "scatter()" to create a scatterplot of prob vs. length.
plt.scatter(prob,length)
plt.show()

#Pearson's- r= cov(x,y)/(sig_x*sig_y)
#covariance is normalized by standard deviation of both variables
#allows us to compare the covariances of different data sets to determine 
#which relationships are strongest

#Null Hypothesis: No linear relationship exists between the two variables (i.e. r=0).
#Alternative Hypothesis: A linear relationship exists between the two variables (i.e., r≠0)

#from scipy.stats.stats import pearsonr
#PearsonOutput = pearsonr(x,y)
#print "Pearson Correlation Coefficient is:", PearsonOutput[0]
#print "The p-value is:", PearsonOutput[1]

###Calculate Pearson's r and the associated p-value using pearsonr().
from scipy.stats.stats import pearsonr
PearsonOutput = pearsonr(prob, length)
print('Pearson Correlation Coefficient is:', PearsonOutput[0])
print('The p-value is:', PearsonOutput[1])
#Pearson Correlation Coefficient is: 0.824951785908
#The p-value is: 3.21088420934e-12
#a strong linear relationship, where the probability of an eruption and the length of volcanic activity vary together.
#Strong evidence of linear relationship

#Spearman's Rank Correlation Coefficient-does not utilize all data so 
#Pearson's is preferred but Spearman's is more flexible
#Measures the strength of any increasing or decreasing relationship.
#Is more robust to outliers.
#Does not assume x,yare bivariate normal.
#Is more robust to heteroscedasticity.
#Does not fully utilize all the information in the data set (due to use of “ranked” data).

#from scipy.stats.stats import spearmanr
#SpearmanOutput = spearmanr(x,y)
#print "Spearman Rank is:", SpearmanOutput[0]
#print "The p-value is:", SpearmanOutput[1]
from scipy.stats.stats import spearmanr
#Define Variables
erup = data[0,]
evac = data[4,]
SpearmanOutput = spearmanr(erup,evac)
print('Spearman Rank is:', SpearmanOutput[0])
print('The p-value is:', SpearmanOutput[1])

#Additional Topics
#Kendall's Tau Coefficient
#Kendall's tau coefficient compares each (x,y) data point to every other (x,y)
#data point and counts the number of concordant and discordant combinations.
#(Concordant-a<c & b<d or c<a&d<b)(Discordant-a<c&d<b or c<a&b<d)
#tau=(nConcordantPairs-nDiscordantPairs)/(n(n-1)/2)
#In the numerator:
#A large positive value indicates that most pairs of points indicate an increasing relationship
#a large negative value indicates that most pairs of points indicate a “decreasing” relationship
##tau is between -1 and 1 with identical interpretations to Spearman's and Pearson's

#Correlation and Multiple Variables
#Correleation Matrix:
#A heat map is one way to create a visual which naturally draws attention to the strongest relationships in the data
#Example Heat Map:

#import matplotlib.pyplot as plt
#import numpy as np
column_labels = list('12345')

row_labels = list('12345')
data = np.array([[1, 0.8, 0.4, 0.1, -0.7],[0.8, 1, 0.5, -0.1, -0.9],[0.4, 0.5, 1, 0.95, 0.05],[0.1, -0.1, 0.95, 1, 0.2],[-.7, -0.9, 0.05, 0.2, 1]])
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.RdBu)

ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(column_labels, minor=False)
plt.show()

#Autocorrelation: Correlation and Time Series
#The autocorrelation function A, is simply a (discrete) function where A(x) 
#is the correlation between points x distance apart.
#Plotting the autocorrelation function is a simple way to identify potential 
#high correlations
#The autocorrelation function is always 1 at x=0 and the value of A(1) tends to 
#be fairly high as events which effect one month are likely to effect the next month
#As a general rule of thumb, if the remaining autocorrelation values are 
#roughly a random scatter centered at zero and generally within 0.2 of zero, 
#then there is little evidence of cyclic behavior

#Following example code from MachineLearningMastery:
#https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
#Obtained Jan 7, 2018
from pandas import Series
#from matplotlib import pyplot
series = Series.from_csv('2-1-3 daily-minimum-temperatures.csv', header=0)
#This dataset was been cleaned of extraneous '?' marks and footer info to stop errors
print(series.head())
series.plot()
plt.show()

#lag plot
from pandas.tools.plotting import lag_plot
#series = Series.from_csv('daily-minimum-temperatures.csv', header=0)
lag_plot(series)
plt.show()

#one-off check
#from pandas import Series
from pandas import DataFrame
from pandas import concat
#from matplotlib import pyplot
series = Series.from_csv('2-1-3 daily-minimum-temperatures.csv', header=0)
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)

#Autocorrelation Plots
#from pandas import Series
#from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
series = Series.from_csv('2-1-3 daily-minimum-temperatures.csv', header=0)
autocorrelation_plot(series)
plt.show()

#alternately from statsmodels library:
#from pandas import Series
#from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
series = Series.from_csv('2-1-3 daily-minimum-temperatures.csv', header=0)
plot_acf(series, lags=31)
plt.show()

##Sample 7-day forecast based on learning data
#from pandas import Series
#from pandas import DataFrame
#from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
series = Series.from_csv('2-1-3 daily-minimum-temperatures.csv', header=0)
# create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
# split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X, train_y = train[:,0], train[:,1]
test_X, test_y = test[:,0], test[:,1]

# persistence model
def model_persistence(x):
	return x

# walk-forward validation
predictions = list()
for x in test_X:
	yhat = model_persistence(x)
	predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)
#MSE is 3.423- our baseline error
# plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()
#This is baseline prediction based on previous day's temperature

#Autocorrelation Sample:
#from pandas import Series
#from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
#from sklearn.metrics import mean_squared_error
series = Series.from_csv('2-1-3 daily-minimum-temperatures.csv', header=0)
# split dataset
X = series.values
#Train on everything but last 7 days, test on last 7 days
train, test = X[1:len(X)-7], X[len(X)-7:] 
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()