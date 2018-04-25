# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:58:53 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Single Variable Linear Regression: Model Section 4.2.2

#%%Objective: Utilize the reformatted and imputed data to create a single 
#variable linear regression model to predict a ship's Nautical Miles Per Gallon 
#(NMPG). 

#Linear regression compares the relationship between two variables, a predictor 
#variable (a.k.a. the x-variable), and a dependent variable (a.k.a. the 
#y-variable), when those relationships are believed to be linear

#Hey, didn't we say single variable? Yes, we're trying to predict the dependent 
#variable, y, using only a single predictor variable, x. This is sometimes 
#called simple linear regression. If we have multiple predictor variables, we 
#might call this multivariate linear regression or general linear regression 
#(not to be confused with a generalized linear regression).

#%%Feature Selection

#Review: feature selection can be done by filtering or wrapper methods:

#Filtering methods analyze features using a test statistic and eliminate 
#redundant or non-informative features. As an example, a filtering method could 
#eliminate features that have little correlation to the class labels.

#Wrapper methods utilize a classification model as part of feature selection. A 
#model is trained on a set of features and the classification accuracy is used 
#to measure the information value of the feature set. One example is that of 
#training a Neural Network with a set of features and evaluating the accuracy 
#of the model. If the model scores highly on the test set, then the features 
#have high information value. All possible combinations of features are tested 
#to find the best feature set. 

#Filtering is faster but wrappers are more complete

#%%Use PyLab to plot each ship engine attribute against the NMPG

import numpy as np
import pylab as pl
# load the data in
data = np.genfromtxt('4-2-1 ship-nmpg-imp.csv', delimiter=",", names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
varnames = ['nmpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin','name']
# we loop through names[1:-1] to skip nmpg and uid
for name in varnames[1:-1]:
    pl.figure()
    pl.scatter(data[name], data['nmpg']) ##### use the scatter method to plot here #####
    pl.title("%s verses nmpg" % name)
    pl.xlabel(name)
    pl.ylabel('NMPG')
    pl.xticks()
    pl.yticks()
    pl.show()
pl.show
#Which features show strong and weak correlations?
    
#we also need to consider which variables should not be selected with respect 
#to the model assumptions:
    
#%%Model Assumptions
    
#Linearity
    
#No kidding sherlock but it does have to be stated explicitly. In order to 
#model some data with linear regression, the data should follow a linear trend. 
#If it does not, then we have to use a different kind of regression model or, 
#at the very least, a transformation
    
#In our data, displacement, horsepower, and weight have the greatest linearity 
#to the data

#Weak Exogeneity

#all this is saying is we know the predictor variables exactly--they do not 
#contain any noise. Rounding and the severity of it can be a common source of 
#noise

#In our data, origin and year are known exactly and noise could have been
#introduced in acceleration, displacement, horsepower, and weight

#Displacement is measured by the volume of the cylinders and can be obtained 
#fairly accurately. Weight is measured using a scale and is also fairly accurate. 
#Though displacement and weight could violate weak exogeneity, we probably 
#would not worry about them very much. 

#Horsepower is a little trickier. First, there are different ways to estimate 
#horsepower, and the method is not indicated in the data; it may even be 
#different for the different ships. Second, horsepower is a derived value; it 
#relies on knowing torque and Revolutions Per Minute (RPM) values accurately.

#Acceleration might also be problematic; it is measured by timing a ship over a 
#distance from a dead start while assuming constant acceleration. Of course, the 
#ship doesn't really accelerate uniformly, so we can only report an average 
#acceleration over that time period. Most values in the data end in .0 or .5 but
#some don't so reporting standards may not be uniform.

#Homoskedasticity or Constant Variance: 

#all of the dependent variables have the same amount of noise associated with 
#measuring them- i.e. a scatterplot of the variables doesn't give a cone
#http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html

#In our data, model year and origin appear to follow homoskedasticity the best

#What if noise does vary too much over the range? Then we can turn to something 
#like weighted Ordinary Least Squares (OLS) regression. 

#%%Performance Evaluation

#We start by looking at the statistical significance of a linear relationship. A 
#p-value less than .05 supports the existence of a linear relationship.

#Task:The scipy.stats.linregress calculates the linear dependence between NMPG 
#and predictor variables. 
#Print the p_value, and log(p_value) to evaluate the significance of the 
#regression. Tip: use np.log10()
#Print the p-value and log of the p-value for each comparison.

import numpy as np
import pylab as pl
from scipy import stats
# load the data in
data = np.genfromtxt('4-2-1 ship-nmpg-imp.csv', delimiter=",", names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
varnames = data.dtype.names
# we loop through names[1:-1] to skip nmpg and uid
for name in varnames[1:-1]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[name], data['nmpg'])
    print("Comparing nmpg to %s" % name)
    ## fill in the whitespace in the line below with the appropriate p_values
    print('p value = ', p_value, ' log(pvalue) = ', np.log10(p_value) , '\n')

#log10(p_value) of nmpg to weight is -103
#this is also the highest magnitude log p_value in the group making it the most
#statistically significant
    
#%%Gut Checking Parameters
    
#Task: Print the slope and intercept for each comparison.
    
import numpy as np
import pylab as pl
from scipy import stats
# load the data in header --> names=True
data = np.genfromtxt('4-2-1 ship-nmpg-imp.csv', delimiter=",", names=True,\
dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
varnames = data.dtype.names
# we loop through names[1:-1] to skip nmpg and uid
for name in varnames[1:-1]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[name], data['nmpg'])
    print("Comparing nmpg to %s" % name)
    print('slope = ', slope, ' intercept = ', intercept, '\n') ## print values here)

#The sign of the slope associated with the model for (model year, NMPG) 
#relationship is positive which is reasonable because engines should  become
#more efficient over time

#The slope of weight to nmpg is negative which also makes sense because you 
#would expect gas mileage to go down as the craft gets heavier

#The R-value is the pearson coefficient, R^2 is the coefficient of determination
#1 means perfect fit, 0 means no fit and is sensitive to outliers


#Task: print the correlation coefficient and coefficient of determination for 
#each comparison

import numpy as np
import pylab as pl
from scipy import stats
# load the data in
data = np.genfromtxt('4-2-1 ship-nmpg-imp.csv', delimiter=",", names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
varnames = data.dtype.names

# we loop through names[1:-1] to skip nmpg and uid
for name in varnames[1:-1]:
    slope, intercept, r_value, p_value, std_err = stats.linregress(data[name],data['nmpg'])
    print("Comparing nmpg to %s" % name)
    print('r_value = ', r_value,' r_sqrd = ', r_value**2 , '\n') ## add code here

#Task 2: For each pair that you calculated the dependence, plot the line found 
#in the previous step on top of the original scatter plot. Y and line color are 
#already defined for you.
    

    pl.figure()
    pl.scatter(data[name], data['nmpg'], color='#FF971C')
    Y = slope * data[name] + intercept
    pl.plot(data[name],Y,color='purple') ## add code beginning of line
    pl.title('%s model \n r^2= %s, log_pvalue= %s' % (name,r_value*r_value,np.log10(p_value)))
    pl.xlabel(name)
    pl.ylabel('NMPG')
    pl.xticks()
    pl.yticks()
    pl.show()

#weight is best winner for building a model
    
#%% Building the Model

#When the relationship between two variables is not actually linear, but we 
#would like to use the linear model, we use a transformation

#For comparison, build and plot two models using the weight feature. In order 
#to examine the  of the residuals, you also create a histogram.
    
#Model 1 - just the weight feature:
import numpy as np
import pylab as pl
from scipy import stats
#load the data in
data = np.genfromtxt('4-2-1 ship-nmpg-imp.csv', delimiter=",", names=True,
dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
#varnames = ['nmpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin','name']
varnames = data.dtype.names
# renaming data vectors for simplicity
weight = data['wt']
nmpg = data['nmpg']
print('Model 1: Weight as a predictor for nmpg')
#Calculating the linear dependence between mpg and
slope,intercept, r_value, p_value, std_err = stats.linregress(weight, nmpg)
Y = slope*weight +intercept
resid = nmpg-Y
fig, (ax1,ax2) = pl.subplots(2)
ax1.scatter(weight,nmpg , color='#FF971C')
ax1.plot(weight,Y, color='black')
ax1.set_xlabel('Weight')
ax1.set_ylabel('NMPG')
ax2.hist(resid, 40, color='#5D5166')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')
ax1.set_title('Weight Model')
pl.show()

#Model 2 with log transformation
# Note that we need to use the natural log, np.log().
print('Model 2: log_weight as a predictor for nmpg')
logwt = np.log(weight) 
#Calculating the linear dependence between mpg and log( )
logslope, logintercept, logr,logp, log_err = stats.linregress(logwt, nmpg)
logY = logslope*logwt +logintercept
logresid = nmpg - logY
fig, (ax1,ax2) = pl.subplots(2)
ax1.scatter( logwt, nmpg, color='#FF971C')
ax1.plot( logwt,logY, color='black')
ax1.set_xlabel('Log(Weight)')
ax1.set_ylabel('NMPG')
ax2.hist(logresid, 40, color='#5D5166')
ax2.set_xlabel('Residuals of log model')
ax2.set_ylabel('Frequency')
ax1.set_title('Log(Weight) Model')
pl.show() 

#%%Prediction on New Data

#Use Model 2 to predict the NMPG for a ship at dock with weight 135 to the 
#nearest whole number

import numpy as np
import pylab as pl
from scipy import stats
#load the data in
data = np.genfromtxt('4-2-1 ship-nmpg-imp.csv', delimiter=",", names=True,
dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
# renaming data vectors
weight = data['wt']
nmpg = data['nmpg']
## model2
slope,intercept, r_value, p_value, std_err = stats.linregress(weight,nmpg)
Y = slope*weight+intercept
resid = nmpg-Y
newshipweight = 135# add code here
prediction = slope*newshipweight+intercept# add code here
print (prediction)


