# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 09:06:46 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Multivariable Linear Regression: Model Section 4.2.3

#We need to expand our single variable to multi-variate regression to make it 
#more robust

#Objective: Create a model with multiple predictor variables to predict 
#nautical miles per gallon (NMPG) 
#Besides weight, based on R^2 and p-values, we should be using cylinder, 
#horsepower, and displacement

#For this mission, we are using a library for OLS from SciPy Cookbook. It 
#handles a lot of testing for us. However, the documentation for it is not that 
#extensive. If you get stuck, you can use print dir(modelname) to see what 
#properties and methods are available.

#%%Model Interpretation Basics

#x1 is defined by combining the weight and cylinders feature arrays. Use the 
#ols method to create our first regression model: 
#ols(nmpg,x1,'nmpg',['weight','cylinders'])
#After executing the command, look at the output. It is an array of p-values, 
#where the first entry corresponds to the constant (b in y=mx+b). The other 
#entries are ordered identical to x1. 

#What is the p-value associated with the "cylinders" variable?

from ols import ols #ols.py file held in this project's directory
import numpy as np
# header: 'mpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin','name'
data = np.genfromtxt('4-2-3 ships.csv', delimiter=",",names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,S5")
wt = data['wt']
cyl = data['cyl']
nmpg = data['nmpg'].transpose()
hp = data['hp']
disp = data['disp']
x1 = np.array([wt,cyl]).transpose()
# OLS Models
model1 = ols(nmpg,x1,'nmpg',['weight','cylinders'])
print(model1.p) #p values for variables. 
print(model1.R2) #R2 value for model
print(model1.R2adj) #R2 value adjusted for complexity of the model
#model summary
print(model1.summary())
#SSE is the sum of squares error. 
#SSTO is the total sum of squares. 
#SSR is the sum of squares regression. 

#Adjusted R2 is always less than R2 because R2 always increases with added 
#features. Also, the total sum of squares is fixed

#%%Assumptions

#All the previous linear assumptions still apply. With multi-variate regresstion,
#we also must assume:
#No Predictor Multicollinearity
#It is assumed for multivariable linear regression that there is no exact 
#linear relationship among any of the independent variables in the model

# Another way to say this is that there should be no column of predictors that 
#can be written as a scaled version of any other predictor column (e.g., 1st 
#column = 2 * 2nd column) or the sum of any combination of scaled predictor 
#columns (e.g., 1st column = 3 * 2nd column + 2 * 3rd column).

#If this assumption is violated-Ridge regression variation may be a better model 
#for data with multicollinearity.

#%%Feature and Model Selection

#task: explore the effects of adding additional features to the regression model.

from ols import ols
import scipy as sp
from scipy import stats
import numpy as np
##header: 'mpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin','name'
data = np.genfromtxt('4-2-3 ships.csv', delimiter=",",names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,S25")
wt = data['wt']
cyl = data['cyl']
nmpg = data['nmpg'].transpose()
hp = data['hp']
disp = data['disp']
x1 = np.array([wt,cyl]).transpose()
# OLS Models
model1 = ols(nmpg,x1,'nmpg',['weight','cylinders'])
x2 = np.array([wt,cyl,hp,disp]).transpose()
print(model1.summary())
model2 = ols(nmpg,x2,'nmpg',['wt','cyl','hp','disp'])
print(model2.summary() )

#note: The durbin watson statistic is the likelihood that the errors are 
#autocorrelated either positively or negatively which is a common problem in 
#time-series analysis. In financial data, this can be handled by converting 
#prices to percentage-price change.

#Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) 
#are criterions of model fit. Given multiple different models, you may want to 
#select the model with smallest AIC or BIC.

x3 = np.array([wt,cyl,hp]).transpose()
model3 = ols(nmpg,x3,'nmpg',['wt','cyl','hp'])
print(model3.summary())
print(model3.b)

#When comparing models, you are looking for a high F and low p-value, 
#indicating that the regression is significant.
#If you compare the F statistic and its associated p-value "Prob (F statistic)" 
#across all three models, you see all three are statistically significant. It 
#is important to note that the resolution for displaying the p-values may not 
#be able to differentiate the significance of some terms. In this case, you 
#should look at the F statistics; again, higher F- tatistics imply more 
#statistical significance. 

#We'll stick with model 3

import pylab as pl

#load the data in
#Calculating the linear dependence between mpg and
[intercept, swt, scyl, shp] = model3.b
Y = swt*wt +scyl*cyl+shp*hp +intercept
resid = nmpg-Y
fig, (ax1,ax2) = pl.subplots(2)
ax1.scatter(Y,resid, color='black')
ax1.set_xlabel('Predicted NMPG')
ax1.set_ylabel('Residual')
ax2.hist(resid, 40, color='#5D5166')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')
ax1.set_title('Weight Model')
pl.show()

#We can see the residual plot has some shape to it- it's biased and heteroscedastic

#%% Using model 3 to predict NMPG

from ols import ols
import numpy as np
import pylab as pl
# read in the CSV data as a structured array
# names=True will assign column names using the file header
# header: 'nmpg', 'cyl', 'disp', 'hp', 'wt', 'accel', 'yr', 'origin','name'
data = np.genfromtxt('4-2-3 ships.csv', delimiter=",",names=True, dtype="f8,i8,f8,f8,f8,f8,i8,i8,U25")
newdata = np.genfromtxt('4-2-3 FarrisShips.csv', delimiter=",", names=True, dtype="f8,i8,f8,U25")
    
# structured arrays can be conveniently indexed by their column names
wt = data['wt']
cyl = data['cyl']
hp = data['hp']
nmpg = data['nmpg'].transpose()
# create an input matrix with wt, cyl, and hp as inputs
# and train a multivariate model of NMPG with those inputs
x = np.array([wt,cyl,hp]).transpose()
model = ols(nmpg,x,'nmpg',['wt','cyl','hp'])
newwt = newdata['wt']
newcyl = newdata['cyl']
newhp = newdata['hp']
newconstant = np.ones([3])
# build a new input matrix from the newdata array
# see above to access data from a structured array
# and use it to build an input matrix
newx = np.array([newconstant, newwt, newcyl, newhp]).transpose()
# taking the dot product of the input matrix and the
# model parameters will yield predictions for new ships
prediction = np.dot(newx,model.b)
# print the model predictions
print(newdata['name'])
print(prediction)

pl.figure()
pl.scatter(wt,nmpg,color='#FF971C')
pl.scatter(newdata['wt'],prediction,color='#5D5166')
pl.title('Weight vs NMPG')
pl.show()

pl.figure()
pl.scatter(hp,nmpg,color='#FF971C')
pl.scatter(newdata['hp'],prediction,color='#5D5166')
pl.title('Horsepower vs NMPG')
pl.show()

#neither of these shows outliers
