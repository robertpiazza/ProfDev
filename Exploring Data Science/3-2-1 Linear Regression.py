# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:44:22 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Discover-Linear Regression Section 3.2.1

#determine which variables are important predictors 

#use linear regression and t-statistics to identify strong predictor variables

#Linear regression with ordinary least squares has some pretty cool statistical 
#properties that can tell us several things useful in determining which variables 
#are important. We'll be using the F test and t-statistics in the analysis to 
#follow

##Notes on Regression Evaluation Criteria
#Residual Analysis
#plot showing residuals vs. fitted values, which plots the residuals of the 
#model (y−yHat)vs. the model results (yHat). The residuals should be independent 
#of the fitted values, which would display a random pattern. If a pattern, the 
#model is not capturing all signal present in the data
#QQ plot
#quantiles of the residuals vs. the theoretical quantiles of the normal 
#distribution. If the residuals were normally distributed, the dots would appear 
#close to the dashed line along the whole line
#Scale Location plot
#The scale-location plot is similar to the residuals vs. fitted values plot 
#except that it uses the square root of the residuals

#F-Test
#The F-test evaluates the hypothesis that the model is statistically different 
#than just using the average of all the data as the model output. In other 
#words, the F test checks that a change in independent variables will result in 
#a change in the dependent variable. The important thing to note is the p-value. 
#The smaller the p-value, the more likely there is pattern in your data. 
#Generally, a p-value less than 0.05 (equivalently, 95% confidence) indicates a 
#statistically significant model.

#T-Test
#The T-test evaluates the hypothesis that the coefficient is statistically 
#different than zero. Generally, a p-value less than 0.05 (equivalently, 95% 
#confidence) indicates that a particular coefficient is statistically 
#significant.

#R-Squared Statistics
#The R-squared statistic measures the squared correlation between the model 
#predictions and the dependent variable. We want to balance model performance 
#with simplicity to avoid fitting the noise in the data instead of the signal.

#Model selection- https://en.wikipedia.org/wiki/Model_selection

#backward elimination, since we are starting with a full model and eliminating 
#variables as we go

#How to select: First, make sure your residuals meet the above criteria. 
#Basically, we don't want to see any pattern in the residuals.

#take a look at the F statistic, which tells us if the whole model is 
#statistically significant or not. A high F statistic indicates model 
#significance

#We want to see a low p-value, which indicates that there is little chance the 
#high F statistic is caused by noise and not signal

#Finally, look at the t-statistic for each variable. The t-statistic should be 
#far away from zero (either positive or negative), which indicates that the 
#variable coefficient is statistically different from zero. A metric for 
#deciding how far should the t-statistic be from zero can be found by looking 
#at the associated p-value. The p-value measures the probability that 
#t-statistic is caused by noise and not signal. Therefore, a lower p-value 
#indicates a significant variable
#A good rule of thumb is that p-values less than 0.05 indicate significance

#Following Model results:
#==============================================================================
#Dependent Variable: log gestation
#Method: Least Squares
# obs: 58
# variables: 5
#==============================================================================
#variable coefficient std. Error t-statistic prob.
#==============================================================================
#const 1.105823 0.162800 6.792532 0.000000
#BodyWt 0.000001 0.000024 0.038572 0.969377
#BrainWt -0.000043 0.000026 -1.647590 0.105354
#lifeSpan 0.410215 0.001594 257.271030 0.000000
#totalSleep -0.043144 0.008661 -4.981686 0.000007
#==============================================================================
#Models stats Residual stats
#==============================================================================
#R-squared 0.999278 Durbin-Watson stat 2.441089
#Adjusted R-squared 0.999224 Omnibus stat 1.165717
#F-statistic 18345.403826 Prob(Omnibus stat) 0.558300
#Prob (F-statistic) 0.000000 JB stat 1.022569
#Log likelihood -16.162422 Prob(JB) 0.599725
#AIC criterion 0.729739 Skew -0.110757
#BIC criterion 0.907363 Kurtosis 2.388393
#==============================================================================

#The statistics are approximately correct because the Central Limit Theorem and 
#certain assumptions on residuals are only slightly violated. Tip: Strong 
#patterns in residuals violate some of the statistical assumptions of linear 
#regression, which results in biased estimates for the F and t-statistics. The 
#pattern in our regression residuals display a bit of drift where errors 
#increase as gestation increases although the variance of the errors remains 
#constant. Since the pattern is slight and there are sufficiently many 
#observations, the Central Limit Theorem ensures that the F and t-statistics 
#will be good asymptotic approximations (more accurate as the sample size 
#increases).

#the regression statistically significant because the p-value is low enough

#BodyWt and BrainWt are condidates for removal because these variables had 
#p-values higher than the general rule of thumb 0.05 threshold


#Now we have removed BodyWt and BrainWt from the model:

#==============================================================================
#Dependent Variable: log gestation
#Method: Least Squares
# obs: 58
# variables: 3
#==============================================================================
#variable coefficient std. Error t-statistic prob.
#==============================================================================
#const 1.011665 0.135114 7.487517 0.000000
#lifeSpan 0.410007 0.001563 262.272350 0.000000
#totalSleep -0.044435 0.008658 -5.132034 0.000004
#==============================================================================
#Models stats Residual stats
#==============================================================================
#R-squared 0.999241 Durbin-Watson stat 2.459823
#Adjusted R-squared 0.999213 Omnibus stat 0.480788
#F-statistic 36191.295784 Prob(Omnibus stat) 0.786318
#Prob (F-statistic) 0.000000 JB stat 0.626655
#Log likelihood -17.633050 Prob(JB) 0.731010
#AIC criterion 0.711484 Skew -0.171712
#BIC criterion 0.818059 Kurtosis 2.624014
#==============================================================================

#Again, The statistics are approximately correct because the Central Limit 
#Theorem and certain assumptions on residuals are only slightly violated.

#The regression statistically significant because the p-value is low enough.

#None of the variables are candidates for removal


