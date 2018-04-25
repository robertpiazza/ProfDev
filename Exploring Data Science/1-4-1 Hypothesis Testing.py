# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:02:30 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Hypothesis Testing Section 1.4.1

#Learn about the hypothesis testing and different ways of performing 
#normalizations. Specifically, the standard score (z-score), feature scaling, 
#feature scaling, gradient descent, t-statistic, and t-test

##Standard Score (z-score)

#The z-score for a datapoint is how many standard deviations a value is from 
#the mean. calculated by subtracting the data point by the mean of the data and 
#dividing by the standard deviation of the data. A positive z-score means that 
#the value is greater than the mean and a negative value means that that the 
#value is below the mean

#Load data of physical measurements of wheat kernels
import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt('1-4-1 seeds_dataset.txt')

#there are three different types of seeds in the data set, listed in the last column
#for now, we're just dealing with seed type 1
isone = data[:,7]==1 #index of seeds of type 1
data1 = data[isone,:]

#histogram of the first attribute, which is the area of the seed

plt.hist(data1[:,0],20, facecolor='#5D5166') #first column, 20 buckets, purple
plt.gca().set(xlabel='seed area', ylabel='frequency') #graph labeling-gca-get current axes
plt.show()

#z-score manually by subtracting mean and dividing by standard deviation
#z-score of each element of the type 1 seed's area
zAreaA = (data1[:,0]-np.mean(data1[:,0]))/np.std(data1[:,0])

print(zAreaA)

#Another way is to use ScipPy's stats package
import scipy.stats as stats
zAreaB = stats.mstats.zscore(data1[:,0])

#same result
print(zAreaA-zAreaB)

#The latter method has the advantage of being able to perform the z-score on 
#all the attributes at once.

zdata1 = stats.mstats.zscore(data1)

#new histogram of the data
plt.hist(zdata1[:,0],20, facecolor='#5D5166')
plt.gca().set(xlabel='standardized seed area', ylabel='frequency')
plt.show()

#new mean and standard deviation:
print([np.mean(zdata1[:,0]),np.std(zdata1[:,0])])
#[1.7541523789077474e-15, 1.0]- essentially 0 and 1


##Feature Scaling-
#Feature scaling ensures the values in our data will be contained in a 
#definite range, something that the z-score cannot do. Usually, data is scaled 
#to be in the range of 0 to 1

#kgl1_scaled = (kgl1 - min(kgl1))/(max(kgl1)-min(kgl1))
#print(kgl1_scaled)

#If you need to preserve the sign of the data, you can divide by the maximum of 
#the absolute values and ensures that they fall within the range of -1 to 1

#useful for datasets with lots of dimensions where some of the dimensions are 
#much larger and will throw off distance calculations or bias the algorithm towards them

##Example Feature Scaling Applied to Gradient Descent

#Normalization is typically very important for gradient descent
#called feature scaling
#Needs to be done if the features are more than a couple orders of magnitude 
#different from one another

#Let's run an animation of gradient descent WITHOUT feature scaling

import numpy as np
import pylab as pl
import matplotlib.animation as animation
# load data
data=np.loadtxt(open("1-4-1 Regression2D.csv","rb"),delimiter=",",skiprows=0)
# number of data points
m = data.shape[0]
# set up data matrix, first two columns in 'data' are independent variables
# we are forcing the regression through zero, i.e. there is no constant term
X = np.ones((m,2))
X[:,0] = data[:,0]
X[:,1] = data[:,1]
# these are the target values
y = np.reshape(data[:,2],(m,1))
# exact solution of linear regression using the normal equations
xxi = np.linalg.inv(X.transpose().dot(X))
xy = X.transpose().dot(y)
a = xxi.dot(xy)
# evaluate cost as a function of values of a1 and a2 and store into a matrix
ndiv = 21
a1vec = np.linspace(0.5*a[0],1.5*a[0],ndiv)
a2vec = np.linspace(0.5*a[1],1.5*a[1],ndiv)
A1,A2 = np.meshgrid(a1vec,a2vec)
cost = np.zeros(A1.shape)
for i in range(ndiv):
    for j in range(ndiv):
        A = np.array([[a1vec[i]],[a2vec[j]]])
        YY = X.dot(A)-y # error between model and data
        cost[i,j] = YY.transpose().dot(YY) # error squared = cost
# initial values of a1 and a2 used in the gradient descent
a10 = 1.6
a20 = 0.00026
# set up the figure
f=pl.figure()
pl.contour(A1,A2,cost,np.logspace(np.log10(15),np.log10(115),20))
pl.plot(a[0],a[1],'rx',markersize=15)
h, = pl.plot([],[],'ok')
line, = pl.plot([],[],color=(0.5,0.5,0.5))
pl.xlim((a1vec[0],a1vec[-1]))
pl.ylim((a2vec[0],a2vec[-1]))
pl.xlabel('a1')
pl.ylabel('a2')
### Make changes to the parameters below.
mu = 0.00001 ### learning rate
nsteps = 100 ### number of gradient descent steps to perform
dt = 500 ### time between frames in animation (in ms), increase to slow down
# precalculate values needed for computing gradient
xx = X.transpose().dot(X)
xy = X.transpose().dot(y)
# initialize parameters
A = np.array([[a10],[a20]])
# precompute gradient for the animation
a1vals = np.zeros(nsteps)
a2vals = np.zeros(nsteps)
a1vals[0] = a10
a2vals[0] = a20
for i in range(1,nsteps):
    gradA = 2*(xx.dot(A) - xy)
    A = A - mu*gradA
    a1vals[i] = A[0]
    a2vals[i] = A[1]
# used to initialize first frame of animation
def init():
    h, = pl.plot([],[],'ok')
    line, = pl.plot([],[],color=(0.5,0.5,0.5))
    return h,
# animation function
def animate(i):
    global a1vals, a2vals
    h.set_data([a1vals[i]],[a2vals[i]])
    line.set_data(a1vals[0:(i+1)],a2vals[0:(i+1)])
    pl.title('step %d'%i)
    return h,
# perform the animation
anim = animation.FuncAnimation(f, animate, init_func=init, frames=nsteps, interval=dt, blit=True)
pl.show()

##Gradient Descent with Feature Scaling

import numpy as np
import pylab as pl
import matplotlib.animation as animation
# load data
data=np.loadtxt(open("1-4-1 regression2D.csv","rb"),delimiter=",",skiprows=0)
# number of data points
m = data.shape[0]
# set up data matrix, first two columns in 'data' are independent variables
# we are forcing the regression through zero, i.e. there is no constant term
X = np.ones((m,2))
X[:,0] = data[:,0]
### SECOND FEATURE IS SCALED BY ITS MAXIMUM
X[:,1] = data[:,1]/max(data[:,1])
# these are the target values
y = np.reshape(data[:,2],(m,1))
# exact solution of linear regression using the normal equations
# don't worry if you don't understand the math here
xxi = np.linalg.inv(X.transpose().dot(X))
xy = X.transpose().dot(y)
a = xxi.dot(xy)
# evaluate cost as a function of values of a1 and a2 and store into a matrix
ndiv = 21
a1vec = np.linspace(0.5*a[0],1.5*a[0],ndiv)
a2vec = np.linspace(0.5*a[1],1.5*a[1],ndiv)
A1,A2 = np.meshgrid(a1vec,a2vec)
cost = np.zeros(A1.shape)
for i in range(ndiv):
    for j in range(ndiv):
        A = np.array([[a1vec[i]],[a2vec[j]]])
        YY = X.dot(A)-y # error between model and data
        cost[i,j] = YY.transpose().dot(YY) # error squared = cost
# initial values of a1 and a2 used in the gradient descent
a10 = 1.6
a20 = 2.8
# set up the figure
f=pl.figure(figsize=(6,6))
pl.contour(A1,A2,cost,np.logspace(np.log10(15),np.log10(115),20))
pl.plot(a[0],a[1],'rx',markersize=15)
h, = pl.plot([],[],'ok')
line, = pl.plot([],[],color=(0.5,0.5,0.5))
pl.xlim((a1vec[0],a1vec[-1]))
pl.ylim((a2vec[0],a2vec[-1]))
pl.xlabel('a1')
pl.ylabel('a2')
### Make changes to the script here.
mu = 0.001 ### learning rate
nsteps = 120 ### number of gradient descent steps to perform
dt = 10 ### time between frames in animation (in ms), increase to slow down
# precalculate values needed for computing gradient
xx = X.transpose().dot(X)
xy = X.transpose().dot(y)
# initialize parameters
A = np.array([[a10],[a20]])
# precompute gradient for the animation
a1vals = np.zeros(nsteps)
a2vals = np.zeros(nsteps)
a1vals[0] = a10
a2vals[0] = a20
for i in range(1,nsteps):
    gradA = 2*(xx.dot(A) - xy)
    A = A - mu*gradA
    a1vals[i] = A[0]
    a2vals[i] = A[1]
# used to initialize first frame of animation
def init():
    h, = pl.plot([],[],'ok')
    line, = pl.plot([],[],color=(0.5,0.5,0.5))
    return h,
# animation function
def animate(i):
    global a1vals, a2vals
    h.set_data([a1vals[i]],[a2vals[i]])
    line.set_data(a1vals[0:(i+1)],a2vals[0:(i+1)])
    pl.title('step %d'%i)
    return h,
# perform the animation
anim = animation.FuncAnimation(f, animate, init_func=init, frames=nsteps, interval=dt, blit=True)
pl.show()
#0.01 was the highest value of mu for it to converge

#if we scale one feature by dividing it by 10000, then our parameter for that 
#feature will be 10000 times greater than it would without feature scaling. To 
#convert back to the unscaled units, simply divide the parameter by 10000 
#(assuming a linear model, non-linear models have to be appropriately scaled).

##T-statistic and T-tests

#Example - a widget uses expensive material but if it's weight is below a 
#threshold, it must be thrown out. We don't want to waste material but also not
# throw out the expensive widgets
#Threshold for throwing out is 1000kg, want our widgets to come in with a mean 
#of 1000.1kg. Measuring the widgets is expensive so we take a sample and 
#measure 1000.09, 1000.07, 1000.08, 1000.11- should we worry or is this probable
#given our sample size? t-test will tell us
#for t-test, we need the t-statistic: t_alpha = (b-b_0)/(standard error(s.e.)(b))
#b is the parameter we're trying to estimate (mean of the widgets)
#b_0 is the value we are testing against- 1000.1kg
#The standard error is defined as the sample standard deviation divided by 
#the square root of the sample size

#t-statistic is basically the difference between the sample mean and some 
#reference value normalized by the standard error

#t-statistic will be high when the sample mean and reference value are very 
#different from one another

#However, even if they are very close to one another, if our standard error is 
#very small, the t-statistic can still be high

#Inversely, if the difference between mean and reference are large and the 
#standard error is large, the t-statistic can be small

#standard deviation is a descriptive statistic that tells us how 
#widely-scattered our data is; standard error is an estimate of how confident 
#we are about the mean of our data
#the standard error is the standard deviation of the distribution of sample means
#(central limit theorem)

#Calculating the t-statistic

import numpy as np
import scipy.stats as stats
x1 = np.loadtxt(open('1-4-1 Damper1.csv','rb'))
nsamples = len(x1)
se1 = np.std(x1,ddof=1)/np.sqrt(nsamples)
tstat = (np.mean(x1)-1000.1)/se1 #-0.066

#negative so sample mean is below target mean
#t-statistics is now a t-distribution
#assumes:
#The sample mean should be normally distributed, which is usually the case 
#because of the central limit theorem
#standard error should be chi-squared distributed
#mean and the standard error should be statistically independent

#t-tests are generally robust to the first two assumptions
#last assumption is more subtle and often requires great care in how data are 
#collected

#For example, if our data were collected over a long time period and 
#production was changing over time, then the last assumption would be violated
#The more samples, the more the t-distribution looks like a normal distribution
#(higher degrees of freedom(DOF)- 10 sample t-distribution has 9 DOF)

#null hypothesis--that the population mean underlying our sampled data is 
#identical to the target value--is true, then the t-value and the (cumulative) 
#t-distribution tells us the probability of obtaining our current results

#CDF of -.066 is .47- about half the distribution- shouldn't be worried
#people begin to doubt the veracity of the null hypothesis when p=0.05 or less
#Some people are more stringent requiring p to be 0.01 or less

#Depending on the relative costs of anomalous production versus the cost of 
#troubleshooting, we may decide on a higher or lower threshold. If, for 
#example, faulty production will cost us much more than fixing the problem, 
#we may opt to take action at higher p-values. Inversely, if troubleshooting is 
#very costly, we may opt for a smaller threshold for p before we stop production

###One very important note about p-values: they do not tell us the veracity of 
#a hypothesis. They tell us the probability of obtaining our measured test 
#statistic (or a value more extreme), given the data and assuming the null 
#hypothesis to be true. That is, given a p-value of 0.1 does not mean that we 
#expect the null hypothesis to be right 10 percent of the time. Instead, we 
#expect to obtain our current result or worse 10 percent of the time if the 
#null hypothesis is true

#You perform a t-test to see if your sample mean is significantly less than a 
#target value (one-sided test). You obtain a p-value of 0.32 . We can you 
#properly infer from this that:
#The obtained t-statistic or one that is lower will occur by chance 32 percent 
#of the time
#The obtained t-statistic or one that is higher will occur by chance 68 percent 
#of the time.
#In a one-sided test, the p-value tells you the probability of obtaining the 
#obtained t-statistic or one that is lower by chance. Therefore, getting one 
#that is higher by chance should be 1-p.

##one-sample, one-sided t-test

import numpy as np
import scipy.stats as stats
x2=np.loadtxt(open('1-4-1 Damper2.csv','rb'))
nsamples = len(x2)
# create a t-distribution with nsamples-1 degrees of freedom
target = 1000.1
se2 = np.std(x2,ddof=1)/np.sqrt(nsamples)
tstat2 = (np.mean(x2)-target)/se2
print('tstat2',tstat2)
# t.pdf(x) and t.cdf(x) returns the value of the pdf and cdf at x respectively
t = stats.t(nsamples-1) #create a 9 DOF t distribution
tcdf2 = t.cdf(tstat2)
print('t-cdf-2', tcdf2)

#This was a one sided test
#Usually, we are interested in performing a two-sided or two-tailed test, 
#where you don't care about the sign of the t-value

#We are asking the question: what is the probability of obtaining a t-value of 
#this magnitude given that the null hypothesis is correct

#In a two-tail test, if the t-value ends up close enough to zero, say anywhere 
#in a symmetric region centered at zero that encompasses 95% of the most likely 
#t-values, then we will not reject the null hypothesis

##Two-sample Tests

#We are testing a new process for building our power couplings. To see whether 
#the new process is an improvement, one assembly line was left unchanged, 
#while the other was modified. We then tally the number of power couplings 
#produced per day from each line. We can tell if the modification makes a 
#significant difference with a two sample t-test aka independent t-test
#Not comparing against a constant, but comparing two samples

#There are two different types of two-sample tests, besides one- and two-tailed, 
#which differ based on whether it is believed that the population variances of 
#the two samples are equal or not. How are you supposed to know whether the 
#population variances are equal? In general, this is a difficult question. 
#Sometimes, there are a priori reasons to believe the variances are equal or 
#unequal (e.g., one recording instrument is noisier than another). If you have 
#normally distributed data, then an F-test is one way to test for equal 
#variances. We might just assume unequal variances from the outset. However, 
#the Welch's t-test, the version used for unequal variances, relies on 
#approximations, reducing its accuracy, and can lead to biases. Therefore, one 
#should be judicious in choosing the test, especially when great sensitivity is 
#required.

#t-stat = (mean(x1)-mean(x2))/((sqrt(1/n1 + 1/n2)*sqrt(((n1-1)*s_1^2+(n2-1)*s_2^2)/(n1+n2-2)) with a degree of freedom of n1+n2-2
#if n1 = n2 = n, this simplifies to: 
#(mean(x1)-mean(x2))/sqrt((s_1^2+s_2^2)/n) with a DOF of 2n-2

#Paired Two-sample Tests
#Experimental influences and noisiness affect both samples equally

#TL:DR
#The last type of t-test that we are going to explain is the paired t-test. In our power coupling example, to test the new process versus the old, we might collect data for x number of days then test the mean between them in a two-sample test. However, our factory conditions might change from day-to-day. The workforce may be depleted in one part of the study because Regina Fever, shipment delays may have bottlenecked production, or there may be "seasonality" effects (more workers take vacation near the weekends and holidays), etc. If we designed our study correctly, the above effects should influence both processes equally, on average; however, they will add to the variance in our production counts, making them seem far noisier than they really are. This, in turn, will tend to reduce our t-values, making it less likely that we will see significant differences between the processes. 
#Another example where paired t-tests should be used: a test to determine if a medicine truly reduces cholesterol. The pairs of data in that case are before and after treatment for each patient. By not pooling the data together, the between patient noise is eliminated.
#The good thing about paired t-tests is that they operate exactly as one-sample t-tests; simply take the difference between the pairs of data and perform a one-sample t-test on the differences. If you are testing only for significant differences between the means, then you set the target value to 0; however, you can test that the difference between the two is any value by appropriately setting the target.

##Using Packages

#scipy.stats

#To perform a two-sided, one-sample test
#stats.ttest_1samp(data,targetvalue)

#To perform a two-sided, two-sample unpaired (independent) test assuming equal 
#population variances:
#stats.ttest_ind(data1,data2)

#Assuming unequal variances (Welch t-test):
#stats.ttest_ind(data1,data2, equal_var=False)

#To perform a two-sided, two-sample paired (dependent) test
#stats.ttest_rel(data1,data2)

#You are performing a study to determine which of two exercise regiments leads 
#to the most weight loss. You randomly assign participants to one of the two 
#regiments, with equal numbers in each group. For each participant, you collect 
#his or her weight at the start of the study and at the end of the study. 
#After collecting the data, you want to see if there is a statistical 
#difference in the weight loss between the two groups.
#Use a two-sample t-test

#In the same study, you want to determine if participants in the first group 
#lost more than 5 lbs, on average, before and after the regiment.
#Use a paired two-sample t-test

import numpy as np
import scipy.stats as stats
f = open('1-4-1 PowerCouplings.csv','rb')
data = np.loadtxt(f, delimiter=',')
day = data[:,0]
x1 = data[:,1]
x2 = data[:,2]
plt.scatter(day,x1, s = 40, color = '#5D5166', marker = 'D', label = "Old")
plt.scatter(day,x2, s = 40, color = '#FF971C', marker = 'o', label = "New")
plt.title('Old and New Processes')
plt.xlabel('day')
plt.ylabel('Number')
plt.axis([0, 30.1, 960, 1100])
plt.legend(loc = 'lower left')
plt.show()
#Treat the data as being independent (unpaired) and having equal population variance.
t_ind,p_ind=stats.ttest_ind(x1,x2)
print('t-stat=',t_ind,'p-value=',p_ind) #p-value is 0.230-not significant(>0.05)

#Now, perform a two-tailed test, assuming the data are paired
t_ind,p_ind=stats.ttest_rel(x1,x2)
print('t-stat=',t_ind,'p-value=',p_ind) #p-value is 0.043-significant (<0.05)

