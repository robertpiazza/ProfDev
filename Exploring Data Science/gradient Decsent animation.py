# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:59:10 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

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