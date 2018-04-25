# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:45:19 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Discover-K-means Clustering Section 3.1.1

#Objective: To understand K-means clustering and apply it To understand K-means 
#clustering and apply it 

#Clustering is an unsupervised learning method
#algorithms like neural networks and support vector machines, which are 
#supervised learning methods

#Unsupervised means that there is no labeled data from which to train. The 
#algorithm learns the labels based on the 'shape' of the data

#One of the most basic (and widely used) techniques for clustering is the 
#K-means algorithm. K-means starts with an initial guess as to how many natural 
#groupings reside in the set of data, and iterates over the data until each 
#data point is assigned membership in one of the classes based on proximity to 
#a group center. A cost function based on an aggregate distance metric is used 
#to compare the goodness of reassignment of a point to another cluster along the 
#way.

#The K-means algorithm is based on a heuristic procedure, as is true for most 
#unsupervised methods for pattern learning.In most variations of the algorithm, 
#one must specify the number of clusters in the data---that is the origin of 
#the K in K-means---and have a reasonable guess for its value. Otherwise, the 
#algorithm may fail to converge or will produce nonsensical results.

#Pseudocode:
#Input: # number of clusters K, data X, max iteration count *maxit* or other halting criterion.

#Initialize cluster centroids { μ_1, μ_2, … , μ_k }. The cluster centroids have the same dimension as the examples in X; they can be chosen randomly (from X) or initialized randomly.  Each centroid has a label associated with it, e.g. 1, 2, 3, etc.

#While not converged:

#- Calculate distance between each data point to each centroid.
#- Assign the data point *x* the same label as its closest centroid. 
#- Update each centroid μ_i using the labeled points (e.g. take mean or median of data points with the same label)
#- Stop if max iterations reached or convergence satisfied.

##Centroids

#The centroid is a point that measures some notion of central tendency of a cluster

#Most commonly, one uses the arithmetic mean but others are possible (median if
#worried about outliers)

#The batch form will wait until all the data has been assigned to the cluster 
#before updating it, whereas the online version can update as soon as each data 
#point is assigned to a centroid. Batch may take longer to converge, but it is 
#guaranteed to move towards a solution. Online may converge faster, but it may 
#temporarily move away from the best solution. 

##Convergence

#One can look to see how much the centroids move between iterations, when they 
#stop moving below a threshold, the algorithm has converged. Equivalently, one 
#can wait until the data points no longer change centroid labels.

import numpy as np
import pandas as pd
df=pd.read_csv('3-1-1 jungle-alien-biometrics.data')
### add code here
print(df.describe())

#Scatter Matrix Plot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_csv('3-1-1 jungle-alien-biometrics.data')
plt.figure()
pd.scatter_matrix(df)
plt.title("Scatter Matrix")
plt.show()
#BodyWidth can separate two groups olong a line of 2.5

from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
data = np.genfromtxt('3-1-1 jungle-alien-biometrics.data',delimiter=',',dtype='f8,f8,f8,f8',names=True)
columns = data.dtype.names
X = np.array((data['WingSpanLength'],data['BodyLength'], data['BodyWidth'],data['FootWidth']))
X = X.transpose()
kmeans = cluster.KMeans(n_clusters= 2,max_iter= 100, random_state=0) 
#Cases where convergence is elusive can come from a variety of reasons and thus
#requires a max iteraction parameter to prevent an infinite loop. A poor choice 
#of K for a given data set is one example; a poor choice of initial centroids 
#is another
kmeans.fit(X)

from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
data = np.genfromtxt('3-1-1 jungle-alien-biometrics.data',delimiter=',',dtype='f8,f8,f8,f8',names=True)
columns = data.dtype.names
X = np.array((data['WingSpanLength'],data['BodyLength'], data['BodyWidth'],data['FootWidth']))
X = X.transpose()
kmeans = cluster.KMeans(n_clusters= 2,max_iter= 100, random_state=0) ### edit this line
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
print(sum(kmeans.labels_==0))
print(sum(kmeans.labels_==1))

#two primary strategies for initialization:
#Random seed: in this method, K seed points are selected from the data X; all 
#the rest of the n - K points in X are aligned with the cluster having the 
#closest seed.
#Random partition: this method assigns each data point into one of the K 
#clusters, at random.



