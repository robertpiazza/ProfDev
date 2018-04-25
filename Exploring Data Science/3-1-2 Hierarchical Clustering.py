# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 08:54:40 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Discover-Hierarchical Clustering Section 3.1.2

#Understand the relationships between morphological features using clustering
#Clustering is an unsupervised technique that groups together data points based 
#on similarity as measured by distance 

#Hierarchical clustering is a method that groups data by iteratively merging 
#together pre-existing groups of data

#TL:DR:
#At each iteration, the clusters with the smallest distance between them are merged together to form a new cluster. This continues until there is only a single cluster. The benefit of this method over other clustering methods is that it does not require you to specify the number of clusters ahead of time (K-means). This provides a view of how the data are related to each other. This is useful if you are interested in the relationships between elements, not just their overall cluster. Elements that are more similar will form clusters at earlier rounds of clustering than elements that are less similar.

#Constructing a dendogram

import numpy as np
import scipy
import pylab as pl
from matplotlib.pyplot import show
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
#load the data
data = np.genfromtxt('3-1-2 zoo4.csv',delimiter=',',names=True)
labels = np.genfromtxt('3-1-2 zooLabels3.csv',delimiter=',',names=True,dtype=None)
#load the data into a numpy array
Xdata=np.array([data['hair'],data['feathers'],data['eggs'],data['milk'],data['airborne'],data['aquatic'],data['predator'],data['toothed'],data['backbone'],data['breathes'],data['venomous'],data['fins'],data['legs'],data['tail'],data['domestic'],data['catsize']]).transpose()
#transpose the labels
labs = np.array(labels['name']).transpose()
Z = scipy.cluster.hierarchy.average(Xdata)
dn = dendrogram(Z)
pl.show()