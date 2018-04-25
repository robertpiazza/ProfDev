# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:03:48 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Singular Value Decomposition Section 1.3.1

#Learn about a technique called Principal Component Analysis (PCA) and a 
#related, more scalable algorithm called Singular Value Decomposition (SVD).

#Computing the principal components amounts to calculating the covariance 
#matrix of the data and computing the eigenvalues and eigenvectors of the 
#covariance matrix. This gets to be computationally expensive for large 
#numbers of rows and columns in the data matrix. When PCA is computationally 
#expensive, Singular Value Decomposition (SVD) can be used instead

#It breaks apart matrices into combinations of column and row vectors. The 
#column and row vectors each form an orthogonal vector space, so there is no 
#redundant information between pairs. SVD also finds a weight, called a 
#singular value, associated with each combination of column and row vectors. 
#The singular values determine how much each combination contributes to the 
#construction of the final matrix

#Since computation of the SVD is much more computationally efficient than the 
#basic PCA algorithm, SVD is usually used to compute PCA.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #First, we have to import the correct module
# load data from csv into numpy arrays
X = np.genfromtxt('1-3-1 mixed_spectra_matrix.csv', delimiter=';')
#X should have observations along the rows, and features along the columns

# do PCA on X
pca = PCA()### Next, we have to create an object of the PCA class.
pcaX = pca.fit(X)### fit the model by performing PCA on the data
# plot tail off of components
plt.scatter(range(len(pcaX.explained_variance_)), pcaX.explained_variance_, color='#5D5166')
plt.title("Tail off of Principal Components")
plt.xlabel("Index of Principal Component")
plt.ylabel("Magnitude of Eigenvalue")
plt.show()

#this plot shows after about the first 25 principle componenents, very little
#information is gained

#we can see the components with the pcaX.components_ function

#SciKit provides a method to transform the data X into PCA space Y which uses 
#the .components_:
#Y = pca.transform(X)

#NumPy's methods to perform SVD
#U, s, Vh = np.linalg.svd(X)

##SVD 1

#Apply SVD to the data set and determine a reasonable number of components to 
#keep by plotting the explained variance as a function of the number of 
#principal components

import numpy as np
import matplotlib.pyplot as plt
# load data from csv into numpy arrays
X = np.genfromtxt('1-3-1 mixed_spectra_matrix.csv', delimiter=';')
# subtract the mean from all columns of X
X = X - X.mean(1)[:, np.newaxis]
# calculate svd
U, s, Vh = np.linalg.svd(X) ### calculate the SVD of X using numpy
# sizes of matricies
print('U\ts\tVh\n', U.shape, s.shape, Vh.shape)
# plot tail off of components
plt.scatter(range(len(s)), s, color='#5D5166')
plt.title("Tail off of Eigenvalues")
plt.xlabel("Index of Eigenvalue")
plt.ylabel("Magnitude of Eigenvalue")
plt.show()

#SVD II - reduce X to 25 dimensions

import numpy as np
# load data from csv into numpy arrays
X = np.genfromtxt('1-3-1 mixed_spectra_matrix.csv', delimiter=';')
# mean center X
X = X - X.mean(1)[:, np.newaxis]
# calculate svd
U, s, Vh = np.linalg.svd(X)
# get the first 25 concepts
S25 = np.diag(s)[:, :25]## diagonalize s, keep only the first 25 columns
print(S25)
Vh25 = Vh[:25,:] ### keep only the first 25 rows of Vh
X25 = X.dot(Vh25.T) ### use Vh25 to transform X into 25 dimensions
# print X25 and the shape
print(X25)
print('num rows, num cols\n', X25.shape)

#SVD III
#clustering with SVD
#Visualize the results of a clustering algorithm using the first two 
#dimensions of the SVD

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# load data from csv into numpy arrays
X = np.genfromtxt('1-3-1 mixed_spectra_matrix.csv', delimiter=';')
# mean center X
X = X - X.mean(1)[:, np.newaxis]
# calculate svd
U, s, Vh = np.linalg.svd(X)
# cluster
km = KMeans(n_clusters=4)
clusters = km.fit_predict(X)
# get the first 25 concepts
S25 = np.diag(s)[:, :25]
Vh25 = Vh[:25, :]
X25 = X.dot(Vh25.T)
# plot clusters on svd in 2d svd space
# use the first component as the x axis and the second as the y axis
plt.scatter(X25[:,0],  X25[:,1] , c=clusters)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

