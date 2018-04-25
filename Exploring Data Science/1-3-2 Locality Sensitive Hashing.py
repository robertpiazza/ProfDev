# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:30:44 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Locality Sensitive Hashing Section 1.3.2

#Locality Sensitive Hashing is particularly useful in comparing high-dimensional data in a very fast and efficient manner and has uses in multimedia indexing, retrieval, and similarity search. In this mission, all you need to do is reduce the dimensionality
#It is a fast and highly-scalable way to reduce the dimensionality of the data

#Hash functions are a mathematical operator that converts data of arbitrary 
#length to a fixed length. LSH is an extension of hash functions that create 
#similar hashes for similar data with a high probability. Sometimes they are 
#called probabilistic hashes. 

#two most popular types of locality sensitive hashes are 
#MinHash and Random Projections

#MinHash can only be applied when all attributes are binary such as in 
#bag-of-words representations of text documents where only presence and 
#absence of a word is stored-The goal of MinHash is to estimate J(A,B) quickly, 
#without explicitly computing the intersection and union

#Random projections can be applied to binary, integer, and continuous attributes
#we draw a bunch of random lines through the data, which form partitions in the 
#data set. Each partition gets an ID, which becomes the hash of that part of 
#the data space.

#X^(RP)_n x k = X_n x d * R_

#Applying LSH I

#Shape of mixed spectra matrix

import numpy as np
from sklearn import random_projection
# load data from csv into numpy arrays
data = np.genfromtxt('1-3-1 mixed_spectra_matrix.csv', delimiter=';')
print('original data shape', data.shape) #360 rows, 1300 columns
#Now we'll multiply our data matrix with a random Gaussian matrix to generate 
#the lower dimensional random projection

#Applying LSH II
# apply a Gaussian random projection
tx = tx = random_projection.GaussianRandomProjection(n_components=25)### create the transformation object
data25 = tx.fit_transform(data) ### create the transformation
print('new data shape', data25.shape)

#Applying LSH III

# first define a transformation function to apply to each element
def tx01(x):
    return 0 if x < 0 else 1

# apply tx01 to each element using vectorize
vtx01  = np.vectorize(tx01)### vectorize the tx01 function
data01 = vtx01(data25) ### apply function to each element in data25

##Compare the two data sets:
print('original')
print(data) #(360, 1300)
print('25 data')
print(data25) #(360, 25)
print('01 data')
print(data01) #(360, 25)
# How many 1's in the nth row? 
#print np.sum(data01[n-1,:])

##Now that you created hashes of your data stored in the rows of the data01 
#matrix, you can use those hashes as keys in a database to store all original 
#data sorted by the LSH hash values. Doing so enables you to query your data 
#using an example (such as: find all images similar to this one), then find all 
#records with the same LSH hash as your query example. Since the hashes are in 
#sorted order, the database can find those records efficiently and fast

#Locality Sensitive Hashing techniques are useful when the size of data makes 
#computing the SVD too computationally intensive or intractable. LSH techniques 
#are a good choice for comparing high dimensional data in a very fast and 
#efficient manner with applications in multimedia indexing, retrieval, and 
#similarity search. LSH is often used as a filtering step where all observations 
#with the same LSH hash are then compared to each other using the full 
#dimensional representation. In this way, LSH helps us spend expensive 
#computation only on observations that are likely to have a high degree of 
#similarity. You might need to experiment a bit with the number of hyperplanes 
#to get the right balance of speed and accuracy
