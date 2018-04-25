# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:34:49 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

###Foundations-Distance and Similarity Metrics Section 1.2.1

#It is important that we know about distance metrics and similarity measures
#Learn about various distance and similarity metrics that are useful in data 
#science techniques such as: clustering, classification, and recommendation

#For this mission, become familiar with the Euclidean, Cosine, Jaccard, 
#Mahalanobis, and Levenshtein metrics.

##Euclidean Distance

#most common distance metric and is simply the length of the line segment
#connecting two vectors in a vector space
#The Euclidean distance is the n-dimensional generalization of the 
#Pythagorean Theorem
#d(a,b) = sqrt(summation i to n (ai-bi)^2) where ai and bi are the ith elements
#in the n-dimensional vectors
#we can easily compute the Euclidean distance between two vectors in NumPy
#np.linalg.norm(x - y)
import numpy as np
np.random.seed(1000)
# generate a list of random vectors
x = [np.random.rand(1,50) for _ in range(50)]
# set z equal to the origin
z = np.zeros(50)
### find the distance between all x_i and z and find the min of those distances
d = [np.linalg.norm(y - z) for y in x]
print(min(d))
#Euclidean problems:
#Inconsistent units across dimensions-when we are computing the distance 
#between multiple features with different units of measure, the features with 
#the largest values tend to account for more of the Euclidean distance
#Upward bias in dimensionality-Euclidean distance is biased upward by the 
#number of dimensions, n, in the feature vector. This means that as we increase
#our number of variables, observations tend to spread out, which increases the 
#difficulty of identifying or clustering similar observations
#rounding errors on a computer or noisy measurements of the same underlying 
#quantity can cause tiny differences, which build into large changes in the 
#Euclidean distances between two high-dimensional vectors.

#What is the Euclidean distance between a vector of ones in a space with 
#1.225 billion dimensions and the origin
#1225000000**.5
print(np.sqrt(1.225 * 10**9)) #35000

##Cosine Similarity

#measures the similarity between two vectors (observations) as the cosine of 
#the angle between them derived from the dot product a dot b = mag(a) * mag(b) * cos(theta)

#modify the cosine distance function

#1 + scipy.spatial.distance.cosine(x, y)
#We add "1" for rescaling purposes, since SciPy's function returns the distance 
#(by computing 1 - cosine similarity) rather than similarity

#Could also do it ourselves with:
#np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

#import numpy as np
# read data and set each row to corresponding movie
data = np.genfromtxt('1-2-1 star_wars_tfidf.csv', delimiter =',')
newHope = data[0]
empire = data[1]
returnJedi = data[2]
# define a cosine similarity function
def cosSim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
## use the cosSim function to find which movies have the most similar script
print(cosSim(newHope, empire)) #max with .999
print(cosSim(newHope, returnJedi))
print(cosSim(empire, returnJedi))
#range of values for the cosine similarity is -1 to 1

##Jaccard Index and Jaccard Distance

#The Jaccard Index, also known as the Jaccard Similarity Coefficient, is 
#designed to measure the proportion of unique data points that exist in two 
#sets A and B

#J(A,B) = (points in both A&B)/(A union B {points in just + points in just B + points in both A and B})
#bounded by [0,1]
#dissimilarity between the sets A and B, known as the Jaccard Distance, 
#which according to De Morgan's Law, is 1 - J(A,B)

#Jaccard index can be computed using the following lines of code
#def jaccardIndex(A, B):
#    A = set(A)
#    B = set(B)
#    num = len(A.intersection(B))
#    return (float(num) / (len(A) + len(B) - num))
#or:
#scipy.spatial.distance.jaccard(x, y)
    
#Which screen crawls have the most number of phrases in common?

#import numpy as np
import scipy
# open and read in data
with open('1-2-1 starWarsWordBag.csv', 'r') as crawls:
    newHope = np.array(crawls.readline().split(','))
    empire = np.array(crawls.readline().split(','))
    jedi = np.array(crawls.readline().split(','))
print(newHope)
print(empire)
print(jedi)
def jaccardIndex(A,B):
    A = set(A)
    B = set(B)
    num = len(A.intersection(B))
    return (float(num) / (len(A) + len(B) - num))
print(jaccardIndex(newHope,empire))
print(jaccardIndex(newHope,jedi))
print(jaccardIndex(empire,jedi))
#Jaccard distance is 1-Jaccard index
print(1-jaccardIndex(newHope,jedi))

#Jaccard Distance  is most often used as a simple similarity matching metric. 
#An obvious shortcoming of the metric is that it fails to capture the relative 
#frequency and weights (i.e., importance) of observations in the sets

##Mahalanobis Distance

#Mahalanobis distance is a generalization of the Euclidean distance, 
#which addresses differences in the distributions of feature vectors, as well 
#as correlations between features

#It uses the covariance matrix of the two vectors:
#The inverse of the covariance matrix is used to transform the data so that 
#each feature becomes uncorrelated with all other features and all transformed 
#features have the same amount of variance, which eliminates the scaling issues 
#present in Euclidean distance

import numpy as np
# read in data, skip the last column which contains class 
data = np.genfromtxt('1-2-1 Jedi.data', delimiter=',', usecols=range(4))
### calculate the covariance matrix of data
#You want NumPy to treat each row as an observation. 
#Each column represents a variable. Make sure rowvar = 0.
covMatrix = np.cov(data, rowvar=0)
print(covMatrix)
#inverse:
invCov = np.linalg.inv(np.cov(data, rowvar=0))
#first and second observations
x = data[0]
y = data[1]
### use x, y, and invCov to calculate the Mahalanobis distance
np.sqrt(np.dot(np.dot((x-y),invCov),(x-y)))
#alternately:
scipy.spatial.distance.mahalanobis(x,y,invCov)
#The Euclidean and Mahalanobis distances are equal when the features in the 
#set are independent of one another and all features have unit variance

##Levenshtein Distance

#The Levenshtein distance gives us a way of computing the edit distance 
#between two strings 
#It is closely related to the "edit distance." 
#Where the Levenshtein distance assigns a uniform cost of 1 for any operation, 
#edit distance assigns an arbitrary cost to addition, deletion, and substitution 
#operations. This allows a user to customize the priorities of certain types of 
#edits

#The Levenshtein distance is an undesirable algorithm for large-scale applications.

#Bottom-Up Approach
#python-Levenshtein package
#edit table to create the distances
#cell by cell, if the two letters are the same, the the top left corner value as is, if not, of the left, top and top left cells, select the minimum and add one. 
A = 'quicksort'
B = 'wuieksoobt'
import Levenshtein as lv
print('Levenshtein Distance from', A, 'to', B, 'is', lv.distance(A,B))






