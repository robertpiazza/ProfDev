# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:45:47 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-K Nearest Neighbors Section 4.1.1

#You study how we can predict the value of one data attribute based on the 
#values of others, using techniques such as classification, regression or 
#recommendation. 

#Classify the animals, based on similarities to other animals,

#Objective: Utilize the k-nearest neighbors (KNN) algorithm to identify 
#similarities between animals

#What's KNN? Given a data point, x, KNN uses information from the k neighboring 
#data points to infer some sort of prediction. In the case of classification, 
#the algorithm predicts class membership of x to be the most frequent class of 
#surrounding neighbors. More complex versions of this algorithm can attach 
#weights to the class membership votes, based on the distance from x. In the 
#case of regression, the predicted value of x is the average of the k neighbors. 
#More complex versions, again, can weight observations by their distance from x. 
#One common weighting scheme is 1/d, where d is the distance between x and its 
#neighbor.

#KNN needs to keep track of the neighbors of every point. Usually this is 
#accomplished using special data structures such as k-dimensional (k-d) trees 
#or ball trees that efficiently index multidimensional keys. K-d trees are more 
#efficient on low dimensional data sets and ball trees are more efficient on 
#high dimensional data sets. As dimensionality becomes larger, exact methods 
#become more inefficient and approximate methods (such as local sensitivity 
#hashing) should be used.

import numpy as np
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
#random seed
random_state = np.random.RandomState(0)
#load the data
data = np.genfromtxt('4-1-1 animalData.csv',delimiter=',',names=True)
#load the data into a numpy array
Xdata=np.array([data['hair'],data['feathers'],data['eggs'],data['milk'],data['airborne'],data['aquatic'],data['predator'],data['toothed'],data['backbone'],data['breathes'],data['venomous'],data['fins'],data['legs'],data['tail'],data['domestic'],data['catsize']]).transpose()
#load yData
yData = np.array(data['type'])
#load the class labels
text_file = open('4-1-1 animalLabels.csv', "r")
#put the class labels into names
names = text_file.read().split('\n')
#create a nearest neighors object using 5 closest neightbors
neighbors = sk.neighbors.NearestNeighbors(n_neighbors=5, algorithm='brute').fit(Xdata)
#find the nearest neighbors and their distances
distances, indices = neighbors.kneighbors(Xdata)
### print out the values
print(indices)
#print(indices)
#print(distances)
### uncomment this
#print all the nearest neighbors
for i in range(0,len(indices)):
#get the distances
    dists = distances[i]
##this is the animal we are searching from
    print(str(names[i])+'\n')
##this is the animals closest neighbors and their distances
    for j in range(0, len(indices[i])):
        print('\t'+str(j)+' '+str(names[indices[i][j]])+' '+str(dists[j]))
   
##Choosing k

#We need to train a KNN classifier and explore the effect that modifying k can 
#have on its performance. A KNN classifier makes predictions with a majority 
#vote of the nearest neighbors. For example, if most of your neighbors are 
#mammals, then you are likely a mammal. A difficult part of using a KNN 
#classifier is deciding the value of k. Using a k value that is too large can 
#include values that are likely unrelated to a particular data point, while 
#using a k that is too small can give individual neighbors too much 
#influence
        
import numpy as np
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
#random seed
random_state = np.random.RandomState(0)
#load the data
data = np.genfromtxt('4-1-1 animalData.csv',delimiter=',',names=True)
#load the data into a numpy array
Xdata=np.array([data['hair'],data['feathers'],data['eggs'],data['milk'],data['airborne'],data['aquatic'],data['predator'],data['toothed'],data['backbone'],data['breathes'],data['venomous'],data['fins'],data['legs'],data['tail'],data['domestic'],data['catsize']]).transpose()
#load yData
yData = np.array(data['type'])
#load the class labels
text_file = open('4-1-1 animalLabels.csv', "r")
#put the class labels into names
names = text_file.read().split('\r')
#create training and test sets, 90% in training 10% in testing set
test_cases = [1,2,3,4,5,10]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(Xdata, yData, test_size=0.90, random_state=random_state)
#create a nearest neighors classifier using 5 closest neightbors
for i in test_cases:
    knnClassifier = KNeighborsClassifier(n_neighbors=i) ### edit this line
#fit the knn model
    knnClassifier.fit(X_train, y_train)
#predict the class for the first element of the test set based on neighbors
    print(knnClassifier.predict(X_test))
#print accuracy of the knn model on a test set
    print(str(i)+'\n')
    print(knnClassifier.score(X_test,y_test))

#Note: Changing k changes how many neighbors are evaluated within the algorithm. 
#If you include more neighbors, each neighbor has less impact. If you have fewer 
#neighbors, each neighbor has more impact on the score. While looking at the 
#single closest neighbor worked well in this example, it might be less 
#effective in other applications you encounter
    
#Which animals are the most dangerous?
    
import numpy as np
import sklearn as sk
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
#random seed
random_state = np.random.RandomState(0)
#load the data
data = np.genfromtxt('4-1-1 animalData.csv',delimiter=',',names=True)
#load the data into a numpy array
Xdata=np.array([data['hair'],data['feathers'],data['eggs'],data['milk'],data['airborne'],data['aquatic'],data['predator'],data['toothed'],data['backbone'],data['breathes'],data['venomous'],data['fins'],data['legs'],data['tail'],data['domestic'],data['catsize']]).transpose()
#load the animal labels
text_file = open('4-1-1 animalLabels.csv', "r")
#put the animal labels into names
names = text_file.read().split('\n')
#create a nearest neighbors object using 5 closest neighbors
neighbors = sk.neighbors.NearestNeighbors(n_neighbors=5, algorithm='brute').fit(Xdata)
#load the data given by the scout team
scoutData = np.loadtxt('4-1-1 scoutData.csv',skiprows=1,delimiter=',').reshape(1,-1)
#find the nearest neighbors and their distances to the scoutData
distances, indices = neighbors.kneighbors(scoutData)
#print all the nearest neighbors
print('\nAnimals most similar to scout data \n')
for i in range(len(indices[0])):
    print("%2f %s" % (distances[0][i], names[indices[0][i]]))

#KNN can be useful for classifying based on similarity. A large advantage of 
#KNN is that it does not require you to train a model. This makes it easy to 
#use right away. However, it is computationally expensive to find the nearest 
#neighbors of a given data point. You may have noticed in the first challenge 
#that sometimes the same animal did not come up as the most similar item. This 
#is because it had the same similarity score as another animal, and the 
#ordering is based on which item is compared first in the case of a tie. 
    
    