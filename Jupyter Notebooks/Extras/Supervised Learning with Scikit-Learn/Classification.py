# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:34:41 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

"Datacamp-Supervised Learning with Scikit-learn: Classification"

#%%Supervised learning

"ML- The art and science of giving computers the ability to learn to make decisions from data"
"Without being explicitly programmed"
"Example: spam/not spam, cluster wikipedia by categories"


#%%Which of these is a classification problem?
"Predicting if a stock will go up or down"
#%%Exploratory data analysis

"Using the iris dataset"

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris = datasets.load_iris()
print(type(iris))
#out: klearn.datasets.base.Bunch
print(iris.keys())
#out: dict_keys(['data', 'target_names', 'DESCR', 'feature_names', 'target'])
print(type(iris.data), type(iris.target) )
#out: <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(iris.data.shape)
#Out[10]: (150, 4))
print( iris.target_names)
#array(['setosa', 'versicolor', 'virginica'], dtype='<U10'))
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print(df.head()) 
pd.plotting.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='x')
#%%Numerical EDA

#loaded congress voting record data set 435 rows, 17 columns
df.head()
df.info()
df.describe()

#There are not 17 predictor variables

#%%Visual EDA


# all the features in this dataset are binary; that is, they are either 0 or 1. 
#So a different type of plot would be more useful here, such as Seaborn's countplot.

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()
#Republicans voted for

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.show()
#Democrats voted for

plt.figure()
sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')
plt.show()
#Democrats voted for

#%%The classification challenge

#k-Nearest Neighbors
#● Basic idea: Predict the label of a data point by
#● Looking at the ‘k’ closest labeled data points
#● Taking a majority vote


#Scikit-learn fit and predict
#● All machine learning models implemented as Python classes
#● They implement the algorithms for learning and predicting
#● Store the information learned from the data
#● Training a model on the data = ‘fi"ing’ a model to the data
#● .fit() method
#● To predict the labels of new data: .predict() method

##Using scikit-learn to fit a classifier

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target'])
#Out[3]: KNeighborsClassifier(algorithm='auto', leaf_size=30,
# ...: metric='minkowski',metric_params=None, n_jobs=1,
# ...: n_neighbors=6, p=2,weights='uniform')
iris['data'].shape
#Out[4]: (150, 4)
iris['target'].shape
#Out[5]: (150,)

##Predicting on unlabeled data

prediction = knn.predict(X_new)
X_new.shape
#(3, 4)
print('Prediction {}’.format(prediction))
#Prediction: [1 1 0]

#%%k-Nearest Neighbors: Fit

#The features need to be in an array where each column is a feature and each 
#row a different observation or data point - in this case, a Congressman's voting record.
#The target needs to be a single column with the same number of observations as the feature data. 

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

#%%k-Nearest Neighbors: Predict

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party',axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)

print("Prediction: {}".format(y_pred))
print("Prediction: {}".format(new_prediction))
print("Accuracy on training set: {}".format(np.mean(y_pred==y)*1))
#%%Measuring model performance

##Measuring model performance
#● In classification, accuracy is a commonly used metric
#● Accuracy = Fraction of correct predictions
#● Which data should be used to compute accuracy?
#● How well will the model perform on new data?
#● Could compute accuracy on data used to fit classifier
#● NOT indicative of ability to generalize
#● Split data into training and test set
#● Fit/train the classifier on the training set
#● Make predictions on test set
#● Compare predictions with the known labels

##Train/test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =
 ...: train_test_split(X, y, test_size=0.3,
 ...: random_state=21, stratify=y)
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
Test set predictions:
 [2 1 2 2 1 0 1 0 0 1 0 2 0 2 2 0 0 0 1 0 2 2 2 0 1 1 1 0 0
 1 2 2 0 0 2 2 1 1 2 1 1 0 2 1]
knn.score(X_test, y_test)
#Out[7]: 0.9555555555555556

##Model complexity
#● Larger k = smoother decision boundary = less complex model
#● Smaller k = more complex model = can lead to overfitting


#%%The digits recognition dataset

# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


#%%Train/Test Split + Fit/Predict/Accuracy

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

#%%Overfitting and underfitting

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
