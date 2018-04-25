# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:33:35 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Support Vector Machines Section 4.1.2

#Use support vector machines to pick the best wine selection

#SVMs are a supervised classification technique that find the maximally 
#separating hyperplane between two classes of data. For two distinct groups of 
#2D data, there are an infinite number of lines which could separate the groups
#What we need is the optimal hyperplane. A hyperplane is just the extension of 
#the line to the number of dimensions in your data (minus 1). How would we 
#determine if a hyperplane is optimal for classifying data in a situation like 
#this? We put the line in the middle of the data groups. How do we find the 
#middle? This is an important point in SVMs: they only use supporting points, 
#or support vectors to determine the middle. is where SVMs draw their name. 
#These points exert the most "force" on the hyperplane, which means that if one 
#were removed or changed, the hyperplane would change
#What we would like to do is place the hyperplane as far away from each of the 
#support vectors as possible. We are trying to maximize the average distance 
#between the margin and the data points chosen as support vectors, while still 
#separating the two classes. 
#Think of the hyperplane and the support vectors as magnets that repel each 
#other. The hyperplane wants to be as far away from each of the closest support 
#vectors, but it is being pushed on both sides by different support vectors.

##What if the data are not linearly separable?

#We use kernels- A kernel is a function that maps data into a higher dimensional 
#space where the data are linearly separable

#example- all data from -6 5 are class 1, all others are class 2- if we use
#x**2 + x, we can use a straight line to bisect the new parabola of data. 

import numpy as np
import scipy as sp
from scipy import stats
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)
#number of observations
print(len(whitedata))
#summary statistics
names = whitedata.dtype.names   
for name in names:
    count,(minval,maxval),mean,var,skew,kurt = stats.describe(whitedata[name])
    print("%20s: min=%.4f, max=%.4f, mean=%.4f, var=%.4f" %(name,minval,maxval,mean,var))

##Classification
    
#need to segment the data into training and testing sets. This code loads the 
#data and its labels
    
#Load up the labels for the wine ratings  
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)

#Save the feature names 
names = whitedata.dtype.names

X = np.array([whitedata[names[10]]]).transpose()
Y = np.array(classlabel['poor']).transpose()

#An important first step with most data is to randomize it to remove any 
#sampling biases in the order of the data. For example, the wine samples may be 
#ordered from low quality to high quality. This can be accomplished by using a 
#shuffling function. In particular, this function preserves the mapping of the 
#data to its labels:
from sklearn.utils import shuffle
X, Y =shuffle(X, Y)

#We then need to split the data into training and testing sets. We construct the 
#model using the training set and evaluate the model on the testing set. This 
#approach is called 2-fold cross-validation, or the holdout method because it 
#involves holding out a portion of the data to evaluate the model. When the 
#model is constructed, it tunes the parameters of the model to reduce the error 
#over the training set. The testing set is used to evaluate how well the 
#constructed model works, which measures the ability of the model to 
#generalize to data it has not seen before. A model can "overfit" the training 
#data by describing characteristics of the training set that are noise or that 
#only exist in the training set. Overfitting is characterized by low error on 
#the training data, but high error on the testing data. Typically, when working 
#with a limited data set, 90% of the data (after shuffling) is used for 
#constructing the model, leaving 10% for model evaluation. 

#Take a look at the syntax below on how to create a SVM. This one is constructed 
#and trained using the training data "X_train" and labels "y_train". 

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
from sklearn import svm
classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)

#Feature selection

#Models are like honored guests; you should only feed them the good parts

#Two main approaches are filtering and wrapper methods. Filtering methods 
#analyze features using a test statistic and eliminate redundant or 
#non-informative features.  As an example, a filtering method could eliminate 
#features that have little correlation to the class labels. 

#Wrapper methods utilize a classification model as part of feature selection. 
#A model is trained on a set of features and the classification accuracy is 
#used to measure the information value of the feature set. One example is that 
#of training a Neural Network with a set of features and evaluating the 
#accuracy of the model. If the model scores highly on the test set, then the 
#features have high information value. All possible combinations of features 
#are tested to find the best feature set. 

#Filtering methods are faster to compute since each feature only needs to be 
#compared against its class label
#Wrapper methods, on the other hand, evaluate feature sets by constructing 
#models and measuring performance. 

#citric acid
from sklearn import svm
import numpy as np
from sklearn import cross_validation
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)
names = whitedata.dtype.names
#names looks like ('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality')
random_state = np.random.RandomState(0)
#citric_acid
X = np.array([whitedata[names[2]]]).transpose()
Y = np.array(classlabel['poor']).transpose()
#90 10
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)
predictionSpace = classifier.predict(X_test)
print("Classification score citric acid",classifier.score(X_test, y_test))

#alcohol
from sklearn import svm
import numpy as np
from sklearn import cross_validation
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)
names = whitedata.dtype.names
#names looks like ('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality')
random_state = np.random.RandomState(0)
#citric_acid
X = np.array([whitedata[names[10]]]).transpose()
Y = np.array(classlabel['poor']).transpose()
#90 10
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)
predictionSpace = classifier.predict(X_test)
print("Classification score alcohol",classifier.score(X_test, y_test))

#ph, citric acid and fixed acidity
from sklearn import svm
import numpy as np
from sklearn import cross_validation
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)
names = whitedata.dtype.names
#names looks like ('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality')
random_state = np.random.RandomState(0)
#citric_acid
X = np.array([whitedata[names[8]],whitedata[names[2]],whitedata[names[0]]]).transpose()
Y = np.array(classlabel['poor']).transpose()
#90 10
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)
predictionSpace = classifier.predict(X_test)
print("Classification score ph, citric acid and fixed acidity",classifier.score(X_test, y_test))

#alcohol, citric acid and fixed acidity
from sklearn import svm
import numpy as np
from sklearn import cross_validation
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)
names = whitedata.dtype.names
#names looks like ('fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality')
random_state = np.random.RandomState(0)
#citric_acid
X = np.array([whitedata[names[10]],whitedata[names[2]],whitedata[names[0]]]).transpose()
Y = np.array(classlabel['poor']).transpose()
#90 10
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)
classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)
predictionSpace = classifier.predict(X_test)
print("Classification score alcohol, citric acid and fixed acidity",classifier.score(X_test, y_test))
#0.7
#If we use this model for determining a wines quality, we will be right 
#approximately 7 out of 10 times.

##Performance - Receiver Operator Characteristic

#Receiver Operator Characteristic (ROC) provides a visual representation of the 
#tradeoff between the True Positive Rate and the False Positive Rate

#Precision and Recall

#used to quantify the quality of performance of a binary classifier in two 
#dimensions. Precision is a measure of exactness and recall measures completeness.


#For example, let's imagine you are building a classifier that should identify 
#if there's an asteroid approaching
#                       Predicted
#                Present         Not Present    Total
#actual
#   Present         96              2           98

#   Not present     18              67          85

#   Total           114             69          183     

#Precision = 96/(96+18) = .84 - 
#if the model predicts an asteroid, an asteroid is actually there 84% of the time.
#Recall = 96/(96+2) = 0.98
#the model will identify 98% of all asteroids

#Precision and recall can be aggregated into a single summary score called the 
#F measure, which ranges from 0 to 1 with 1 being best. The F measure for our 
#classifier above is 
#F = 2*(precision x recall)/(precision + recall) =
#F = 2*(0.84 x 0.96) / (0.84 + 0.96) = 0.90

#ROC Curve

#Some applications, such as medical tests used for screening, may require a 
#high true positive rate, and false positive rates are not as important. Other 
#applications, such as spam filtering, may accept a low true positive rate to 
#ensure a low false positive rate since email users will not want good email to 
#be discarded. Receiver Operating Characteristic (ROC)curves plot a visual 
#tradeoff between true positive and false positive rates.

#If the classifier were perfect, then the ROC curve would pass through the 
#point (0, 1), indicating a 100% true positive rate with 0% false positives.

#ROC curves are useful for comparing different models by simultaneously plotting 
#their curves on the same chart. Depending on your true positive/false 
#positive requirements, you may choose one model over another.

#One common evaluation metric derived from the ROC is the Area Under the Curve, 
#or AUC. A perfect model will result in an AUC of 1.

#The ideal curve comes as close to the top left as possible, which means that 
#the True Positive Rate is much higher than the False Positive Rate.

#Another way of measuring the performance with the curve is the Area Under the 
#Curve (AUC). The maximum AUC is 1, and randomly classifying data points into 
#one of two classes will yield an AUC of 0.5.

#Before we construct a ROC curve, we must first obtain the correct labels that 
#we can use to measure our performance against. We must obtain some measure of 
#classification strength for each example. For two-class classification 
#problems, a strength of 0 implies 100% certainty that the example is negative, 
#while a strength of 1 implies 100% certainty that the example is positive. A 
#strength of 0.5 implies no certainty either way. 

#Now construct the ROC curve by taking the following steps

#0: Once the strengths have been obtained, rank the examples by strength

#1: Take the first value and assign it, and every strength value greater than it, 
#an assigned classification of '+' or 'positive'. Everything less than this 
#value will be assigned a classification of '-' or 'negative'. In the first 
#iteration, there are no values less than this; they will all be assigned '+'.

#2: Calculate the True Positive (TP), True Negative (TN), False Positive (FP), 
#and False Negative (FN) rates.

#Example Strength    Label   Assigned
#1       .13         -       + <---
#2       .24         +       +
#3       .31         +       +       
#4       .33         -       +
#5       .35         -       +
#6       .36         +       +
#7       .52         -       +
#8       .85         +       +
#9       .98         +       +

#True positive - +/+ -> 5
#True negative - -/- -> 0
#False Positive- -/+ -> 4
#False Negative- +/- -> 0

#3: Calculate the True Positive Rate (TPR) and the False Positive Rate (FPR). 
#The formula for these are:
#True Positive Rate  = TP/(TP+FN) = 5/(5+0) = 1
#False Positive Rate = FP/(FP+TN) = 4/(4+0) = 1

#Plot this value in X Y coordinates, where the X axis is the FPR and the Y axis 
#is TPR.

#5: Repeat. Go back to Step 1, but start at Example 2. Your assigned values 
#should look like this:

#Example Strength    Label   Assigned
#1       .13         -       - 
#2       .24         +       + <---
#3       .31         +       +       
#4       .33         -       +
#5       .35         -       +
#6       .36         +       +
#7       .52         -       +
#8       .85         +       +
#9       .98         +       +

#Scikit-learn ROC Curve

#Scikit-learn provides methods to calculate the TPR, the FPR, and the AUC, and 
#plot the ROC as shown.

from sklearn import svm
import numpy as np
from matplotlib import pylab as pl
from scipy import stats
from sklearn import cross_validation
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import LeaveOneOut
whitedata = np.genfromtxt('4-1-2 wineQuality-white.csv',delimiter=';',names=True)
classlabel = np.genfromtxt('4-1-2 wineQuality-2classLabels-white.csv',delimiter=',',names=True)
names = whitedata.dtype.names
random_state = np.random.RandomState(0)
#citric_acid
X = np.array([whitedata[names[2]]]).transpose()
Y = np.array(classlabel['poor']).transpose()
#90 10
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=random_state)
classifier = svm.SVC( probability=True, random_state=0)
classifier = classifier.fit(X_train,y_train)
predictionSpace = classifier.predict(X_test)
print("Classification score citric acid",classifier.score(X_test, y_test))
probas_ = classifier.predict_proba(X_test)
# get the false positive rate, true positive rate and threshold
fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
# calculate AUC 
roc_auc = auc(fpr,tpr)
print("Area under the ROC curve : %0.4f" % roc_auc)

# plot the ROC
pl.figure()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--') # coin toss line
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic example')
pl.legend(loc="lower right")
pl.show()