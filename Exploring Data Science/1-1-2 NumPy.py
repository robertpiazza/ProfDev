# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:38:20 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

# Data Science Examples from ExploringDataScience.com

###Foundations-NumPy Section 1.1.2

#SciPy, Pandas, and matplotlib, depend on NumPy

##Loading Data

#Using the NumPy genfromtxt method, we can load in data from numerous formats
import numpy as np
data = np.genfromtxt('1-1-2 sample2.csv', delimiter=',', dtype='f8,i8,S15,S15',names=True)
print(data)
#4 lines of data with ',' as the delimiter
#Types of data
print(data.dtype)
#Column names-This can be useful for iterating through columns or to not have 
#to hard code the column names in a script
print(data.dtype.names)
#We can access the data in the object in different ways. To select an entire 
#column of the CSV, use the column name
print(data['Field2'])
#To select an entire row of the CSV, use the row number
print(data[0])

##Accessing Arrays and Computation of Statistics

#Access the 'Field0' array in the data set
print(data['Field0'])
#NumPy's mean() method to calculate the mean of the 'Field0' array
print(data['Field0'].mean())
#Use NumPy's max() method to calculate the maximum of the 'Field0' array
print(data['Field0'].max())
#Use NumPy's argmax() method to find the array location of the maximum element of the 'Field0' array
print(data['Field0'].argmax())
#Check the element location to be sure it is equal to the maximum of the 'Field0' array.
print(data['Field0'][0])
#Use NumPy's std() method to calculate the standard deviation of the 'Field0' array
print(data['Field0'].std())
#Note: You can also calculate the variance this way with the var() method

#Use NumPy's min() method to calculate the min of the zeroth row
#print(data[0].min())-commented out to prevent error
#Note: The row contained strings causing a TypeError.


##Loading Data Differently

#If you know you do not have variable type data or missing data, 
#then a faster and simpler method of loading data is loadtxt

#There are times when we would like to ignore the strings in our CSV and 
#just work with the numbers. In order to skip the header, skiprows, and only 
#load the columns with numbers, usecols, enter this line into the console
data2 = np.loadtxt('1-1-2 sample2.csv', delimiter=',',usecols=(0,1),skiprows=1)
print(data2)

#Accessing the rows is the same, but accessing columns is not. 
#We no longer have fields to use to identify them. 
#The first option is to transpose the matrix
data2.transpose()
#Another option is to unpack the data as it comes in. 
#Load data into x and y, then print x
x,y = np.loadtxt('1-1-2 sample2.csv', delimiter=',',usecols=(0,1),skiprows=1,unpack=True)
print(x)

##Give Me An Array, Then Double It

#Use the range function and NumPy's array method to create an array of 
#the numbers 0 through 14
x = np.array([range(15)])
print(x)
print(x*2)

#Other NumPy references here:
#https://engineering.ucsb.edu/~shell/che210d/numpy.pdf



