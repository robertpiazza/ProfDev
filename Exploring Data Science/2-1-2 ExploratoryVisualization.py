# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 09:44:23 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton

Need help installing mplfacet library
"""

## Data Science Examples from ExploringDataScience.com

#Describe-Exploratory Visualizations Section 2.1.2

#MatplotLib reference: https://scipy-lectures.github.io/intro/matplotlib/matplotlib.html

#Line Plots
#https://explore-data-science.thisismetis.com/Handbook/matplotlib
#Import Python libraries. 
#Load data.
#Generate the plot. 
#View the plot. 
#Import Python libraries.

#Population of Traverse city since 1890
import numpy as np #matplotlib is dependent on numpy
from matplotlib import pyplot as plt #pyplot is the primary plotting tool in matplotlib

#Load Data 
population = [4353, 9407, 12115, 10925, 12539, 14455, 16974, 18432, 18048, 15516, 15155, 14532, 14674] #Data entered as a list

#Generate the plot.
plt.plot(population) #Creates the plot of your data. Couldn’t be simpler!

#View the plot.
#plt.show() #Opens a new window and displays your plot.

#Title and Axis Labels
plt.title('Population of Traverse City, MI by Decade')
plt.xlabel('Year')
plt.ylabel('Population')
years = list(range(1890,2020,10))
plt.plot(years, population)

#Changing the Window
#axis([min_x, max_x, min_y, max_y])
plt.axis([1890, 2010, 0, 20000])
#plt.yticks([2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
plt.yticks(list(range(2500,22500, 2500)))
#see handbook for line color and symbol type
plt.show()

#Multi-variable Plots
#plot the population of Traverse City vs. Grand Traverse County
years = list(range(1890,2020,10))
TCPop= [4353, 9407, 12115, 10925, 12539, 14455, 16974, 18432, 18048, 15516, 15155, 14532, 14674]
GTCPop = [13355, 20479, 23784, 19518, 20011, 23390, 28598, 33490, 39175, 54899, 64273, 77654, 86986]

plt.plot(years, TCPop, 'b-', label = 'Traverse City')
plt.plot(years, GTCPop, 'r-', label = 'Grand Traverse County')

plt.title('Traverse City Population vs. Grand Traverse County Population')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(loc = 'upper left')
plt.show() 

#Histograms
#Histogram of Standardized Test Scores
from numpy import loadtxt
from numpy import transpose
from matplotlib import pyplot as plt
data = loadtxt("GPAData.txt")
temp = transpose(data)
examsc = temp[2,]
plt.hist(examsc) #defaults to 10 bins
#plt.xticks(np.arange(min(var_name),max(var_name)+1,(max(var_name)-min(var_name))/(number_of_bins-10 by default))
plt.xticks(np.arange(min(examsc),max(examsc)+1,(max(examsc)-min(examsc))/10))
#plt.hist(var_name,[list_of_endpoints],color = ‘your_color’)- match xticks and endpoints for bin delineation
plt.show

#Create a histogram of exam scores suitable for a report or presentation
plt.hist(examsc,np.arange(0, 110, 10),color = '#5D5166')
plt.xticks(np.arange(0, 110, 10))
plt.show()

#Scatterplots
data0 = np.loadtxt("2-1-2 GPAData.txt")
data = transpose(data0)
gpa = data[1,]
exam = data[2,]
year = data[4,]
gpa2024 = gpa[0:200]
gpa2026 = gpa[400:600]
exam2024 = exam[0:200] ### exam grades of class of 2024
exam2026 = exam[400:600] ### exam grades class of 2026
plt.scatter(gpa2024,exam2024, s = 40, color = '#5D5166', marker = 'D', label = "Class of 2024")
plt.scatter(gpa2026,exam2026, s = 40, color = '#FF971C', marker = 'o', label = "Class of 2026")
plt.title('Class of 2024 vs. Class of 2026')
plt.xlabel('GPA')
plt.ylabel('Exam Score')
plt.axis([0, 4.1, 0, 101])
plt.legend(loc = 'lower left')
plt.show()

#Faceted plot for showing multiple plots by graphs- mplfacet library
#Facet(standcat, [gpa], xlabel='GPA', ylabel='Number of Students').hist(np.arange(0,4.5,0.5), color = '#5D5166')

stand = data[3,] #standardized test scores
var = data[2,] ###Free hint: Exam scores are stored in the 3rd column. Python starts counting at 0. Set "var" equal to the exam column.
standcat = [0 for i in range(len(var))] #standardized test score category (19 & below, 20 to 29, 30 & up)
#This loop determines which standardized test category each student is in. Leave it alone.
for i in range(len(var)):
	if stand[i]<=19:
		standcat[i]='19 and below'
	else:
		if stand[i]<=29:
			standcat[i]='20 to 29'
		else:
			standcat[i]='30 and up'
from mplfacet import Facet
#Create a faceted histogram for exam scores over the standardized test score categories
#Bin width to 10, and bin range from 0 to 100
Facet(standcat, [var], xlabel="Exam Score", ylabel="Number of Students").hist(bins = np.arange(0,110,10), color = "#5D5166") ### This is your plot command. Change as necessary.
plt.show()
