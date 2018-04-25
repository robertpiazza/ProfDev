# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 13:25:05 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""


## Data Science Examples from ExploringDataScience.com

#Foundations-Standard CSV Library Section 1.1.1

#Reading a CSV File
import csv 
infile = open('1-1-1 sample.data','r') #the r means open for reading, use w when writing
reader = csv.reader(infile)
for line in reader:
    print(line)
infile.close()
#Quote use- it's a style question:
#'He turned to me and said, "Hello there"'

#Writing Data to a File
#Import the library 
#Open the file for writing ('w') 
#Construct a writer 
#Write the rows using the writer 
#Close the file 
#Note: The reader is hard-coded to recognise either \r\n or \n as 
#end-of-line and ignores a line terminator if given.
outfile = open('1-1-1 newfile.csv', 'w')
out = csv.writer(outfile, lineterminator='\n')
#Note: The string used to terminate lines produced by the writer defaults 
#to \r\n. This may cause issues for non-Windows users if you do not know this 
#is the default.
out.writerow(['this','is','your','header'])
for i in range(10):
    out.writerow([i,i+1,i+2,i+3])
outfile.close()
