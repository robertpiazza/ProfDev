# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:20:58 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Recommender Systems Section 4.3.1

#How do Amazon and Netflix recommender systems work? How do they know that since 
#you like X you probably want to buy Y? These services use techniques known as 
#recommendation engines, or recommender systems. While these systems come in 
#many flavors, the two main categories are called collaborative filtering and 
#content-based filtering.

#%%Collaborative Filtering

#This term was first introduced by David Goldberg in 1992 (while working at 
#Xerox PARC) for a system called Tapestry that could filter electronic documents. 
#In short, collaborative filtering utilizes similarities between people's 
#preferences to recommend future products. This algorithm can be summarized as, 
#“if you and I generally like the same things, and I like something new, you 
#will probably like it.” This is similar to how you would decide if you trusted 
#Aunt Shirley’s recommendation for a good TV show.

#Issues Surrounding Collaborative Filtering
#There are a few issues surrounding collaborative filtering that any system 
#faces:
#Cold start: Without a large corpus of information, it is difficult to make 
#recommendations. We require a certain degree of overlap between users of the 
#system to make recommendations. This is a limitation of this type of 
#recommender system. If the system doesn't have a large collection of 
#recommendations, it is difficult to find overlap between users.
#Scalability: Another concern with this type of system is how the algorithm 
#scales when more users are added. With a large number of users, the required 
#computation grows very quickly. Imagine Netflix's problem of trying to scale 
#this to thousands or millions of users!
#Shilling: You always have to watch out for that bad apple. People will try to 
#skew the system's recommendations to promote their own agendas. Safeguards 
#must be put in place to prevent this.
#Black sheep: Some users have no discernible pattern to their preferences. 
#This may make it difficult to recommend anything at all.

#Content-Based Filtering

#Content-based filtering utilizes elements of the items that are to be 
#recommended as features. If the product was a book, features could include the 
#author, the word count, the year it was published, the genre, etc. Does this 
#remind you of any other techniques that we have covered?

#Data Set:
#In this mission, we are building a recommender system using collaborative 
#filtering for text books. On our long journey, we host courses to keep our 
#crew on their toes and reduce boredom. We have digital copies of all the 
#textbooks and circulation data from a university in the United Kingdom:
#University of Huddersfield -- Circulation and Recommendation Data
#There are 2 files of interest:
#Circulation_Data.xml: the full xml file with the circulation history
#Courses.xml: mapping of course numbers to course names 
#This library provided book circulation data at the course level, rather than 
#the individual level. The data shows how books were checked out in relation to 
#students taking courses at the university. An example entry is below: 

#Mission
#The library asked you to construct a recommender system based on only this 
#data, which limits you to a collaborative filtering approach.
#The general framework for a simple collaborative filter is to find similar 
#users and recommend items that they have not used yet.

#To use the circulation data, we need to parse the XML into a more manageable 
#form. We recommend BeautifulStoneSoup for this process. Beautiful Soup is a 
#parser for XML and HTML that can load files into memory for manipulation.

#The following code loads the circulation file into memory and parses it. It may 
#take a little while, since it is a fairly large file. 

xml = open("circulation_data.xml", "r")
soup = BeautifulStoneSoup(xml)

#To find a collection of a particular tag in the "soup," use the following syntax:
collectionOfItems = soup(‘item’)
#This returns all of the tags (and their children) that are labeled "item" as
# a collection. 

#To get the value for a particular tag use: 
item = collectionOfItems[0]
itemValue = item.text

#Load the XML data into memory and construct dictionaries mapping ISBN to 
#title and mapping course code to course title:

#stores a particular rating for a course
ratings = {}
#stores the mapping of isbn numbers to titles
booktitles = {}
#pairwise course similarities
sims = {}

#parse only the elements we need from the xml
for element in soup('item'):
    title = element.title.text
    isbn = element.isbn.text
    booktitles[isbn]=title
    courses = element.courses
    if courses is not None:
        for course in courses('course'):
            line =  isbn+' '+course['id']+' '+course.text+' \n'
            courseid = course['id']
            if courseid in ratings:
                temp = ratings[str(courseid)]
                temp[str(isbn)]=int(course.text)
            else:
                ratings[str(courseid)]={str(isbn):int(course.text)}
          
#The overlappingbooks function finds the intersection between two courses.
def overlappingbooks(book1, book2):
    count=0
    course1 ={}
    course2 ={}
    for key in book1.keys():
        for key2 in book2.keys():
            if key == key2:
                course1[key]=book1[key]
                course2[key2]=book2[key]
                count = count+1
    return count, course1, course2

#Given the intersection between courses we now need a way to calculate the 
#similarity of two courses in regard to the books that were checked out from 
#the library. To do this we use Pearson Correlation.
#Luckily, there is an easy-to-use Pearson's R function that can be loaded from 
#SciPy:
import scipy as scipy
scipy.stats.pearsonr(a,b)


#%%
import scipy as sp
from scipy.stats import pearsonr
import pickle
#load ratings from file
ratings = pickle.load( open( "ratings.p", "rb" ) )
def overlappingbooks(book1, book2):
    count=0
    course1 ={}
    course2 ={}
    for key in book1.keys():
        for key2 in book2.keys():
            if key == key2:
            course1[key]=book1[key]
            course2[key2]=book2[key]
            count = count+1
    return count, course1, course2
key = 'DC230'
key2 = 'HB250'
counted, course1, course2 = overlappingbooks(ratings[key],ratings[key2])
correlation = pearsonr( sp.array(course1.values()), sp.array(course2.values()) )[0]
print(correlation)

#%% An interesting way to visualize these similarities is to show the data in a scatter plot. What does a scatter plot tell us? Let's take a look at an example:
import pylab as pl
course1 = 'HP115'
course2 = 'HP117'
count, ar, br = overlappingbooks(ratings[course1],ratings[course2])
isbns,titles,a,b = converttoarray(ar,br)
pl.scatter(a,b)
pl.xlabel(course1+": "+coursenames[course1])
pl.ylabel(course2+": "+coursenames[course2])
pl.show()
#The plot above shows the number of times that HP115 and HP117 each checked out several books. The points in this figure are annotated with the book title for reference. These points indicate a positive correlation. Actually, the correlation for these two courses is 1.0. Both courses are about coaching and checkout the overlapping texts identically. We see that both of these courses seem to checkout "Successful Coaching" at a greater frequency than the other books that they share in common.

#%%We now need to compute the pairwise similarity between all courses. 
#This function calculates the best matches for a particular course. A "best
#match" is defined here as a course that has correlation greater than 0 with 
#the course we are investigating.

def findbestmatches(key):
    matches={}
    for key2 in ratings.keys():
        if key==key2:
            continue
        counted, course1, course2 = overlappingbooks(ratings[key],ratings[key2])
        if counted>0:
            c2total = len(ratings[key2])
            c1total = len(ratings[key])
            a,b=converttoarray(course1,course2)
            correlation = scipy.stats.pearsonr(a,b)[0]
            if correlation>0:
                scale1 = float(counted)/float(c1total)
                scale2 = float(counted)/float(c2total)
                matches[key2]=correlation*scale1*scale2
    return matches

six man team from gdms 

#%%Write a function that calculates and stores the best matches for all of the courses and stores the values in a dictionary called sims:

def calculatesims():
    count=0
    for key in ratings.keys():
        count= count+1
        matches = findbestmatches(key)
        sims[key]=matches



def recbook(course):
    if len(sims[course])<1:
        return "Nothing yet"
    simlist = sims[course]
    totals={}
    counts={}
    recs = {}
    for key in simlist.keys():
        for isbn in ratings[key].keys():
            if isbn in ratings[course]:
                continue
            rating = str(ratings[key][isbn])
            factor = sims[course][key]
            department=0
            weighted = (int(rating)*factor)+department   
            #already saw this book in another record
            if isbn in counts:
                count = counts[isbn]
                count=count+1
                counts[isbn]=count
                existing = totals[isbn]
                new = existing+weighted
                totals[isbn]=new
            #havent seen this yet
            else:
                totals[isbn]=weighted
                counts[isbn]=1
    for key in totals.keys():
        recs[key]=totals[key]/counts[key]
    max = 0        
    maxisbn = 0
    if len(recs) >0:
        sorted_x = sorted(recs.iteritems(), key=operator.itemgetter(1))
        sorted_x.reverse()
        print '\t'+booktitles[sorted_x[0][0]]+' '+sorted_x[0][0]+' '+str(sorted_x[0][1])
        print '\t'+booktitles[sorted_x[1][0]]+' '+sorted_x[1][0]+' '+str(sorted_x[1][1])
        print '\t'+booktitles[sorted_x[2][0]]+' '+sorted_x[2][0]+' '+str(sorted_x[2][1])
        print '\t'+booktitles[sorted_x[3][0]]+' '+sorted_x[3][0]+' '+str(sorted_x[3][1])
        print '\t'+booktitles[sorted_x[4][0]]+' '+sorted_x[4][0]+' '+str(sorted_x[4][1])
        
#%%

import pickle
import operator
def recbook(course):
	print course+' '+coursenames[course]
	if len(sims[course])<1:
		return "Nothing yet"
	simlist = sims[course]
	totals={}
	counts={}
	recs = {}
	for key in simlist.keys():
		for isbn in ratings[key].keys():
			if isbn in ratings[course]:
				continue
			rating = str(ratings[key][isbn])
			factor = sims[course][key]
			department=0
			weighted = (int(rating)*factor)+department   
            #already saw this book in another record
			if isbn in counts:
				count = counts[isbn]
				count=count+1
				counts[isbn]=count
				existing = totals[isbn]
				new = existing+weighted
				totals[isbn]=new
            #havent seen this yet
			else:
				totals[isbn]=weighted
				counts[isbn]=1
	for key in totals.keys():
		recs[key]=totals[key]/counts[key]
	max = 0        
	maxisbn = 0
	if len(recs) >0:
		sorted_x = sorted(recs.iteritems(), key=operator.itemgetter(1))
		sorted_x.reverse()
		print '\t'+booktitles[sorted_x[0][0]]+' '+sorted_x[0][0]+' '+str(sorted_x[0][1])
		print '\t'+booktitles[sorted_x[1][0]]+' '+sorted_x[1][0]+' '+str(sorted_x[1][1])
		print '\t'+booktitles[sorted_x[2][0]]+' '+sorted_x[2][0]+' '+str(sorted_x[2][1])
		print '\t'+booktitles[sorted_x[3][0]]+' '+sorted_x[3][0]+' '+str(sorted_x[3][1])
		print '\t'+booktitles[sorted_x[4][0]]+' '+sorted_x[4][0]+' '+str(sorted_x[4][1])
ratings = pickle.load( open( "newRatings.p", "rb" ) )
sims = pickle.load( open( "newSims.p", "rb" ) )
coursenames = pickle.load( open( "coursenames.p", "rb" ) )
booktitles = pickle.load( open( "newBookTitles.p", "rb" ) )
recbook('NE200')
recbook('D654')

#%%Content-Based
#In this section, we take a look at content-based recommendation techniques. 
#Dr. Melon has a data set of seeds for us to work with. The goal is to make a 
#recommendation on which seeds grow best on this water world. Since land to 
#grow plants is very expensive here, we are only be able to plant 10 different 
#plants. We need at least 8 of these plants to grow properly in order to feed 
#all of the inhabitants. We have information that seeds most similar to Type 1 
#grow best on this planet and produce enough fruit to be eaten. Unfortunately, 
#Dr. Melon misplaced all of the labels on the seeds, except for one package of 
#seeds that he knows is Type 1. He has profusely apologized in hopes that you 
#won't leave him on this water world. To make it up to you, he examined the 
#remaining seed types and collected several metrics describing them: 

#Area -A
#Perimeter-P
#Compactness 
#C=4πA/P^2
#Length of kernel 
#Width of kernel 
#Asymmetry coefficient 
#Length of kernel groove 

#The Type 1 seed that Dr. Melon saved has the following properties (in the 
#order listed above):
#16.12, 15, 0.9, 5.709, 3.485, 2.27, 5.443

import numpy as np
seeds = np.genfromtxt('4-3-1 seeds.txt',delimiter='\t')
seed_labels = np.genfromtxt('4-3-1 seed_labels.txt',delimiter='\t')
type1mixed = np.array([16.12, 15, 0.9, 5.709, 3.485, 2.27, 5.443])

import numpy as np  
from sklearn.neighbors import NearestNeighbors  

#load the seed data
seeds = np.genfromtxt('4-3-1 seeds.txt',delimiter='\t')  

#calculate nearest neighbors  
nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(seeds)  
distances, indices = nbrs.kneighbors(seeds)

#%%Recommend top 3 seeds
import numpy as np
from sklearn.neighbors import NearestNeighbors
seeds = np.genfromtxt('4-3-1 seeds.txt',delimiter='\t')
seed_labels = np.genfromtxt('4-3-1 seed_labels.txt',delimiter='\t')
type1mixed = np.array([[16.12, 15, 0.9, 5.709, 3.485, 2.27, 5.443]])
nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute').fit(seeds)
distances, indices = nbrs.kneighbors(type1mixed)
print(indices)

#%% Expand to 10 nearest seeds
#Use the nearest neighbor's code to return the top 10 indices and distances for 
#the labeled instance. What are the indices of the top 10 most similar to the 
#type 1 example? 

import numpy as np
import scipy as sp
from sklearn.neighbors import NearestNeighbors
import numpy as np
seeds = np.genfromtxt('4-3-1 seeds.txt',delimiter='\t')
seed_labels = np.genfromtxt('4-3-1 seed_labels.txt',delimiter='\t')
type1mixed = np.array([[16.12, 15, 0.9, 5.709, 3.485, 2.27, 5.443]])
nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute').fit(seeds)
distances, indices = nbrs.kneighbors(type1mixed)
#print(indices)
print(distances)


