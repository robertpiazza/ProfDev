# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:58:38 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

#%%Final Scores after Multiple Generations

import numpy as np
import random as r
from matplotlib import pyplot as plt #pyplot is the primary plotting tool in matplotlib
np.random.seed(100)
r.seed(100)
#load distances
dist = np.genfromtxt("5-1-1 intercity.csv", delimiter=',')
rankDict={} #create dictionary where first key is the city, second key is the rank (1-length) (smallest to largest) of distances from original key city to other cities and value is the city with that  rank
for i in range(0, len(dist)): #create a dictionary of dictionaries with the distance matrix
    current=dist[i]
    indDict={}
    for j in range(0,len(current)):
        indDict.update({j:dist[i][j]})
        #rank the cities by distance and key to that rank
        RankIndDict = {rank: key for rank, key in enumerate(sorted(indDict, key=indDict.get, reverse = False), 1)}
    rankDict.update({i:RankIndDict})
    
#creates population of size n
def initializePopulation(n):
    print(len(dist))
    print(len(rankDict))
    pop=[] #create empty population array
    for individual in range(0,n):
        ind = np.arange(48)
        #print('individual '+str(individual))
        ind[0]=r.randint(0,len(rankDict)-1) #first stop on the route will iteratively be the next planet
        set1 = set() #create a set to ensure we don't visit the same planet twice for each route
        set1.add(ind[0]) #add the inital planet
        for nextnumber in range(1,len(dist)): #go through each stop on the route
            for rank in range(2,len(dist)+1): #for the last planet, go through all the nearest planets and pick the closest one that hasn't been visited yet
                if rankDict[ind[nextnumber-1]][rank] in set1:
                    continue
                else:
                    set1.add(rankDict[ind[nextnumber-1]][rank])
                    ind[nextnumber]=rankDict[ind[nextnumber-1]][rank]
                    break
        pop.append(ind)
    return pop
#score route, return the total
# distance of all the legs
def score(v):
    sum = 0
    for j in range(0, len(v)):
        current=v[j]
        if j==len(v)-1:
            next = v[0]
        else:
            next = v[j+1]
        distance = dist[current-1][next-1]
         #print str(current)+' '+str(next)+' '+str(distance)
        sum = sum + distance
    return sum
#evaluates each individual in a population
def scorePopulation(pop):
    popScore = []
    for p in range(0,len(pop)):
        s = score(pop[p])
        popScore.append(s)
    return popScore
#returns a new population, created by selecting best individual from samples ofspecified size
def tournamentSelectPop(size,pop,popScore):
    newpop = []
    while len(newpop)<len(pop):
        sample = None
        sampleScore = 1000000000
        #select sample
        for i in range(0,size):
            s1 = r.randint(0,len(pop)-1)
            sscore = popScore[s1]
            #print 'score' + str(sscore)
            #minimize the overall score
            if sscore<sampleScore:
                sample = pop[s1]
                sampleScore = sscore
                newpop.append(sample)
    return newpop
#performs ordered crossover operation on population, randomly orders population and crosses pairs
def crossover(pop):
    newpop = []
#shuffle the data
    index_shuf = range(len(pop))
    for i in range(0,len(index_shuf)-1,2):
        ind1 = pop[index_shuf[i]]
        ind2 = pop[index_shuf[i+1]]
        startIndex = r.randint(0,len(ind1)-1)
        endIndex = r.randint(startIndex,len(ind1)-1)
#construct new individuals from crossing parents using ordered crossover
        child1 = ox(startIndex, endIndex, ind1,ind2)
        child2 = ox(startIndex, endIndex, ind2,ind1)
        newpop.append(child1)
        newpop.append(child2)
    return newpop
#Ordered crossover, returns a single individual by crossing ind1 and ind2, ind elements are used between start and end, ind1 elements are used everywhere else
def ox(start, end, ind1, ind2):
    set1 = set()
    set2 = set()
#copy the individuals
    ind1Copy = list(ind1)
    ind1Copy.reverse()
    ind2Copy = list(ind2)
    ind2Copy.reverse()
    newInd = list(np.zeros(len(ind1)))
#elements that are in the crossover region
    for i in range(start,end+1):
        set1.add(ind1[i])
        set2.add(ind2[i])
    for i in range(0,len(ind1)):
        val = ind1[i] #? saved for printing what the value is we're replacing?
        #in crossover region
        if((i>=start) & (i<=end)):
            #print 'continuing '+str(i)
            newInd[i]=ind2[i]
            continue;
            #get next that isnt already in set
        val2 = ind1Copy.pop() #remove last value from ind1Copy and place in val2
        #print 'trying: '+str(val2)
        alreadyIn = val2 in set2;
        while alreadyIn:
            #print 'skipping '+str(val2)
            val2 = ind1Copy.pop()
            #print val2
            alreadyIn = val2 in set2;
        newInd[i] = val2
    return newInd
#mutate a population
def mutate(rate,pop):
	newpop=[]
	for p in pop:
		if r.random()<rate:
			index1 = r.randint(0,len(p)-1)
			index2 = r.randint(0,len(p)-1)
			newp = swap(index1,index2,p)
			newpop.append(newp)
		else:
			newpop.append(p)
	return newpop
#swaps two elements in an individual at a given index
def swap(index1,index2,p):
	i1value = p[index1]
	i2value = p[index2]
	p[index1]=i2value
	p[index2]=i1value
	return p
#returns average score of a population
def avescore(popscore):
	sum=0
	for i in range(0, len(popscore)):
		sum = sum+float(popscore[i])
		n = len(popscore)
		ave = float(sum)/float(n)
	return ave
#returns best score in a population
def bestScore(popscore):
	min = popscore[0]
#for all the individual scores
	for p in popscore:
		if p<min:
			min = p
	return min

totalPop = []
for group in range(10):
    print('gen '+str(group))
    pop = initializePopulation(50)
    currentMin = 100000000
    ScoreData = []
    for i in range(0,10000):### Add code here to create generations
        popscore = scorePopulation(pop)
        newpop = tournamentSelectPop(2,pop,popscore)
        crossedpop = crossover(newpop)
        mutatedpop = mutate(0.01,crossedpop)
        newscore = scorePopulation(mutatedpop)
        pop = mutatedpop
        ave = avescore(newscore)
        if i % 100 == 0 or i==1:
            print(i)
            print(currentMin)
        bestscore = bestScore(newscore)
        ScoreData.append(bestscore)
        if bestscore<currentMin:
            currentMin = bestscore
    if group == 0:
        totalPop = pop
    else:
        print('population is '+str(group))
        totalPop = np.concatenate((totalPop,pop))        
    print('best '+str(currentMin))
    plt.plot(ScoreData)
pop = totalPop
for i in range(0,50):### Add code here to create generations
    popscore = scorePopulation(pop)
    newpop = tournamentSelectPop(2,pop,popscore)
    crossedpop = crossover(newpop)
    mutatedpop = mutate(0.001,crossedpop)
    newscore = scorePopulation(mutatedpop)
    pop = mutatedpop
    ave = avescore(newscore)
    if i % 3 == 0 or i==1:
        print(i)
        print(currentMin)
    bestscore = bestScore(newscore)
    ScoreData.append(bestscore)
    if bestscore<currentMin:
        currentMin = bestscore
plt.title('ScoreData')
plt.xlabel('Generation')
plt.ylabel('Distance')
plt.plot(ScoreData)
#Load Data 
 #Data entered as a list

#Generate the plot.


#View the plot.
#plt.show() #Opens a new window and displays your plot.

#Title and Axis Labels