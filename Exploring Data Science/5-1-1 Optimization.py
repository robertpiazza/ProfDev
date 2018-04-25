# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:21:35 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Advise-Optimization Section 5.1.1

#Objective: TTo learn about an optimization technique called Genetic Algorithms 
#and to apply that knowledge in finding efficient routes

#%% Optimal Path

#Traveling saleman problem

#Instead of just traveling to the closest city, let's find the optimal route of
#a five node problem

import itertools
import numpy
#create all permutations of the nodes
c = itertools.permutations([1,2,3,4,5])
cl = list(c)
#Adjacency matrix of problem. **Modify this**
#m = numpy.matrix([[0,20,10,30,20],[20,0,30,12,35],[10,30,0,20,15],[30,12,20,0,10],[20,35,15,10,0]])
m = numpy.matrix([[0,500,10,30,20],[500,0,30,12,35],[10,30,0,20,15],[30,12,20,0,10],[20,35,15,10,0]])
#loop through each potential route and calculate its total path distance
for i in range(0,len(cl)):
    l = cl[i]
    print(l)
    next = 0
    sum = 0
#for each leg of the route add the distance to the total
    for j in range(0,len(l)):
        if j==len(l)-1:
            next = l[0]
        else:
            next = l[j+1]
        distance = m.item(l[j]-1,next-1)
        sum = sum+distance
    print("\tPath: "+str(sum))

#This is exhaustive and will only work on a small scale.
    
#%%Empirical Testing
#Since the computers will start to get overwhelmed if we start adding dozens of
#cities, we need better heuristics
    

import itertools
import numpy as np
#number of planets
n = 7 ### Add the number of planets here!
#set seed
np.random.seed(0)
#generate the permutations
c = itertools.permutations(range(1,n+1))
cl = list(c)
print(len(cl))### determine length of "cl" (path list) here
#create a random matrix
b = np.random.random_integers(10,200,size=(n,n))
#make it symmetric
m = (b + b.T)/2
#set the diagonal to 0
for k in range(0,len(m)):
    m[k][k]=0
#dictionary to store all the paths
paths={}
##calculate the path
for i in range(0,len(cl)):
    l = cl[i]
#    print(l)
    next = 0
    sum = 0
    for j in range(0,len(l)):
        if j==len(l)-1:
            next = l[0]
        else:
            next = l[j+1]
        distance = m.item(l[j]-1,next-1)
        sum = sum+distance
#    print("\tPath: "+str(sum))
    paths[str(l)]=sum
#find the minimum path
minSum =100000
for key in paths.keys():
#    print(key+" "+str(paths[key]))
### Add code here to find the minimum path length!
    if paths[key]<minSum:
        minSum = paths[key]
        minKey = key
print("Minimum is"+str(minKey)+"-"+str(minSum))

#%% Genetic Algorithms

#Like biological evolution, they adapt a solution to the problem by making 
#small incremental changes. They allow for efficient exploration of the search 
#space


#TL:DR
#The basic format of a genetic algorithm is to encode a population of solutions and loop over the population performing operators over the solutions. The individuals in the population are evaluated using a fitness function, which means that the solutions in the population are evaluated and scored and assigned a 'fitness.' Individuals that are more fit are more likely to reproduce and make it into the next generation.

#The process of scoring and ranking is called selection. Once solutions are selected, they are crossed using the crossover operator. Crossover is inspired by sexual reproduction in nature. Two solutions swap parts of their solutions and create a new solution. There is also randomness in nature that randomly modifies solutions

#Using a selection method called tournament selection, we sample from our population with replacement with a fixed sample size. 

#For instance, in the examples from the last challenge, we can sample 2 with replacement. The fitness of each of the vectors is compared and the 'more fit' is retained. Shown below, in the sample of 2, we would keep vector B because its score of 7 is more fit than vector A, which has a score of 4.

#The next operator that we will develop is the Crossover operator. This operator combines solutions into new solutions. As previously mentioned, this operator is inspired by sexual reproduction in nature. Solutions that are more 'fit' are more likely to reproduce. This drives the population towards combinations that are more fit, while also trying out combinations of two parents.

#This crossover combined with selection moves the vectors in the population towards more fit solutions. Vectors that have higher scores are more likely to be selected for crossover. When more fit vectors combine together, each could potentially have a beneficial characteristic. If these two characteristics combine together then their 'child' can potentially have a higher fitness than either of the parents.

#If the population is fixed, we will be limited in the ways that features can be recombined. For instance, if no vector has an element of '0' at position 1 in the population, then it will not be possible through crossover for that value to ever occur at position 1. To help introduce new features and combinations, the mutation operator is used. This operator randomly changes a vector. Typically, this is set to a low number (<5%). This operator will allow for features to appear in the population that were not in the original population.

#%% Full Genetic Algorithm

import numpy as np
import random as r
from ga import initializePopulation
from ga import score
from ga import tournamentSelectPop
from ga import crossover
from ga import ox
from ga import mutate
from ga import swap
from ga import scorePopulation
from ga import bestScore
pop = initializePopulation(500)
popscore =[]
for i in range(0,len(pop)):
    print(i)
    popscore[i] = score(pop[i])
    
#%%Final Scores after 500 Generations

import numpy as np
import random as r
from matplotlib import pyplot as plt #pyplot is the primary plotting tool in matplotlib
np.random.seed(100)
r.seed(100)
#load distances
dist = np.genfromtxt("5-1-1 intercity.csv", delimiter=',')
#creates population of size n
def initializePopulation(n):
    pop=[]
    for i in range(0,n):
	#create list of planets
        ind = np.arange(48)
	#randomly order them
        np.random.shuffle(ind)
	#add to population
        pop.append(ind)
    print((pop))
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
    for i in range(0,100):### Add code here to create generations
        popscore = scorePopulation(pop)
        newpop = tournamentSelectPop(2,pop,popscore)
        crossedpop = crossover(newpop)
        mutatedpop = mutate(0.001,crossedpop)
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
