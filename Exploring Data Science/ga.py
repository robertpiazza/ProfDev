import numpy as np
import random as r
from sets import Set
#load matrix of data
dist = np.genfromtxt("interCity.csv", delimiter=',')
np.random.seed(100)
r.seed(100)
def initializePopulation(n):
# n is number of individuals in population
pop=[]
for i in range(0,n):
ind = np.arange(48)
np.random.shuffle(ind)
pop.append(ind)
return pop
def score(route):
sum = 0
#for each of the planets in the route
for j in range(0, len(route)):
#get the current planet
current=route[j]
#if last element connect back to first planet
if j==len(route)-1:
next = route[0]
#get the next planet
else:
next = route[j+1]
#get the distance to the next planet
distance = dist[current-1][next-1]
#add to the total route distance
sum = sum + distance
return sum
def bestScore(popscore):
min = popscore[0]
for p in popscore:
if p<min:
min = p
return min
def scorePopulation(pop):
popScore = []
for p in range(0,len(pop)):
s = score(pop[p])
popScore.append(s)
return popScore
def tournamentSelectPop(size,pop,popScore):
newpop = []
while len(newpop)<len(pop):
sample = None
sampleScore = 1000000000
#select sample
for i in range(0,size):
s1 = r.randint(0,len(pop)-1)
#print len(pop)
#print s1
sscore = popScore[s1]
#print 'score' + str(sscore)
#minimize the overall score
if sscore<sampleScore:
sample = pop[s1]
sampleScore = sscore
newpop.append(sample)
return newpop
def crossover(pop):
newpop = []
#shuffle the data
index_shuf = range(len(pop))
for i in range(0,len(index_shuf),2):
ind1 = pop[index_shuf[i]]
ind2 = pop[index_shuf[i+1]]
startIndex = r.randint(0,len(ind1)-1)
#print str(startIndex)
endIndex = r.randint(startIndex,len(ind1)-1)
#print str(startIndex)+' '+str(endIndex)
#construct new individuals from crossing parents
child1 = ox(startIndex, endIndex, ind1,ind2)
child2 = ox(startIndex, endIndex, ind2,ind1)
newpop.append(child1)
newpop.append(child2)
return newpop
def ox(start, end, ind1, ind2):
set1 = Set()
set2 = Set()
ind1Copy = list(ind1)
ind1Copy.reverse()
ind2Copy = list(ind2)
ind2Copy.reverse()
newInd = list(np.zeros(len(ind1)))
#elements that are in the crossover region
for i in range(start,end+1):
set1.add(ind1[i])
set2.add(ind2[i])
index2 = 0
for i in range(0,len(ind1)):
val = ind1[i]
#in crossover region
if((i>=start) & (i<=end)):
#print 'continuing '+str(i)
newInd[i]=ind2[i]
continue;
#get next that isnt already in set
val2 = ind1Copy.pop()
alreadyIn = val2 in set2;
while alreadyIn:
val2 = ind1Copy.pop()
alreadyIn = val2 in set2;
newInd[i] = val2
return newInd
def mutate(rate,pop):
newpop=[]
for p in pop:
if r.random() < rate:
index1 = r.randint(0,len(p)-1)
index2 = r.randint(0,len(p)-1)
newp = swap(index1,index2,p)
newpop.append(newp)
else:
newpop.append(p)
return newpop
def swap(index1,index2,p):
i1value = p[index1]
i2value = p[index2]
p[index1]=i2value
p[index2]=i1value
return p