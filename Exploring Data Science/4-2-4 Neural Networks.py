# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:40:15 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Neural Networks Section 4.2.4

#The penalties for having such flexibility are:
#the possibility of overfitting the data 
#slow convergence 
#the number of hidden units and layers must still be selected by hand, which leads to the bias-variance dilemma.

#%%Task 1
#Create a network with 1 input unit, 10 hidden units, and 1 output unit that 
#uses the tanh transfer function at the hidden layer and a linear output layer. 

# IMPORTS
import numpy as np
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import LinearLayer
# LOAD DATA
datain=np.loadtxt(open("4-2-4 vespene.csv","rb"), delimiter=",", skiprows=0)
# NUMBER OF DATA POINTS
m = datain.shape[0]
# MEAN CENTER THE DATA
datain[:,0]=datain[:,0]-np.mean(datain[:,0])
datain[:,1]=datain[:,1]-np.mean(datain[:,1])
# DATA CLASS AND ENTRY
# data = ClassificationDataSet(1,1) <-- For classification
data = SupervisedDataSet(1,1) # For regression
for i in range(m):
    data.addSample(datain[i,0], datain[i,1])

#Build a network with 1 input neurons, 10 hidden neurons, and 1 output neuron 
#using PyBrain. Make sure you are using TanhLayer for hidden layers and 
#LinearLayer for output layers. 
net = buildNetwork(1,10,1, hiddenclass=TanhLayer, outclass=LinearLayer)

#Create a BackpropTrainer with a learning rate of .00005, and verbose = True
trainer = BackpropTrainer(net, dataset=data, learningrate=0.00005, verbose = True)
#Train the network for 1000 iterations .
Nsteps = 1000
trainer.trainEpochs(Nsteps)      

#The code is almost identical to that used for performing classification. We 
#used a different data class; instead of using ClassificationDataSet, we used 
#SupervisedDataSet. We also used a tanh for the hidden layer transfer function, 
#but a logistic function would work as well. 

##More importantly, we used a linear transfer function for the output layer. 
#This is critical, as any sigmoid transfer function saturates between a minimum 
#and a maximum value and is not able to cover any arbitrary range of the output 
#values.

#%% Overfitting

#Task 2:
#Build four different networks. All four networks should have 1 input and 1 
#output neuron each, but should have the following structure for the hidden 
#layer(s): 

#One network has 1 hidden layer with 10 neurons; call this net0. 
#One network has 2 hidden layers with 5 neurons each; call this net1. 
#One network has 1 hidden layer with 200 neurons; call this net2. 
#One network has 2 hidden layers with 100 neurons each; call this net3. 
#Set hiddenclass to be tanh and outclass to be linear.
# IMPORTS
import numpy as np
import pylab as pl
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import LinearLayer
def plot_networks(nets,x,y):
    pl.figure()
    pl.plot(x,y,'ok')
    xmin = min(x)
    xmax = max(x)
    x2 = np.linspace(xmin,xmax,200)
    data = SupervisedDataSet(1,1)
    for i in range(len(x2)):
         data.addSample([x2[i]], [0])
    colors = ['#5D5166','#FF971C','#447BB2','#B24449']
    for i in range(len(nets)):
        y2 = nets[i].activateOnDataset(data)
        pl.plot(x2,y2[:,0],color=colors[i],linewidth=1.8)
    pl.legend(('data','10','5,5','200','100,100'), loc='upper left')
    pl.show()
# LOAD DATA
datain=np.loadtxt(open("vespene.csv","rb"), delimiter=",", skiprows=0)
# NUMBER OF DATA POINTS
m = datain.shape[0]
# MEAN CENTER THE DATA
datain[:,0]=datain[:,0]-np.mean(datain[:,0])
datain[:,1]=datain[:,1]-np.mean(datain[:,1])
# DATA CLASS AND ENTRY
# data = ClassificationDataSet(1,1) <-- For classification
data = SupervisedDataSet(1,1) # For regression
for i in range(m):
	data.addSample(datain[i,0], datain[i,1])
### Building networks!
net0 = buildNetwork(1,10,1, hiddenclass=TanhLayer, outclass=LinearLayer)
net1 = buildNetwork(1,5,5,1, hiddenclass=TanhLayer, outclass=LinearLayer)
net2 = buildNetwork(1,200,1, hiddenclass=TanhLayer, outclass=LinearLayer)
net3 = buildNetwork(1,100,100,1, hiddenclass=TanhLayer, outclass=LinearLayer)


nets = [net0,net1,net2,net3]
for net in nets:
    trainer = BackpropTrainer(net, dataset=data, learningrate=0.005)
    trainer.trainEpochs(200)
 
plot_networks(nets,datain[:,0],datain[:,1])

#%% Regularization (weight decay)

#Task

#Build three different networks and trainers. All three networks should be 
#identical. Each network should have: 1 input and 1 output neuron and 2 hidden 
#layers of 100 neurons in each layer. All networks should use tanh for the 
#hidden transfer function and linear for the output transfer function. Name the 
#networks: net0, net1, and net2. 

#Create three different trainers, trainer0, trainer1 etc., that have different 
#values for weightdecay.
#Train net0 with weightdecay=0. 
#Train net1 with weightdecay=0.05. 
#Train net2 with weightdecay=0.5. 
#Train each network for 200 epochs each using learningrate=0.005. Run the code 
#and look at the graph.

import numpy as np
import pylab as pl
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import TanhLayer
from pybrain.structure.modules import LinearLayer
# FOR PLOTTING
def plot_networks(nets,x,y):
    pl.figure()
    pl.plot(x,y,'ok')
    xmin = min(x)
    xmax = max(x)
    x2 = np.linspace(xmin,xmax,200)
    data = SupervisedDataSet(1,1)
    for i in range(len(x2)):
        data.addSample([x2[i]], [0])
    colors = ['#5D5166','#FF971C','#447BB2','#B24449']
    for i in range(len(nets)):
        y2 = nets[i].activateOnDataset(data)
        pl.plot(x2,y2[:,0],color=colors[i],linewidth=1.8)
    pl.legend(('data','decay=0.0','decay=0.05','decay=0.5'), loc='upper left')
    pl.show()
# LOAD DATA
datain=np.loadtxt(open("vespene.csv","rb"), delimiter=",", skiprows=0)
# NUMBER OF DATA POINTS
m = datain.shape[0]
# MEAN CENTER THE DATA
datain[:,0]=datain[:,0]-np.mean(datain[:,0])
datain[:,1]=datain[:,1]-np.mean(datain[:,1])
# DATA CLASS AND ENTRY
# data = ClassificationDataSet(1,1) <-- For classification
data = SupervisedDataSet(1,1) # For regression
for i in range(m):
    data.addSample(datain[i,0], datain[i,1])

net0 = buildNetwork(1,100,100,1, hiddenclass=TanhLayer, outclass=LinearLayer)
net1 = buildNetwork(1,100,100,1, hiddenclass=TanhLayer, outclass=LinearLayer)
net2 = buildNetwork(1,100,100,1, hiddenclass=TanhLayer, outclass=LinearLayer)

trainer0 = BackpropTrainer(net0, dataset=data, learningrate=0.005, weightdecay=0)
trainer1 = BackpropTrainer(net1, dataset=data, learningrate=0.005, weightdecay=0.05)
trainer2 = BackpropTrainer(net2, dataset=data, learningrate=0.005, weightdecay=0.5)

trainer0.trainEpochs(200)
trainer1.trainEpochs(200)
trainer2.trainEpochs(200)
### Add your code here!
# Don't forget to call your networks: net0, net1, etc.
# DON'T TOUCH CODE BELOW THIS
nets = [net0,net1,net2]
plot_networks(nets,datain[:,0],datain[:,1])

#When the decay is 0, the network is free to try to fit the points as best as 
#it possibly can. As the decay increases, the network must balance minimizing 
#the squared values of the weights against reducing the error in the fit; 
#therefore, the curve shows less variance and begins to smooth out. Finally, if 
#we set the regularization too high, as it is when weightdecay=.5, the network 
#is overly concerned about minimizing the weights, to the point where it doesn't 
#care about the error at all

#%% Cross-validation

#How are we then, to select the best value for regularization and the number of 
#hidden layers and neurons? Unfortunately, there is no theoretical answer that 
#points you to a "sweet spot." The choice of parameters is a function of how 
#much error in the fit you are willing to tolerate. If you are looking to fit 
#every single point with near perfection, then you should choose no regularization 
#and many hidden neurons and layers. If, however, you are using regression for 
#prediction, fitting every data point is not desirable as you will be 
#overfitting the data and will likely not generalize well to out-of-sample 
#(i.e., new) data.

#This is where cross-validation enters. You may recall splitting up data into a 
#training set and a test set in previous missions. The training set was used for 
#learning, while the test set was used to determine the accuracy of the trained 
#network on out-of-sample data points. We are now adding a third split in the 
#data: a validation set. The validation set, like the test set, is used to 
#determine the accuracy of the trained model on out-of-sample data points. 
#Unlike the test set, it is not used to report the model's performance. It is 
#used instead to select the model with the best parameters.

#This is how it works. Choose which parameters to vary, say the regularization 
#parameter and the number of hidden units in a single layer. Train a different 
#network for each parameter pair using the same training set. Test each network 
#on the validation set. Choose the network with the best performance on 
#validation. Finally, take the best network and run it on the test set to 
#report its accuracy.

#The choice of the split in the data is somewhat arbitrary. A good starting 
#point is to use 60 percent for training, 20 percent for validation, and 20 
#percent for testing. One important note: when splitting your data, you must 
#ensure that it is really being randomized. For example, if your data is sorted, 
#you cannot simply take the first 60 percent and assign it to the training set 
#as that would exclude the range of the predictor variables in the last 40 
#percent. Even if your data is not sorted by predictor values, it may be sorted 
#by collection time, and taking the first 60 percent means training only on the 
#earliest collected data which can lead to biases.

#PyBrain can split the data for you:
#trn,valtst = data.splitWithProportion(proportion=0.6)
#val,tst = valtst.splitWithProportion(proportion=0.5)

#There is one other pitfall that you must worry about. When you change network 
#parameters, like the number of hidden neurons, convergence rates of learning 
#may differ. Earlier, we trained networks for 1000 epochs regardless of how 
#many neurons we had in the network; we then compared the final networks at the 
#end of the 1000 epochs. In practice, this is not a great idea, since some of 
#$those networks may have converged before training terminates, while others 
#may not have done so. A better approach is to train until the learning slows 
#below a threshold, i.e. the cost function decreases by less than some set 
#percentage for some consecutive number of training epochs. This is easier to 
#implement using batch gradient descent, where the cost function always 
#decreases, rather than stochastic gradient descent, where the cost function 
#can sometimes increase.

