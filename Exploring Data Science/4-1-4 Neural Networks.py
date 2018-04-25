# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:40:19 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

##Predict-Neural Networks Section 4.1.4

#%% Use neural networks for classification problems

#We first look at some quick applications of neural networks; then we move on 
#to building our own neural network piece by piece. Finally, we familiarize you 
#with a Python module known as PyBrain

#Neural Networks

#The artificial neuron has inputs, which pass "features" or the outputs from other neurons through a set of weights into the "soma" of the neuron. The soma sums the weighted inputs together and applies a (usually nonlinear) function f to produce the output. The output is then fed into another layer of neurons or are used to report the outcome of a classification or regression.

#The "neuron"

#The neurons within neural networks have none of that. Indeed, the output of each neuron can be replaced with the following formula a=sigmoid_function(summation(weight x activity))
#When there is a strong negative input, the neuron produces no output. As the input is increased, the output will start to grow slowly then increase rapidly 

#Multilayer perceptrons

# A single neuron by itself---also known as a perceptron---can do some classification on its own. However, it cannot be used to solve all types of problems; for instance, it cannot solve problems that are not “linearly separable.” 
#That is, any classification problem where the different classes cannot be separated by a straight line or a hyperplane in higher dimensions, is not solvable by a single neuron.
#However, by combining perceptrons into layers and stacking these layers onto one another---known as multilayer perceptrons---neural networks can solve nonlinearly separable problems. 

#Input layer

#The first layer of the network is the input layer. They perform no calculations, instead they simply pass input values into the next layer. The input values to this layer are the data itself, one input neuron per "feature", a.k.a. dimension or variable, in the data.For example, if we were trying to predict whether a loan applicant was a big risk for default, we will use features such as his/her salary, their current debt payments, the number of untapped available credit lines, age, etc. Each one of these numbers would be a "neuron" in the input layer.

#Another example: imagine we are trying to perform handwritten digit recognition. We are given images that are 20x20 pixels large containing grayscale intensities (i.e. 0-255). There will then be 20×20=400 input neurons. In this representation, the spatial ordering of the pixels is lost. That is, the neural network will not know that two pixels neighboring each other in the image sit next to each other. The image is "unrolled" from a matrix into a long array; it's only important to ensure consistency of the mapping from row and column index to array index across all images: row i and column j should always map to array index k. Though the spatial ordering is lost in this representation, the neural network basically sees all pixels as neighbors. Therefore, it's free to use correlations in non-neighboring pixels in classifying digits. 

#Hidden layers

#The next layer(s) after the input layer are called hidden layer(s). There can be one or many hidden layers, and, unlike the input layer, the number of hidden units per layer is not determined by the data. We can use as few or as many units as we wish, and the number can vary in each layer. 

#Moreover, there is no theory or analytical work to guide us on the number of 
#hidden layers or neurons required to solve a particular kind of problem; it's 
#more or less guesswork and rules of thumb.

#Output layer

#The final layer is the output layer. They are the neurons that calculate and report the output of the neural network. The number of neurons in the output layer depend on the application. In credit risk, for example, one could use a single output neuron for pass/reject. On the other hand, if we wanted to classify the applicant as a low-, medium-, or high-risk borrower, we would need three output neurons. Similarly for digit recognition, we would need 10 output neurons, one for each possible digit. For regression applications, we need only a single output neuron.

#How do output layers "report" their decisions? Each output unit uses some function for their input-output relationship. For example, we could use a sigmoidal function we showed above. Those output values are continuous. To convert them into a categorization, we can simply use a threshold; for example, anything above 0.5 can be a positive case and anything below can be a negative case. For regression, the output units use a linear input-output relationship and there is no need for thresholding.

#Formula for multilayer perceptrons

#With multilayer perceptrons, the formula for the outputs are very similar to a single perceptron, but now we have to index the layer and the neuron number. Starting with the first hidden layer, we write for neuron i in the 1st layer:
#a_1,i =∑_j=0 to M f(w^1_ij * x_j)
#where the first subscript indexes the layer number; the second subscript, i, indexes the neuron number within the layer.
#j indexes the input neuron number
#w_ij is the weight coming from input neuron j to neuron i in hidden layer
#x_j is the output of neuron j, and
#M is the number of features in the data
#The 0 unit base is the bias
#Their role is analogous to the constant value in linear regressions, i.e. they allow the network to output a non-zero value even when all the inputs are zero. Bias units are added to every layer in the network except the output layer; they receive no inputs from the previous layer since their output is forced to be one.

#Since each layer's activity depends upon the activity from the previous layer, these are called "feed-forward" networks---the input propagates only in one direction. There are no "lateral" connections between neurons within a layer, and no connections that go backwards from one layer to a previous one. Also, the feed-forward networks we will deal with only propagate one layer at a time, without skipping layers.

##Backpropagation

#Besides hidden layers, neural networks require one more ingredient: a learning algorithm. That is, the weights in the neural networks are adjustable. Weights change values using an algorithm known as backpropagation. It is named for the fact that errors in the network move backwards from the output layer all the way back to the input layer
#"Backprop" allows the network to perform gradient descent on the chosen cost function for the errors. That is, if a function is provided to quantify errors in the output layer, backpropagation changes the weights of the network in a way that reduces the error as quickly as possible towards locally optimal points--sets of weights that maximize performance compared to other similar weight values. It is important to note that it may be difficult to find a set of weights that outperforms all other sets. Usually the best way is to find many different locally optimal sets and compare their performance.


#%%Background

#In classification, we try to determine whether a data point is a positive or negative case of a class (or classes) based on a model induced from a set of training data. Given a training sample, i.e. a set of labeled data, we attempt to fit a function to it, such that the output of the function matches the given labels. In this mission, you learn about Multi-Layer Perceptrons (MLP), a type of neural network that performs classification by feeding the data into layers of neurons which weigh, sum, and then transform the data with a (usually nonlinear) function. The last layer of the network, the output layer, is where the results of the classification are read off.

#Calculating the outputs of a neural network from a data point is called forward propagation. In order for the network to produce the correct output, it must be able to learn. Neural networks learn by adjusting their weights through a process called backpropagation, which allows the network to minimize classification error through gradient descent.

#Data set is chemical analysis of beverages from raw materials grown in the same region

#%%Building an XOR (exclusive gate)

#XOR is an example of data which is not linearly separable

import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]]).transpose()
print(X)

#now map inputs to outputs
y = np.array([0, 1, 1, 0])

##Transfer Function

#We'll use the logistic function (1/(1+e^-x)) for the binary output

import numpy as np
def logistic(x):
    return 1/(1+np.exp(-x)) 

#Now we can logistic function values at any input:
    
x = np.linspace(-8,8,1000)
f1 = logistic(-x)
f2 = logistic(0.25*x)
f3 = logistic(x)
f4 = logistic(4*x)
from matplotlib import pyplot as plt
plt.plot(x,f1, 'b-', x,f2, 'o-', x,f3, 'p-', x,f4, 'm-')
plt.title('Logistic Function')
plt.xlabel('input')
plt.ylabel('output')
plt.axis([-10, 10, 0, 1])
plt.legend(loc = 'upper left')
plt.show()

#Higher multipliers make f(x) more step-like, while smaller (absolute) 
#multipliers make the function vary more gradually; negative multipliers flip 
#the sigmoid. The logistic function saturates to 0 on one side and to 0 on the 
#other, regardless of how high or low the input goes. Finally, note that f(0)=.5

#The logistic function is ideal for binary classification

##Network Architecture

#Input neurons are pass throughs

#Hidden Layers

#As a rule of thumb, we should have at least as many hidden neurons per layer 
#as there are input neurons, and we should keep the number of neurons per hidden 
#layer constant. Parsimony suggests that we start with a single hidden layer and 
#increase as needed, unless experience with the data dictates otherwise.

#Output neurons 

#output provide classification or regression
#%%Objective 4
import numpy as np
def logistic(x):
    return 1/(1+np.exp(-x))
x = [1, 1.2, -.9, 3.2]
w = [-.4, -2.9, -.1, 1.6]
print(x)
print(w)
print(logistic(np.dot(x,w))) #0.79
# or:
print(logistic(-0.4 - 2.9*1.2 + 0.1*0.9 + 1.6*3.2))

#or:
X = np.array([1,1.2,-0.9,3.2]).transpose()
W = np.array([-0.4,-2.9,-0.1,1.6])
y = logistic(W.dot(X))
print(y)

#For a single data point with only three features, vector multiplication is 
#merely a convenience; for larger data sets or data with many features, it is a 
#necessity. 

#The weights in a neural network are learned, but before that happens, they 
#must be initialized to random values. If all the neurons have the same 
#initial value, no learning can happen. A good rule of thumb is to pick values 
#uniformly from -a to +a, where a = sqrt(6/(N_before + N_after)) where N_before
#and N_after are the number of neurons in the layer before and after the 
#weights respectively

#%%Forward Propagation

#Forward propagation is where we calculate the activity for all of the neurons.

#Using matrix multiplication, it is possible to perform forward propagation in 
#batch, to "vectorize" your code. 

import numpy as np
# logistic
def logistic(x):
    return 1/(1+np.exp(-x))
# forward prop
def forwardprop(w1,w2,x):
    '''
    w1 are the weights between input and hidden layer
    w2 are the weights between hidden and output layer
    x is a column vector for a single data point
    '''
    a0 = np.insert(x,0,1) # insert bias unit at input layer
    z1 = w1.dot(a0) # calculate input to hidden layer
    a1 = logistic(z1) # calculate output from hidden layer
    a1 = np.insert(a1,0,1) # insert bias unit
    z2 = w2.dot(a1) # calculate input to output layer
    a2 = logistic(z2) # calculate output of neural network
    return a2
# set inputs
X = np.array([[0,0],[0,1],[1,0],[1,1]]).transpose()
m = X.shape[1]
# network architecture
nin = X.shape[0]
nhid = 2
nout = 1
# randomize the weights
# input to hidden
w1 = (np.sqrt(6)/np.sqrt(nhid+nin+1))*2*(np.random.rand(nhid,nin+1)-0.5)
# hidden to output
w2 = (np.sqrt(6)/np.sqrt(nhid+1+nout))*2*(np.random.rand(nout,nhid+1)-0.5)
# initialize the activity matrix
a2 = np.zeros(m)

### Add your code here!
for i in range(0,m):
    a2[i]= forwardprop(w1,w2,X[:,i])### Add your code here!
print(a2)

#%% Learning

#The learning algorithm should adjust the weights in the network such that it 
#reduces the output neurons' errors to an acceptable level

#The standard way of doing this is to define a cost function. Cost functions 
#define how to penalize errors in the network. They are typically defined such 
#that they increase as errors increase, and decrease as errors decrease. With 
#appropriate choice of the cost function, we can obtain an analytical answer to 
#the question: in which direction should we change our parameters, i.e. network 
#weights, given our current values and data? That is, cost functions allow us 
#to perform gradient descent to minimize the network errors (disregarding, for 
#the moment, regularization, which helps prevent overfitting).

#By iterating this process: moving, calculating the gradient, moving, etc., we 
#are guaranteed to reach a minimum of the function so long as our step sizes 
#are sufficiently small enough.

#The cost function for neural networks is typically non-convex, meaning there 
#are many local minima to which we can converge. In practice, this is usually 
#not a terrible problem for neural networks, but it may require training a 
#network multiple times to ensure a decent solution. 

##Cost Function

#J(w)=∑ from i=1 to m[−y_i*log(O(x_i))−(1−y_i)*log(1−O(x_i))]
#where J is a cost function of all network weights w.
#weights are not explicit in the function, but they are implicit in the output 
#values of the network represented as O(x). Remember that the output of the 
#network represents the classification of the data point x_i. If the output is 
#below 0.5, then the network is classifying the point as a negative case of the 
#class; if the output is above 0.5, the network is classifying the point as a 
#positive case of the class.

#The values y_i represent whether the data point is a negative or positive case 
#of the class, taking the value 0 or 1 respectively. Note, that since y_i
#must be either 0 or 1, only one of the terms in the sum survives.

#In both cases, the cost is low when O(x_i) agrees with y_i

#J(W) is summed over all m data points

#we ignored regularization in our current cost function. In real problems, it 
#is important to include it to avoid overfitting.

#regularization adds the norm (squared amplitudes) of the weights in the 
#network to the cost function. This prevents the network from decreasing the 
#cost function using arbitrarily large weights.

#Finally, the cost function we are using is good for classification problems 
#and for the logistic transfer function. It is superior to the usual squared 
#error cost function, in that it has faster convergence and less tendency to 
#get stuck in local minima---cost functions for neural networks are usually 
#non-convex. When we switch to regression problems or if we use a different 
#kind of transfer function, we will need to switch cost functions (usually to 
#ordinary-least squares).

#%%Backpropagation

#Recall, we want the gradient of the cost function, with respect to the network weights, so that we can adjust them in a way that descends the cost function the fastest at our current point. Backpropagation (or backprop) does this by first calculating the error the network produces at the current weights, and then propagating the error backward in the network. 
#First, we need to forward propagate the activity in the network for a data point. Let's do this for the first one (zeroth element). A function forwardprop is provided for you. Call the function for the first data point (X[:,0]), and get all the return variables. Call them a2,z2,a1,z1,a0 in that order.

import numpy as np
# logistic function
def logistic(x):
    return 1/(1+np.exp(-x))
# this is the derivative of logistic with respect to x
def dlogistic(x):
    u = logistic(x)
    return u*(1-u)
# forward propagation step for a single data point
def forwardprop(w1,w2,x):
    a0 = np.insert(x,0,1) # insert bias unit at input layer
    z1 = w1.dot(a0) # calculate input to hidden layer
    a1 = logistic(z1) # calculate output from hidden layer
    a1 = np.insert(a1,0,1) # insert bias unit at hidden layer
    z2 = w2.dot(a1) # calculate input to output layer
    a2 = logistic(z2) # calculate output of neural network
    return a2,z2,a1,z1,a0
# the data and desired outputs
X = np.array([[0,0],[0,1],[1,0],[1,1]]).transpose()
y = np.array([0, 1, 1, 0])
m = X.shape[1]
# architecture
nin = X.shape[0]
nhid = 2
nout = 1
# randomize weights
np.random.seed(561999213) # ensures same randomization every time for everyone
w1 = (np.sqrt(6)/np.sqrt(nhid+nin+1))*2*(np.random.rand(nhid,nin+1)-0.5)
w2 = (np.sqrt(6)/np.sqrt(nhid+1+nout))*2*(np.random.rand(nout,nhid+1)-0.5)
### Add your code here!
a2,z2,a1,z1,a0 = forwardprop(w1,w2,X[:,0])
print(a2,z2,a1,z1,a0 )

#The gradient for the weights from the hidden layer to the output layer, w2, is 
#the most straightforward. Every weight gets updated by the difference of the 
#output neuron with its target activity multiplied by the activity in the 
#hidden layer. This can be written as: 
d2 = a2 - y[0]
dw2 = np.outer(d2,a1)
print('dw2', dw2)
#d2 calculates the difference between the activity of the output neuron, a2, 
#with the target value of the first data point, y[0]. The amount that w2 needs 
#to change, dw2, is set by the multiplication between d2 and a1, the activity 
#in the hidden layer. Therefore, if d2 is close to zero, the amount of 
#adjustment is also close to zero. That's exactly what we want, because we 
#don't want to change the network very much if a2 is close to its target.

#If, however, d2 is greater than 0, then the network has output a value which is higher than the target value. We should, therefore, reduce the weights into the output neuron so that a2 will decrease. That is, we want dw2 to be positive, since it will be subtracted from w2. dw2 is positive when d2 is positive, so long as all the elements in a1 are also positive. Since, we are using the logistic function to model neural activity, this is strictly true in our case (backprop still works out even if the transfer function can have negative values, it's just easier to see how it works in our case).

#If d2 is less than 0, then the opposite of the above argument holds. We need to increase a2 and hence w2; therefore, dw2 must be negative.

#Now, why should backprop modulate dw2 based on the activity in the hidden units, a1? There isn't a great answer to this; it's just what's required to calculate the gradient. For the logistic transfer function, learning slows down for any weights emanating from a hidden neuron whose activity is near zero, i.e. has large negative inputs, even if the error of the output neuron is large. 

#For a network using tanh, which ranges from -1 to 1 instead of 0 to 1, learning slows down for weights emanating from a neuron with balanced inputs, i.e. near zero; for a neuron with large negative inputs, the activity becomes negative, which flips the direction of the gradient for that neuron. The bias unit always has an activity of 1, thus the weights emanating from it are adjusted by an amount determined by d2.

#matters for backpropagation is where the transfer function is zero and, as we see later, where the derivative with respect to the input becomes zero.

#Unfortunately, it's difficult to know a priori which transfer function is the best for learning. Each one has a unique set of conditions under which learning slows. 

#The one thing you need to ensure is that your transfer function is compatible with your cost function. For example, tanh is not compatible with our current cost function, as the derivate diverges at some values of the output neuron. 

#Let's perform a test gradient descent step only on w2, and then recalculate a2 
#with the new weights; we'll use a learning rate of 0.1.

w2tmp = w2 - 0.1*dw2
a2tmp = logistic(w2tmp.dot(a1))
print('a2tmp', a2tmp)

#This value is smaller than the original value of a2, and hence the gradient 
#descent step is moving us in the right direction, at least for one set of 
#weights and for this data point. Of course, gradient descent has to move us in 
#the right direction for all of the weights and all of the data points 
#considered together. Backprop does the former; there are some choices for the 
#latter, but we discuss that later.

#Let's push backpropagation one layer back. This step is less intuitive than 
#the first, but we'll get through it. For this, we need to pass the difference 
#in output back through the weights of the hidden layer, then multiply that by 
#the value of the derivative of the logistic function at the input level of the 
#hidden neurons. For this step, we strip out the bias unit of the hidden layer 
#because its activity is always 1. 

v2 = np.reshape(w2[:,1:],(nout,nhid))
d1 = v2.transpose().dot(d2)*dlogistic(z1)
dw1 = np.outer(d1,a0)

#For this last part: The first thing we should notice is that the quantity d2 
#is propagated back to the calculation of the gradient with respect to the 
#input layer weights. As we saw earlier, when the activity of the output neuron 
#is close to the target values, d2 is close to zero. Therefore, just as with w2, 
#w1 will not change if the output neuron is already correctly classifying the 
#point. 

#The second thing to notice is that we multiply d2 by the weights w2 (stripped 
#of the weights from the bias unit). It's as if the difference between the 
#target and the output is being passed backward to the previous layer, and 
#hence the name of the algorithm. This quantity is modulated by the derivative 
#of the logistic function, dlogistic measured at the input of the hidden units.

#TL:DR

#So what does this bell-like shape mean for how w1 will change? Well, any weight feeding into a hidden neuron that has strong positive or negative inputs does not get changed very much; those feeding into a hidden neuron that has balanced inputs, i.e. sum to zero, do get changed the most. This rule ensures that a neuron that is already driven to a "decision", i.e. being quiescent (activity close to 0, inputs strongly negative) or being active (activity close to 1, inputs strongly positive) does not have its input weights changed very much, even if the error at the output is large. Neurons that are "undecided," i.e. have inputs close to zero and activity close to 0.5 will have their weights changed the most.
#What is the logic behind this? It ensures that any weights entering a neuron with balanced inputs want to keep changing those weights until they induce an imbalanced input to the neuron, unless the output neurons produce the correct classification. So long as the difference between the output and correct classification is non-zero, the weights have an instability to drive the hidden neurons to either have activity 0 or 1.

#Finally, in this step, the gradient is multiplied by the activity of the input neurons. Unlike the hidden layer, the input layer is not bounded between the extrema of the transfer function. 

#A few important points:

#One, it's a good idea to perform feature scaling on your data. If one of the 
#features (dimensions) in your data ranges from 0 to 10 and another ranges from 
#-10000 to 10000, then it is good practice to normalize each feature so that 
#all are within the same range.

print(dw1)

#Notice that only the weights from the bias unit are updated; the weights from the two input neurons do not change at all. We could remedy this by translating our data. 
#Last, we used the outer product between d1 and a0 to obtain dw1, and unlike the calculation of dw2, d1 is not a scalar, so the outer product is useful. Note that we have 2 hidden neurons (excluding the hidden-layer bias unit) and 3 neurons in the input layer (2 input neurons and a bias unit). Therefore, our weight matrix has 2 rows and 3 columns; dw1 must be of the same dimensions. To obtain dw1, we need to multiply each value of d1, which is associated with the 2 hidden neurons, with all the values of a0, which is associated with the 3 neurons in the input layer. The outer product handles that for us compactly. It takes the 2x1 vector in d1 and matrix multiplies it with the 1x3 vector in a0 to make the 2x3 matrix in dw1.

#%%Gradient Descent with backpropogation
#In a process known as stochastic gradient descent, the network weights are 
#updated for each data point, as opposed to accumulating the gradients for all 
#data points and updating at the end, known as batch gradient descent. We could 
#also choose to perform gradient descent in mini batches, where the gradient is 
#updated after summing over some number of data points. In stochastic gradient 
#descent, it is important to randomize the order of the data points.

#Stochastic gradient descent is advantageous when the number of data points is large; the algorithm may converge before batch gradient descent performs even a single step through all of the data. However, it may not always decrease the cost function at every step, meaning a step that is good for a single data point may cause the network to perform worse for other data points.

#This is not an issue for batch gradient descent, which factors in all of the training data together. Batch gradient descent also allows for more sophisticated techniques to be employed such as line search, which will allows for finding an optimum value of the learning rate for each iteration. 

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# logistic function
def logistic(x):
    return 1/(1+np.exp(-x))
# this is the derivative of logistic with respect to x
def dlogistic(x):
    u = logistic(x)
    return u*(1-u)
# forward propagation step for a single data point
def forwardprop(w1,w2,x):
    a0 = np.insert(x,0,1) # insert bias unit
    z1 = w1.dot(a0) # calculate input to hidden layer
    a1 = logistic(z1) # calculate output from hidden layer
    a1 = np.insert(a1,0,1) # insert bias unit
    z2 = w2.dot(a1) # calculate input to output layer
    a2 = logistic(z2) # calculate output of neural network
    return a2,z2,a1,z1,a0
# backprop
def backwardprop(y,a2,a1,a0,z1,w2,nout,nhid):
    d2 = a2 - y
    dw2 = np.outer(d2,a1)
    v2 = np.reshape(w2[:,1:],(nout,nhid))
    d1 = v2.transpose().dot(d2)*dlogistic(z1)
    dw1 = np.outer(d1,a0)
    return dw1,dw2
# batch forward propagation
def batchforwardprop(x,w1,w2):
    m = x.shape[1]
    n = x.shape[0]
    h = w1.shape[0]
    a0 = np.reshape(np.append(np.ones(m),x),(n+1,m))
    z1 = w1.dot(a0)
    a1 = logistic(z1)
    a1 = np.reshape(np.append(np.ones(m),a1),(h+1,m))
    a2 = logistic(w2.dot(a1))
    return a2
# cost function
def costfunction(y,a):
    return -y*np.log(a)-(1-y)*np.log(1-a)
# learn
def learn(Nsteps,eta,X,y,w1,w2):
    m = X.shape[1]
    nout = w2.shape[0]
    nhid = w1.shape[0]
    costLog = np.zeros(Nsteps)
    for i in range(0,Nsteps):
        order = np.random.permutation(m) # get random ordering of data points
        cost = 0
        for j in range(0,m):
            idx = order[j]
            a2,z2,a1,z1,a0 = forwardprop(w1,w2,X[:,idx])
            dw1,dw2 = backwardprop(y[idx],a2,a1,a0,z1,w2,nout,nhid)
            w1 -= eta*dw1
            w2 -= eta*dw2
            cost += costfunction(y[idx],a2)
        costLog[i]=cost
        if i%500==0:
            print('iteration %d:, cost = %.6f'%(i,cost[0]))
    return w1,w2, costLog
# plot the output
def plotoutput(x,y,w1,w2):
    m = x.shape[1]
    xmin = x.min()
    xmax = x.max()
    xr = xmax-xmin
    ngrid = 13
    xl=np.linspace(xmin-xr*0.1,xmax+xr*0.1,ngrid)
    [X1,X2]=np.meshgrid(xl,xl)
    A=batchforwardprop(np.array([X1.flatten(), X2.flatten()]),w1,w2)
    A = np.resize(A,(ngrid,ngrid))
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X1, X2, A, rstride=1, cstride=1, linewidth=0.5, cmap=cm.autumn, antialiased=False)
    for i in range(0,m):
        ax.plot([x[0,i],x[0,i]],[x[1,i],x[1,i]],[0,y[i]],'k')
        ax.plot([x[0,i]],[x[1,i]],[y[i]],'og',markersize=10)
    pl.setp(ax,xlabel='X1', ylabel='X2')
    pl.show()
### Feel free to adjust code below this
# the data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).transpose()
y = np.array([0, 1, 1, 0])
# weights
nin = X.shape[0]
nhid = 4 #Change number of hidden layers
nout = 1
np.random.seed(98110044)
w1 = (np.sqrt(6)/np.sqrt(nhid+nin+1))*2*(np.random.rand(nhid,nin+1)-0.5)
w2 = (np.sqrt(6)/np.sqrt(nhid+1+nout))*2*(np.random.rand(nout,nhid+1)-0.5)
plotoutput(X,y,w1,w2)
# learn
#Change first and second numbers for number of iterations and learning rate
Nsteps = 1000
w1,w2, costLog=learn(Nsteps,5,X,y,w1,w2) #learn(Nsteps,eta,X,y,w1,w2):
batchforwardprop(X,w1,w2) # batchforwardprop calculates forward prop for all data at once.
plotoutput(X,y,w1,w2)
pl.plot(range(0,Nsteps),costLog)
pl.xlabel('Step number')
pl.ylabel('Cost')

# plots the output of the neural network along with the data points
#Video of the surface plot learning XOR:
#https://www.youtube.com/watch?v=5l6fWHk9rCg&color2=FBE9EC&rel=0&showsearch=0&version=3&modestbranding=1

#%% PyBrain

# Think keras is a better implementation to use now

#https://github.com/keras-team/keras

# Imports
import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SigmoidLayer
# Data and outputs
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]]).transpose()
y = np.array([0, 1, 1, 0])
data = ClassificationDataSet(2,1)
for i in range(X.shape[1]):
	data.addSample(X[:,i],y[i])
#Build a network with 2 input neurons, 4 hidden neurons, and 1 output neuron using PyBrain. Make sure you are using SigmoidLayer for both hidden and output layers. 
net = buildNetwork(2,4,1, hiddenclass=SigmoidLayer, outclass=SigmoidLayer)
#Create a BackpropTrainer with a learning rate of 1, momentum of 0.001, weight 
#decay of 0.000001, and uses batch learning.
trainer = BackpropTrainer(net, dataset=data, learningrate=1, momentum=0.001, weightdecay=0.000001, batchlearning=True)
#Train the network for 3000 epochs.
Nsteps = 3000
N=Nsteps/100
for i in range(N):
#Display the cost every one hundred steps. To do this. use a loop where you 
    #toggle trainer.verbose= between True andFalse.
    trainer.trainEpochs(99)
    trainer.verbose=True
    trainer.trainEpochs(1)
    trainer.verbose=False
    
#To check how the network classifies the data, type:
    
print(net.activateOnDataset(data))

#You can also have the network activate for an arbitrary data point:
net.activate([0,0])
#gives: array([[ 0.01943982],
#       [ 0.9722507 ],
#       [ 0.99191829],
#       [ 0.02710997]])

#which outputs:
#array([ 0.93072265])

#Let's standardize the data:

from scipy import stats
X = stats.zscore(X, axis=0)

#%% Beer classification
# Imports
import numpy as np
from scipy import stats
from PyBrain.datasets import ClassificationDataSet
# Data and outputs
datain=np.loadtxt(open("4-1-4 beerData.csv","rb"), delimiter=",", skiprows=0)
y = datain[:,0]-1 # 178x1 vector classifications
X = datain[:,1:] # 178x13 matrix of data points
X = stats.zscore(X, axis=0) # normalize the data by feature
m = X.shape[0] # number of data points
### Add your code here!
data = ClassificationDataSet(13)
for i in range(m):
	data.addSample(X[i,:],y[i])    

#We need to perform two last steps on the data before training. One, the 
#classifications are currently valued 0 through 2. The neural network needs 
#them to be translated into a vector, i.e. [1,0,0] for 0, [0,1,0] for 1, and 
#[0,0,1] for 2. Fortunately, PyBrain has a method for this.
    
# before
print(data.data['target'][[0,65,150],:])
#[[0]
# [1]
# [2]]
# after
data._convertToOneOfMany()
print(data.data['target'][[0,65,150],:])
#[[1 0 0]
# [0 1 0]
# [0 0 1]]

#The next thing we need to do is to split the data into two: a training set and 
#a test set. The neural network learns from the training set.

tstdata, trndata = data.splitWithProportion(0.25)

                                            