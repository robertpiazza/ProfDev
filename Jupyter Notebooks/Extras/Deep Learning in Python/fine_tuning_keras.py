# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 12:57:08 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

"Datacamp-Deep Learning in Python: Fine-tuning keras models"

#%%Understanding model optimization

"simplest learning optimizer: stochastic gradient descent- or sgd"
"try out the model with different learning rates in a for loop"

from keras.optimizers import SGD

def get_new_model(input_shape=input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    return(model)

lr_to_test = [0.000001, 0.01, 1]

for lr in lr_to_test:
    model = get_new_model()
    my_optimizer = SGD(lr=lr)
    model.compile(optimizer = my_optimizer,loss='categorical_crossentropy')
    model.fit(predictors,target)

"dying neuron problem- when negative weight continues and never adds anything to the model"
"if you're not getting good results, chaning the activation function may help"

#%%Diagnosing optimization problems

"Learning rate too high, low, or poor activation function can prevent a model from showing improved loss in its first few epochs"

#%%Changing optimization parameters

# Import the SGD optimizer
from keras.optimizers import SGD

def get_new_model(input_shape=input_shape):
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    return(model)
    
# Create list of learning rates: lr_to_test
lr_to_test = [.000001, .01, 1]

# Loop over learning rates
for lr in lr_to_test:
    print('\n\nTesting model with learning rate: %f\n'%lr )
    
    # Build new model to test, unaffected by previous models
    model = get_new_model()
    
    # Create SGD optimizer with specified learning rate: my_optimizer
    my_optimizer = SGD(lr=lr)
    
    # Compile the model
    model.compile(optimizer = my_optimizer,loss='categorical_crossentropy')
    
    # Fit the model
    model.fit(predictors,target)

#%%Model validation
    
"We don't have to do k-fold cross validation tests because a single validatio test is usually pretty large"
"Commonly use validation split rather than cross validation-large computational expense"
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(predictors, target, validation_split=0.3)

"early stopping model-how long a model can go without improvement- 2 or3 is reasonable"
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)

model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks = [early_stopping_monitor])

"validation scores let you experiment"


#%%Evaluating model accuracy on validation dataset

#Exercise 1
# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(predictors,target,validation_split=0.3)



#%%Early stopping: Optimizing the optimization

# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = input_shape))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit the model
model.fit(predictors, target, validation_split=0.3, epochs=30, callbacks = [early_stopping_monitor])

#%%Experimenting with wider networks

# Import EarlyStopping
from keras.callbacks import EarlyStopping

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]
input_shape = (n_cols,)

# Specify the model
model_1 = Sequential()
model_1.add(Dense(10, activation='relu', input_shape = input_shape))
model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(2, activation='softmax'))

# Compile the model
model_1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Create the new model: model_2
model_2 = Sequential()

# Add the first and second layers
model_2.add(Dense(100, activation='relu', input_shape=input_shape))
model_2.add(Dense(100, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit model_1
model_1_training = model_1.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Fit model_2
model_2_training = model_2.fit(predictors, target, epochs=15, validation_split=0.2, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()

#%%Adding layers to a network

# The input shape to use in the first hidden layer
input_shape = (n_cols,)

# Create the new model: model_2
model_2 = Sequential()

# Add the first, second, and third hidden layers
model_2.add(Dense(50, activation='relu', input_shape=input_shape))
model_2.add(Dense(50, activation='relu'))
model_2.add(Dense(50, activation='relu'))

# Add the output layer
model_2.add(Dense(2, activation='softmax'))

# Compile model_2
model_2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit model 1
model_1_training = model_1.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Fit model 2
model_2_training = model_2.fit(predictors, target, epochs=20, validation_split=0.4, callbacks=[early_stopping_monitor], verbose=False)

# Create the plot
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
#%%Thinking about model capacity

"Underfitting vs. overfitting is closely related to model capacity"

"Measure loss, then adjust the architecture with more layers or more nodes"

"Adjust the number of layers or nodes until the lowest loss is created"

"Start with a small network"
"Get the validation score"
"Keep increasing capacity until validation score is no longer improving."
"Back track a little on the architecture but you're probably near the ideal."


#%%Expereimenting with model structures

"You've just run an experiment where you compared two networks that were identical except that the 2nd network had an extra hidden layer. You see that this 2nd network (the deeper network) had better performance. Given that, which of the following would be a good experiment to run next for even better performance?"

"Use more units in each hidden layer"

#%%Stepping up to images

#Recognizing handwri!en digits
#● MNIST dataset
#● 28 x 28 grid fla!ened to 784 values for each image
#● Value in each part of array denotes darkness of that pixel

#%%Building your own digit recognistion model

# Create the model: model
model = Sequential()

# Add the first hidden layer
model.add(Dense(50, activation='relu', input_shape=(784,)))

# Add the second hidden layer
model.add(Dense(50, activation='relu'))

# Add the output layer
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.3)


#%%Final Thoughts

#Next Steps

#Start with a standard prediction problems on tables of numbers
#Images (with convolutional neural networks) are common next steps
#Kaggle is a great place to experiment
#keras.io has great documentation
#GPU provides dramatic speedups
#Need CUDA compatible GPU
#GPU cloud: http://bit.ly/2mYQXQb