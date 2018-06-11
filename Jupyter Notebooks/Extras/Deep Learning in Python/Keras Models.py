# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:34:58 2018

@author: 593787 Robert Piazza - Booz Allen Hamilton
"""

"Datacamp-Deep Learning in Python-Building Deep Learnign Models with keras"

"Specify architecture"
"Compile"
"Fit"
"Predict"

#data frame on wage/hour for predictions

"Specify Architecture"
# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

"Compile"
"Adam is usually a good optimizer"
"Mean_squared_error is a good loss function"
"Scaling data before fitting can ease optimization"

# Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Verify that model contains information from compiling
print("Loss function: " + model.loss)

"Fit the model"

# Fit the model
model.fit(predictors, target)


#%% Categorical Classification

"Use Categorical_crossentropy loss function"
"Similar to log loss: lower is better"
"Add metrics [accuracy] to compile steps to make easy to understand diagnostics"
"Output layer has a separate node for each possible outcome and uses 'softmax' activation instead of 'relu'"

from keras.utils import to_categorical

#read
data = pd.read_csv('basketball_shot_log.csv')

#pull out just predictors
predictors = data.drop(['shot_result'],axis=1).as_matrix()

#create targets splitting the one column to multiple columns using to_categorical
target=to_categorical(data.shot_result)

model = Sequential()
# Add the first layer
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(100, activation='relu'))

# Add the third layer
model.add(Dense(100, activation='relu'))

# Add the output layer with two nodes for two categories and using softmax activation function
model.add(Dense(2, activation='softmax'))

#Compile
model.compile(optimizer='adam', loss  = 'categorical_crossentropy',metrics=['accuracy'])

#fit
model.fit(predictors,target)

#%% Saving and Reloading models


from keras.models import load_model
model.save('model_file.h5')
my_model=load_model('my_model.h5')
predictions = my_model.predict(data_to_predict_with)
probability_true=predictions[:,1]

#%% Exercise
# Specify, compile, and fit the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape = (n_cols,)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='sgd', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(predictors, target)

# Calculate predictions: predictions
predictions = model.predict(pred_data)

# Calculate predicted probability of survival: predicted_prob_true
predicted_prob_true = predictions[:,1]

# print predicted_prob_true
print(predicted_prob_true)
