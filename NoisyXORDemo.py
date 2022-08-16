#!/usr/bin/python

import numpy as np
import pyximport; pyximport.install(setup_args={
                              "include_dirs":np.get_include()},
                            reload_support=True)

import MultiClassTsetlinMachine

# Parameters for the Tsetlin Machine
T = 15 
s = 3.9
number_of_clauses = 20
states = 100 

# Parameters of the pattern recognition problem
number_of_features = 12
number_of_classes = 2

# Training configuration
epochs = 200

# Loading of training and test data
training_data = np.loadtxt("NoisyXORTrainingData.txt").astype(dtype=np.int32)
test_data = np.loadtxt("NoisyXORTestData.txt").astype(dtype=np.int32)

X_training = training_data[:,0:12] # Input features
y_training = training_data[:,12] # Target value

X_test = test_data[:,0:12] # Input features
y_test = test_data[:,12] # Target value

# This is a multiclass variant of the Tsetlin Machine, capable of distinguishing between multiple classes
tsetlin_machine = MultiClassTsetlinMachine.MultiClassTsetlinMachine(number_of_classes, number_of_clauses, number_of_features, states, s, T)

# Training of the Tsetlin Machine in batch mode. The Tsetlin Machine can also be trained online
tsetlin_machine.fit(X_training, y_training, y_training.shape[0], epochs=epochs)

# Some performance statistics

print "Accuracy on test data (no noise):", tsetlin_machine.evaluate(X_test, y_test, y_test.shape[0])
print "Accuracy on training data (40% noise):", tsetlin_machine.evaluate(X_training, y_training, y_training.shape[0])
print
print "Prediction: x1 = 1, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([1,0,1,1,1,0,1,1,1,0,0,0],dtype=np.int32))
print "Prediction: x1 = 0, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([0,1,1,1,1,0,1,1,1,0,0,0],dtype=np.int32))
print "Prediction: x1 = 0, x2 = 0, ... -> y = ", tsetlin_machine.predict(np.array([0,0,1,1,1,0,1,1,1,0,0,0],dtype=np.int32))
print "Prediction: x1 = 1, x2 = 1, ... -> y = ", tsetlin_machine.predict(np.array([1,1,1,1,1,0,1,1,1,0,0,0],dtype=np.int32))


