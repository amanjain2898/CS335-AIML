# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation
import util
import numpy as np
import sys
import random
from numpy.random import choice

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    Note that this time around the weights are referred to just a single lable instead of a list.
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations

        ##################IMPORTANT######################
        # The self.weights is just one instance of Counter unlike last time
        #################################################
        self.weights = util.Counter()

    def setWeights(self, weights):
        assert type(weights) == type(self.weights)
        self.weights = weights

    def train( self, trainingData, trainingLabels, sample_weights=None):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the assignment description for details.

        Use the provided self.weights data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        if sample_weights is not None:
            trainingData, trainingLabels = self.sample_data(trainingData, trainingLabels, sample_weights)

        for iteration in range(self.max_iterations):
            for i in range(len(trainingData)): 
                # "*** YOUR CODE HERE ***"
                # util.raiseNotDefined()
	            guess = int(np.sign(self.weights * trainingData[i]))
	            if guess == 0:
	                guess = np.random.choice(self.legalLabels)
	            if guess != trainingLabels[i]:
	            	if guess == -1:
	            		self.weights += trainingData[i]
	            	else:
	            		self.weights -= trainingData[i]



    def sample_data(self, trainingData, trainingLabels, sample_weights):
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        idx = np.arange(len(trainingData))

        x1 = util.nSample(sample_weights,idx,int(0.5*len(trainingData)))


        tData = []
        tLabels = []

        for i in range(len(x1)):
        	tData.append(trainingData[x1[i]])
        	tLabels.append(trainingLabels[x1[i]])

        return tData,tLabels
        


    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the assignment description for details.

        Note that this time around the labels are just -1 and 1. 

        Recall that a datum is a util.counter.
        """
        guesses = []
        vectors = util.Counter()
        for datum in data:
            guess = int(np.sign(self.weights * datum))
            if guess == 0:
                guess = np.random.choice(self.legalLabels)
            guesses.append(guess)
        return guesses
