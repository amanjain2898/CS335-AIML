from __future__ import division
import util
import numpy as np
import sys
import random
import math
from operator import add

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

def small_classify(y):
    classifier, data = y
    return classifier.classify(data)

class AdaBoostClassifier:
    """
    AdaBoost classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, boosting_iterations):
        self.legalLabels = legalLabels
        self.boosting_iterations = boosting_iterations
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.boosting_iterations)]
        self.alphas = [0]*self.boosting_iterations

    def train( self, trainingData, trainingLabels):
        """
        The training loop trains weak learners with weights sequentially. 
        The self.classifiers are updated in each iteration and also the self.alphas 
        """
        
        self.features = trainingData[0].keys()
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        sample_weights = []

    	for l in range(len(trainingData)):
    		sample_weights.append(1/len(trainingData))

        for k in range(self.boosting_iterations):
        	error = 0

        	self.classifiers[k].train(trainingData,trainingLabels,sample_weights)

        	guesses = self.classifiers[k].classify(trainingData)
        	for j in range(len(trainingData)):
        		if guesses[j] != trainingLabels[j]:
        			error = error + sample_weights[j]

        	total = 0
        	for j in range(len(trainingData)):
        		if guesses[j] == trainingLabels[j]:
        			# print("hello",error/(1-error))
        			sample_weights[j] *= (error)/(1-error)
        		total += sample_weights[j]

        	for j in range(len(trainingData)):
        		sample_weights[j] /= total

        	self.alphas[k] = math.log((1-error)/error)




    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        guesses = []

        for datum in data:
        	sigma = 0
        	for i in range(self.boosting_iterations):
    			sigma += self.alphas[i]*self.classifiers[i].classify([datum])[0]
        	guess = int(sigma/abs(sigma))
        	guesses.append(guess)
        return guesses
