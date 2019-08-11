import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]
		###############################################
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		self.data = np.matmul(X,self.weights)+self.biases
		return sigmoid(self.data)
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		# raise NotImplementedError
		self.biases = self.biases - lr*sum((sigmoid(self.data)*(1-sigmoid(self.data)))*delta)
		new_delta = np.matmul(delta,self.weights.T)
		self.weights = self.weights - lr*(np.matmul(activation_prev.T,(sigmoid(self.data)*(1-sigmoid(self.data)))*delta))
		return new_delta
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		output2 = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		# self.data.shape = (n,self.out_depth,self.out_row,self.out_col)
		for i in range(n):
			for j in range(self.out_depth):
				# weights[j], X[i], data[i]
				for k1 in range(self.out_row):
					for k2 in range(self.out_col):
						m = k1*self.stride
						n = k2*self.stride
						if(m+self.filter_row <= self.in_row and n+self.filter_col <= self.in_col):
							output2[i,j,k1,k2] = np.sum(X[i,:,m:m+self.filter_row,n:n+self.filter_col]*self.weights[j]) + self.biases[j]

		self.data = output2
		return sigmoid(output2)
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size


		###############################################
		# TASK 2 - YOUR CODE HERE

		xdelta = delta*sigmoid(self.data)*(1-sigmoid(self.data))
		grad_w = np.zeros([self.out_depth, self.in_depth, self.filter_row, self.filter_col])
		# del_activation_prev = np.zeros([n,self.in_depth,self.in_row,self.in_col])
		grad_bias = np.zeros([self.out_depth])
		new_delta = np.zeros([n,self.in_depth,self.in_row,self.in_col])

		for i in range(n):
			for j in range(self.out_depth):
		# 		# weights[j], X[i], data[i]
				for k1 in range(self.out_row):
					for k2 in range(self.out_col):
						m = k1*self.stride
						n = k2*self.stride
						if(m+self.filter_row <= self.in_row and n+self.filter_col <= self.in_col):
							grad_w[j] += activation_prev[i,:,m:m+self.filter_row,n:n+self.filter_col]*xdelta[i,j,k1,k2]
							grad_bias[j] += xdelta[i,j,k1,k2]
							new_delta[i,:,m:m+self.filter_row,n:n+self.filter_col] += xdelta[i,j,k1,k2]*self.weights[j]


		self.weights -= lr*grad_w
		self.biases -= lr*grad_bias

		return new_delta


		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		# raise NotImplementedError
		output = np.zeros((n,self.out_depth,self.out_row,self.out_col))
		for i in range(n):
			for j in range(self.out_depth):
				for k1 in range(self.out_row):
					for k2 in range(self.out_col):
						m = k1*self.stride
						n = k2*self.stride
						if(m+self.filter_row <= self.in_row and n+self.filter_col <= self.in_col):
							output[i,j,k1,k2] = sum(sum(X[i,j,m:m+self.filter_row,n:n+self.filter_col]))/(self.filter_row*self.filter_col)

		return output
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		new_delta = np.zeros([n,self.in_depth,self.in_row,self.in_col])


		for i in range(n):
			for j in range(self.out_depth):
				for k1 in range(self.out_row):
					for k2 in range(self.out_col):
						m = k1*self.stride
						n = k2*self.stride
						if(m+self.filter_row <= self.in_row and n+self.filter_col <= self.in_col):
							new_delta[i,j,m:m+self.filter_row,n:n+self.filter_col] += delta[i,j,k1,k2]/(self.filter_row*self.filter_col)
		
		return new_delta
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
