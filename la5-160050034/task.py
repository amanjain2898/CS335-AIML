import numpy as np
from utils import *

# import matplotlib.pyplot as plt

def preprocess(X, Y):
	''' TASK 0
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	# pass
	X1 = np.ones([X.shape[0],1])
	for i in range(1,X.shape[1]):
		if(not(isinstance(X[0][i],str))):
			arr=[]
			mean = np.mean(X[:,i])
			std= np.std(X[:,i])
			for j in range(X.shape[0]):
				val = (X[j][i] - mean)/std
				arr.append(val)
			X1 = np.insert(X1,X1.shape[1],arr,axis=1)
		else:
			label_arr = X[:,i]
			labels = np.unique(label_arr).tolist()
			newX = one_hot_encode(X[:,i],labels)
			X1= np.append(X1,newX,axis=1)

	X1= X1.astype(float)
	Y = Y.astype(float)

	return X1,Y
	

def grad_ridge(W, X, Y, _lambda):
	'''  TASK 2
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	# pass
	# print(type(_lambda))
	Grad = X.T @ (X @ W-Y) + _lambda * W

	return Grad

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' TASK 2
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	# pass
	W = np.zeros((X.shape[1],1))
	for i in range(max_iter):
		G = grad_ridge(W,X,Y,_lambda)
		if(np.linalg.norm(G,ord=2) < epsilon):
			break
		W = W - lr*G
	return W

def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' TASK 3
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	# pass
	sse_list = []
	num = X.shape[0]
	range1 = int(num/k)
	X1 = np.array_split(X,k)
	Y1 = np.array_split(Y,k)
	# print(X1[0].shape)
	# print(len(X1))

	for l in lambdas:
		sse_sum = 0
		for i in range(len(X1)):
			validation_set_x = X1[i]
			validation_set_y = Y1[i]
			training_set_x = np.delete(X1,i,0)
			training_set_y = np.delete(Y1,i,0)
			training_set_x = np.concatenate(training_set_x)
			training_set_y = np.concatenate(training_set_y)
			W1 = algo(training_set_x,training_set_y,l)
			sse_sum += sse(validation_set_x,validation_set_y,W1)
		sse_list.append(sse_sum/k)

	return sse_list


def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' TASK 4
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	'''
	W = np.zeros((X.shape[1],1))
	denom = np.zeros((X.shape[1],1))
	X_Y = np.zeros((X.shape[1],1))

	for j in range(X.shape[1]):
		denom[j] = 2*(X[:,j].T @ X[:,j])
		X_Y[j] = X[:,j].T @ Y

	for i in range(max_iter):
		for j in range(X.shape[1]):
			W_prev = W[j][0]
			d1 = _lambda
			d2 = -_lambda
			X_t = X[:,j].T
			if(denom[j] == 0):
				continue
			W_k_exp1 = ( 2*(X_Y[j] - X[:,j].T @ X[:,:j] @ W[:j] - X[:,j].T @ X[:,j+1:] @ W[j+1:]) - d1 )
			W_k_exp2 = W_k_exp1 + d1 - d2
			if(W_k_exp1 > 0) :
				W[j] = W_k_exp1
			elif(W_k_exp2 < 0):
				W[j] = W_k_exp2
			else:
				W[j] = 0
			W[j] /= denom[j]
	return W


if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	# lambdas = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	lambdas = [200000,225000,250000,275000,300000,325000,350000,400000,425000,450000,475000,500000,525000,550000,575000,600000]
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, coord_grad_descent)
	plot_kfold(lambdas, scores)
	