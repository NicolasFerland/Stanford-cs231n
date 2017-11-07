import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
        # f is an array of the score
        f = X[i].dot(W)
        # for numerical stability, we shift f so the the max is 0. It doesn't change the softmax function because it is a quotient of exponential.
        f -= np.max(f)
       
        # ef is np.exp(f)
        # softmax is ef/sum(ef) and loss is -log(ef/sum(ef)) but one can write that 
        # -f + log(sum(ef)) which is easier.
        ef = np.exp(f)
        sf = np.sum(ef)
        
        loss += - f[y[i]] + np.log(sf)
        dW[:,y[i]] -= X[i,:] # first term
        for l in range(num_class):
            dW[:,l] += X[i,:] * ef[l] / sf # 2nd term
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
        
  # regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  f = X.dot(W) # same as before but now the first index is the training case.
  f -= np.max(f,axis=1)[:, np.newaxis]
  ef = np.exp(f)
  sf = np.sum(ef,axis=1)
  cf = f[np.arange(0,len(y)),y] # f for correct class
  loss = np.sum(-cf + np.log(sf))
    
  # M is NxC, it's ef[i,l]/sf[i] for N=i C=l (then modify for l=y[i])
  M = ef / sf[:, np.newaxis]
  # it's ef[i,y[i]]/sf[i]-1 for N=i C=y[i]
  M[np.arange(0,len(y)),y] -= 1
  dW = X.transpose().dot(M)
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
        
  # regularization
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

