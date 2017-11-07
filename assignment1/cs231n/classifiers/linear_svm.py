import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]
        dW[:,y[i]] -= X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W) # score is a matrix NxC
  correct_class_score = scores[np.arange(0,len(y)),y][:, np.newaxis]
  # https://docs.scipy.org/doc/numpy-1.10.1/user/basics.indexing.html
  # To avoid "frames are not aligned" error, correct_class_score must be Nx1. One can use [:, np.newaxis] to add x1 to an array
  # correct_class_score is an array N. y is an array N containing the value of C for each N.
  margin = scores - correct_class_score + 1 # matrix NxC. However, we get 1 rather than 0 for correct_class
  np.maximum(margin, 0, margin) # keep only positive term, in-place function
  loss = np.sum(margin)/num_train - 1 # To remove the 1 for every correct_class in each row.
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # M is the matrix NxC which is 1 if the training example n has a positive margin for class c, -Sum_j if c is the true class of n, 0 otherwise
  #print(X.shape[0], W.shape[1], margin.shape)
  #print(margin[0:10,:])
  M = np.where(margin,1,0) # For margin element = 0, it's false, so pick 0. If != 0, it's true, so pick 1.
  #print(M[0:10,:])
  M[np.arange(0,len(y)),y] = -np.sum(M,axis=1)+1 # Replace every case where c is the true class by -Sum_j, that is -1*number of time margin>0 for this row. The true class itself is 1, so it stays 1 in M, so it is counted and must be removed by a +1.
  #print(M[0:10,:])
  #print(np.sum(M,axis=1)[0:10])
  dW = X.transpose().dot(M) / num_train # M is a matrix NxC
  dW += 2 * reg * W # add regularization
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
