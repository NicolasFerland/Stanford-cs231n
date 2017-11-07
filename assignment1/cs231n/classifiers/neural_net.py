from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
import copy

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    #scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # input - fully connected layer - ReLU - fully connected layer - softmax
    
    # 1st layer
    f0 = X.dot(W1) + b1
    # print(f)
    # 1st relu
    f = np.maximum(f0, 0)
    # print(f)
    # 2nd layer
    # for reason unknown to me, the result of the 2nd layer is the score wanted by the exercice
    scores = f.dot(W2) + b2
    # f = f.dot(W2) + b2
    # print(f)
    # softmax
    # ef = np.exp(f)
    # sf = np.sum(ef,axis=1)
    # scores = ef/sf[:, np.newaxis]

    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    scores -= np.max(scores,axis=1)[:, np.newaxis] # Make it numerically stable
    ef = np.exp(scores)
    sf = np.sum(ef,axis=1)
    cf = scores[np.arange(0,len(y)),y] # f for correct class
    loss = np.sum(-cf + np.log(sf))
    loss /= N
    loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    # include bias in the regularization. I don't think I should, but loss is not equal to correct
    #loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2)+ b1.dot(b1) + b2.dot(b2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # M is NxC, it's ef[i,l]/sf[i] for N=i C=l (then modify for l=y[i])
    M = ef / sf[:, np.newaxis]
    # it's ef[i,y[i]]/sf[i]-1 for N=i C=y[i]
    M[np.arange(0,len(y)),y] -= 1
    # f is the input of the 2nd layer, that replace the X in the softmax gradiet
    dW = f.transpose().dot(M)
    dW /= N
    dW += 2 * reg * W2
    grads['W2'] = dW
    grads['b2'] = np.sum(M,axis=0)/N
    
    # for the first layer gradient, we have dL/dW = dL/df * df/dW
    # df[n,h1]/dW[d,h2] = X[n,d] delta(h1,h2) omega(X[n,d]*W[d,h1]+b[h1])
    # dL/df[n,h] is like before but replace f and W2.
    passRelu = np.where(f,1,0) # When f element is 0, it's false, so it's the second choice. NxH
    df = M.dot(W2.transpose())*passRelu    # NxH      It's dL/df[n,h]*omega(X[n,d]*W[d,h1]+b[h1])
    # X is NxD, df is NxH, W1 is DxH. As it should be we sum over
    grads['W1'] = X.transpose().dot(df)/N + 2 * reg * W1 # W1 is DxH
    grads['b1'] = np.sum(df,axis=0)/N
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      #X_batch = None
      #y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      num_dim = X.shape[1]
      index = np.random.choice(num_dim,batch_size)
      X_batch = X[index,:]
      y_batch = y[index]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params.update((x, y-learning_rate*grads[x]) for x, y in self.params.items())
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    #y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X) # loss give scores in absence of y
    # Scores is (N, C)
    y_pred = np.argmax(scores,axis=1) # array size N
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

  # I add train2 myself to do early stopping
  def train2(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, tolerence = 20,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - tolerence: Number of epoch for which val_acc diminished to get before stopping.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    it = 0
    badEpoch = 0
    best_val = -1
    epoch = 0
    while badEpoch < tolerence:
      it += 1
      num_dim = X.shape[1]
      index = np.random.choice(num_dim,batch_size)
      X_batch = X[index,:]
      y_batch = y[index]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params.update((x, y-learning_rate*grads[x]) for x, y in self.params.items())

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Check if val_acc increased
        if val_acc > best_val:
            best_val = val_acc
            best_params = copy.deepcopy(self.params) # So that we can go back to best result
            badEpoch = 0
        else:
            badEpoch += 1

        # Decay learning rate
        learning_rate *= learning_rate_decay
        
        # Verify it works well
        if verbose:
            epoch += 1
            print('iteration %d, num_epoch %d, badEpoch %d, loss %f, train_acc %f, val_acc %f, best_val %f' % (it, epoch, badEpoch, loss, train_acc, val_acc, best_val))

    self.params = best_params
    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }


