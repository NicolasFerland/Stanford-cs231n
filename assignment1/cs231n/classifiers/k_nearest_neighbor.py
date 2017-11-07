import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i][j] = np.linalg.norm(X[i] - self.X_train[j])
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #print(X[i].shape)  # There is 3072 pixels = 32*32*3
      #print(self.X_train.shape) # There is 5000 images with 3072 pixels
      #print((X[i]-self.X_train).shape) # There is still 5000 images with 3072 pixels
      dists[i,:] = np.linalg.norm(X[i] - self.X_train,axis=1)
      #print(dists[i].shape) # Norm should have be takeb on the 3072 pixels. There should be 5000 images
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
        
    # X has 500 images with 3072 pixels, X_train has 5000 images with 3072 pixels
    # We want a matrix which is 500x5000 and sumed over the 3072.
    
    #Xtmp = X.reshape(num_test,3072,1).repeat(num_train,axis=2).swapaxes(1,2)
    #Xtraintmp = self.X_train.reshape(num_train,3072,1).repeat(num_test,axis=2).swapaxes(1,2).swapaxes(0,1)
    #dists = np.linalg.norm(Xtmp - Xtraintmp, axis=-1) # Sum over last axis with is the one with 3072 pixels
    # doesn't work because of memory
    
    # We want sqrt(Sum((X - Xtrain)^2,axis=pixels)) =           Where X and Xtrain have shape (
    #         sqrt(Sum(X^2,axis=pixels)-2*Sum(X*Xtrain,axis=pixels)+Sum(Xtrain^2,axis=pixels))
    a = np.sum(X*X,axis=-1) # multiplication is elementwise
    b = X.dot(self.X_train.transpose()) # multiplication must make the shape be (num_test,num_train) by summing over pixel axis.
    # That means shape must be (num_test,pixels).dot(pixels, num_train)
    c = np.sum(self.X_train*self.X_train,axis=-1) # multiplication is elementwise
    a0 = a.reshape(num_test,1).repeat(num_train,axis=1)
    c0 = c.reshape(num_train,1).repeat(num_test,axis=1).transpose()
    dists = np.sqrt(a0-2*b+c0)

    # test
    #idtrain = 3
    #idtest = 7
    #print(np.linalg.norm(X[idtest] - self.X_train[idtrain]),dist[idtest,idtrain])
    #print(X[idtest].dot(X[idtest]),a[idtest])
    #print(X[idtest].dot(self.X_train[idtrain]),b[idtest,idtrain])      
    #print(self.X_train[idtrain].dot(self.X_train[idtrain]),c[idtrain])
    
    # print(dists.shape) # Want to have (num_test, num_train)
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      for j in dists[i].argsort()[:k]:    # First argument of dists is the test index, 2nd is the train index
        closest_y.append(self.y_train[j])
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #print(closest_y)
      y_pred[i] = max(map(lambda val: (closest_y.count(val), val), set(closest_y)))[1]
      # Set make that we keep only unique element, then we count the number of value for
      # each of them. We take the one that has the max count, then we pick the value
      # which is the [1] of the tupple.
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

