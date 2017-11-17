from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    #out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = x.reshape(x.shape[0],w.shape[0]).dot(w) + b # by default reshape change the last index first which is what we want since we don't want the first index to be changed.
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    #dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # out = X.w + b   where X=(N,D) is x reshaped.
    # da = dL/da ~ dL/dout dout/da       dL/dout = dout (N,M)  General formula
    dX = dout.dot(w.transpose())
    dx = dX.reshape(x.shape)
    X = x.reshape(x.shape[0],w.shape[0])
    dw = X.transpose().dot(dout)
    db = np.sum(dout,axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x,0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # dL/dx = dL/dout dout/dx                 dL/dout = dout
    # dout/dx = np.where(x<0,0,1)
    dx = np.where(x<0,0,dout)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    cache = {}
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        mean = x.mean(axis=0) # Array D
        var = x.var(axis=0) # Array D
        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * mean
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * var
        out = (x - mean) / np.sqrt(var + eps) * gamma + beta # Change mean and std of x to beta and gamma
        cache = (mean, var, gamma, eps, x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        out = (x - running_mean) / np.sqrt(running_var + eps) * gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    #dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    (mean, var, gamma, eps, x) = cache
    
    # out = (x - mean) / np.sqrt(var + eps) * gamma + beta
    # da = dL/da ~ dL/dout dout/da       dL/dout = dout (N,D)  General formula
    # dout/gamma = (x - mean) / np.sqrt(var + eps) # (N,D)
    # dout/dbeta = 1
    
    vareps = var + eps
    sqrtVar = np.sqrt(vareps)
    doutdgamma = (x - mean) / sqrtVar
    dgamma = np.sum(dout * doutdgamma,axis=0)
    dbeta = np.sum(dout,axis=0)
    
    # split dL/dx[n,d] into 2 partial derivatives that we will sum
    
    # dL/dout dot dout/d(x-mean) dot d(x-mean)/dx[n,d] = sum( dout[n1,d1] * gamma/sqrtVar[n1,d1] * (delta(n1,n)-1/N) delta(d1,d) ,n1 d1)
    # = dout[n,d] * gamma/sqrtVar[n,d] - sum(dout[n1,d] * gamma/sqrtVar[n1,d],n1)/N
    
    # dVar[d1]/dx[n,d] = d(sum((x[n2,d1]-mean[d1])^2,n2))/dx[n,d]/N = sum( d((x[n2,d1]-mean[d1])^2)/dx[n,d] ,n2)/N
    # = 2*sum( (x[n2,d1]-mean[d1])*(delta(n2,n)-1/N) delta(d1,d) ,n2)/N
    
    # dL/dout dot dout/d(1 / sqrtVar) dot d(1 / sqrtVar)/dx[n,d] = 
    # sum( dout[n1,d1] * gamma*(x-mean)[n1,d1] * sqrtVar[d1]^-3*-0.5*dVar[d1]/dx[n,d]    ,n1 d1)
    # = -sum( dout[n1,d] * gamma*(x-mean)[n1,d] * sqrtVar[d]^-3 * ((x[n,d]-mean[d]) - sum((x[n2,d]-mean[d]),n2)/N )/N    ,n1)
    
    N = x.shape[0]
    
    #dx1 = dout * gamma/sqrtVar - np.sum(dout * gamma/sqrtVar, axis=0)/N
    #dx2 = -np.sum(dout * gamma*(x-mean),axis=0) * sqrtVar^-3* ((x-mean) - np.sum((x-mean),axis=0)/N) /N
    
    # Simplify
    tmp = dout * gamma/sqrtVar
    dx1 = tmp - np.mean(tmp,axis=0)
    tmp2 = x-mean
    # dx2 = -dgamma/N*gamma*sqrtVar^-2 * (tmp2 - np.mean(tmp2,axis=0))
    dx2 = -dgamma/N*gamma/vareps * (tmp2 - np.mean(tmp2,axis=0))
    
    dx = dx1 + dx2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p)/p
        out = x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # out = x*mask
        # da = dL/da ~ dL/dout dout/da       dL/dout = dout (N,D)  General formula
        # dout/dx = mask
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    H = x.shape[2]
    HH = w.shape[2]
    W = x.shape[3]
    WW = w.shape[3]
    pad = conv_param['pad']
    stride = conv_param['stride']
    Hprime = int(1 + (H + 2 * pad - HH) / stride)
    Wprime = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros(shape=[x.shape[0],w.shape[0],Hprime,Wprime])
    x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)) ,'constant') # zero padding
    for hprime in range(Hprime):
        for wprime in range(Wprime):
            h1 = hprime * stride
            w1 = wprime * stride
            h2 = h1 + HH
            w2 = w1 + WW
            out[:,:,hprime,wprime] = np.sum(w[np.newaxis,:,:,:,:]*x[:,np.newaxis,:,h1:h2,w1:w2],axis=(2,3,4))
            
    # Add bias
    out += b[np.newaxis,:,np.newaxis,np.newaxis]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param, HH, WW, Hprime, Wprime)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    #dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w, b, conv_param, HH, WW, Hprime, Wprime) = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    # out = np.sum(w[np.newaxis,:,:,:,:]*x[:,np.newaxis,:,h1:h2,w1:w2],axis=(2,3,4)) + b
    # da = dL/da ~ dL/dout dout/da       dL/dout = dout (N, F, H', W')  General formula
    
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape) 
    for hprime in range(Hprime):
        for wprime in range(Wprime):
            h1 = hprime * stride
            w1 = wprime * stride
            h2 = h1 + HH
            w2 = w1 + WW
            # out[:,:,hprime,wprime] = np.sum(w[np.newaxis,:,:,:,:]*x[:,np.newaxis,:,h1:h2,w1:w2],axis=(2,3,4))
            
            douttmp = dout[:,:,hprime,wprime] # [:,:,np.newaxis,np.newaxis,np.newaxis] # 3 newaxis are for C H W
            
            dx[:,:,h1:h2,w1:w2] += np.sum(w[np.newaxis,:,:,:,:]*douttmp[:,:,np.newaxis,np.newaxis,np.newaxis],axis=1)
            # (N, C, H, W)                 (F, C, HH, WW)       (N, F, H', W')                    sum over F
            
            dw +=            np.sum(x[:,np.newaxis,:,h1:h2,w1:w2]*douttmp[:,:,np.newaxis,np.newaxis,np.newaxis],axis=0)
            # (F, C, HH, WW)        (N, C, H, W)                (N, F, H', W')                    sum over N
    
    dx = dx[:,:,pad:(dx.shape[2]-pad),pad:(dx.shape[3]-pad)] # Remove zero padding
    db = np.sum(dout,axis=(0,2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    #out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    H = x.shape[2]
    HH = pool_param['pool_height']
    W = x.shape[3]
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    Hprime = int(1 + (H - HH) / stride)
    Wprime = int(1 + (W - WW) / stride)
    out = np.zeros(shape=[x.shape[0],x.shape[1],Hprime,Wprime])
    for hprime in range(Hprime):
        for wprime in range(Wprime):
            h1 = hprime * stride
            w1 = wprime * stride
            h2 = h1 + HH
            w2 = w1 + WW
            out[:,:,hprime,wprime] = np.max(x[:,:,h1:h2,w1:w2],axis=(2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, Hprime, Wprime, out)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    (x, pool_param, Hprime, Wprime, out) = cache
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    
    # out = np.sum(w[np.newaxis,:,:,:,:]*x[:,np.newaxis,:,h1:h2,w1:w2],axis=(2,3,4)) + b
    # da = dL/da ~ dL/dout dout/da       dL/dout = dout (N, F, H', W')  General formula
    
    dx = np.zeros(x.shape)
    for hprime in range(Hprime):
        for wprime in range(Wprime):
            h1 = hprime * stride
            w1 = wprime * stride
            h2 = h1 + HH
            w2 = w1 + WW
            # out[:,:,hprime,wprime] = np.max(x[:,:,h1:h2,w1:w2],axis=(2,3))
            mask = np.equal(x[:,:,h1:h2,w1:w2],out[:,:,hprime,wprime][:,:,np.newaxis,np.newaxis])
            mask = mask / np.sum(mask,axis=(2,3))[:,:,np.newaxis,np.newaxis] # Normalize
            douttmp = dout[:,:,hprime,wprime] [:,:,np.newaxis,np.newaxis]
            dx[:,:,h1:h2,w1:w2] +=  mask * douttmp
            # dx only for the one that pass np.max
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    C = x.shape[1]
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

    if mode == 'train':
        mean = x.mean(axis=(0,2,3)) # N, C, H, W -> C
        var = x.var(axis=(0,2,3))  # N, C, H, W -> C
        bn_param['running_mean'] = momentum * running_mean + (1 - momentum) * mean
        bn_param['running_var'] = momentum * running_var + (1 - momentum) * var
        out = (x - mean[:,np.newaxis,np.newaxis]) / np.sqrt(var[:,np.newaxis,np.newaxis] + eps) * gamma[:,np.newaxis,np.newaxis] + beta[:,np.newaxis,np.newaxis] 
        # Change mean and std of x to beta and gamma
        cache = (mean, var, gamma, eps, x)
    elif mode == 'test':
        out = (x - running_mean[:,np.newaxis,np.newaxis]) / np.sqrt(running_var[:,np.newaxis,np.newaxis] + eps) * gamma[:,np.newaxis,np.newaxis] + beta[:,np.newaxis,np.newaxis]
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    
    # It's the same as before, but every sum over 0 becomes a sum over (0,2,3) and N becomes x.shape[0]*x.shape[2]*x.shape[3]
    
    (mean, var, gamma, eps, x) = cache
    
    mean = mean[:,np.newaxis,np.newaxis]
    gamma = gamma[:,np.newaxis,np.newaxis]
    vareps = var[:,np.newaxis,np.newaxis] + eps
    sqrtVar = np.sqrt(vareps)
    doutdgamma = (x - mean) / sqrtVar
    dgamma = np.sum(dout * doutdgamma,axis=(0,2,3))
    dbeta = np.sum(dout,axis=(0,2,3))
    
    N = x.shape[0]*x.shape[2]*x.shape[3]
    tmp = dout * gamma/sqrtVar
    dx1 = tmp - np.mean(tmp,axis=(0,2,3))[:,np.newaxis,np.newaxis]
    tmp2 = x-mean
    dx2 = -dgamma[:,np.newaxis,np.newaxis]/N*gamma/vareps * (tmp2 - np.mean(tmp2,axis=(0,2,3))[:,np.newaxis,np.newaxis])
    
    dx = dx1 + dx2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
