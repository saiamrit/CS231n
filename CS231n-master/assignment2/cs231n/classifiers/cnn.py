from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        
        C,H,W = input_dim
        F = num_filters
        HH = filter_size
        WW = filter_size
        
        W1 = weight_scale*np.random.randn(F,C,HH,WW)
        b1 = np.zeros((F))
        W2 = weight_scale*np.random.randn(F*(H//2)*(W//2),hidden_dim)
        b2 = np.zeros((1,hidden_dim))
        W3 = weight_scale*np.random.randn(hidden_dim,10)
        b3 = np.zeros((1,10))
        
        self.params['W1'],self.params['W2'],self.params['W3'] = W1,W2,W3
        self.params['b1'],self.params['b2'],self.params['b3'] = b1,b2,b3
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        reg = self.reg
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        
        N,C,H,W = X.shape
  
        h1,cache_conv = conv_forward_naive(X,W1,b1,conv_param)
        h2,cache_relu1 = relu_forward(h1)
        h3,cache_pool = max_pool_forward_naive(h2,pool_param)
        N1,C1,H1,W1 = h3.shape
        h3_reshaped = np.reshape(h3,(N1,C1*H1*W1))
        h4,cache_affine1 = affine_forward(h3_reshaped,W2,b2)      
        h5,cache_relu2 = relu_forward(h4) 
        scores,cache_affine2 = affine_forward(h5,W3,b3)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        
        data_loss,dscores = softmax_loss(scores,y)
        reg_loss = 0.5*reg*np.sum(W1**2) + 0.5*reg*np.sum(W2**2) + 0.5*reg*np.sum(W3**2)
        loss = data_loss + reg_loss
        
        
        dh5,dW3,db3 = affine_backward(dscores,cache_affine2)
        dh4 = relu_backward(dh5,cache_relu2)
        dh3_reshaped,dW2,db2 = affine_backward(dh4,cache_affine1)
        dh3 = np.reshape(dh3_reshaped,(N1,C1,H1,W1))
        dh2 = max_pool_backward_naive(dh3,cache_pool)
        dh1 = relu_backward(dh2,cache_relu1)
        dX,dW1,db1 = conv_backward_naive(dh1,cache_conv)
        
        dW1 += reg*W1
        dW2 += reg*W2
        dW3 += reg*W3
        
        grads['W1'],grads['W2'],grads['W3'] = dW1,dW2,dW3
        grads['b1'],grads['b2'],grads['b3'] = db1,db2,db3
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
