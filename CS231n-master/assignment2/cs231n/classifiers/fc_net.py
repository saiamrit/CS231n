from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        #pass
        
        W1 = weight_scale * np.random.randn(input_dim,hidden_dim)
        b1 = np.zeros((1,hidden_dim))
        W2 = weight_scale * np.random.randn(hidden_dim,num_classes)
        b2 = np.zeros((1,num_classes))
       
        self.params = {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        
        
        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']
        
        D=1
        N = X.shape[0]
        for i in range(1,len(X.shape)):
            D *= X.shape[i]
    
        X = np.reshape(X,[N,D])
        hidden_layer = np.maximum(0,np.dot(X,W1)+b1)
        scores = np.dot(hidden_layer,W2) + b2
        #scores = np.exp(y_)
        #scores = scores / np.sum(scores,axis=1,keepdims=True)
        #print(y_)
        
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        
        '''
        data_loss = -np.log(scores[range(X.shape[0]),y])
        data_loss = np.sum(data_loss) / X.shape[0]
        reg_loss = 0.5*self.reg*np.sum(W1*W1) + 0.5*self.reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        '''
        
        data_loss , dscores = softmax_loss(scores,y)
        reg_loss = 0.5*self.reg*np.sum(W1*W1) + 0.5*self.reg*np.sum(W2*W2)
        loss = data_loss + reg_loss
        
        #dy_ = scores 
        #dy_ -= 1
        #dy_ /= X.shape[0]
        dW2 = np.dot(hidden_layer.T,dscores) + self.reg*W2
        dhidden_layer = np.dot(dscores,W2.T)
        dhidden_layer[hidden_layer <= 0] = 0
        dW1 = np.dot(X.T,dhidden_layer) + self.reg*W1
        db2 = np.sum(dscores,axis=0)
        db1 = np.sum(dhidden_layer,axis=0)
        
        
        grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2}
        
        '''
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * np.sum(W1**2)
        reg_loss += 0.5 * self.reg * np.sum(W2**2)
        loss = data_loss + reg_loss

        # Backpropagaton
        grads = {}
        # Backprop into second layer
        dx1, dW2, db2 = affine_backward(dscores, cache_scores)
        dW2 += self.reg * W2

        # Backprop into first layer
        dx, dW1, db1 = affine_relu_backward(
            dx1, cache_hidden_layer)
        dW1 += self.reg * W1
        
        grads = {'W1':dW1,'b1':db1,'W2':dW2,'b2':db2}
        '''
        
        
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        
        dim_list = []
        dim_list.append(input_dim)
        for i in range(len(hidden_dims)):
            dim_list.append(hidden_dims[i])
        W = 'W'
        b = 'b'
        gamma = 'gamma'
        beta = 'beta'
            
        
        for i in range(self.num_layers-1):
            w = weight_scale*np.random.randn(dim_list[i],dim_list[i+1])
            W+=str(i)
            self.params[W] = w
            b1 = np.zeros((1,dim_list[i+1]))
            b+=str(i)
            self.params[b] = b1
            if self.use_batchnorm:
                gamma1 = np.ones(dim_list[i+1])
                gamma+=str(i)
                self.params[gamma] = gamma1
                beta1 = np.zeros(dim_list[i+1])
                beta+=str(i)
                self.params[beta] = beta1     
                gamma = 'gamma'
                beta = 'beta'
                pass
            W = 'W'
            b = 'b'
        
        
        w = weight_scale*np.random.randn(dim_list[i+1],num_classes)
        W+=str(i+1)
        self.params[W] = w
        b1 = np.zeros((1,num_classes))
        b+=str(i+1)
        self.params[b] = b1
        
       
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        W = 'W'
        b = 'b'
        gamma = 'gamma'
        beta = 'beta'
        dropout_param = self.dropout_param
        L = self.num_layers
        bn_params = self.bn_params
        Input = []
        Input.append(X)
        out_dict = {}
        cache = {}
        for i in range(L-1):
            out = []
            W = 'W'
            b = 'b'
            gamma = 'gamma'
            beta = 'beta'
            cache_affine = 'cache_affine'
            cache_relu = 'cache_relu'
            cache_batchnorm = 'cache_batchnorm'
            cache_dropout = 'cache_dropout'
            cache_affine+=str(i)
            W+=str(i)
            b+=str(i)
            gamma+=str(i)
            beta+=str(i)
            h1,cache[cache_affine] = affine_forward(Input[i],self.params[W],self.params[b])
            if self.use_batchnorm:
                cache_batchnorm+=str(i)
                h2,cache[cache_batchnorm] = batchnorm_forward(h1,self.params[gamma],self.params[beta],self.bn_params[i])
                cache_relu+=str(i)
                h3,cache[cache_relu] = relu_forward(h2)
                if self.use_dropout:
                    cache_dropout+=str(i)
                    h4,cache[cache_dropout] = dropout_forward(h3,dropout_param)
                    Input.append(h4)
                else:
                    Input.append(h3)
                             
            else:
                cache_relu +=str(i)
                h2,cache[cache_relu] = relu_forward(h1)
                if self.use_dropout:
                    cache_dropout+=str(i)
                    h3,cache[cache_dropout] = dropout_forward(h2,dropout_param)
                    Input.append(h3)
                    
                else:
                    Input.append(h2)
            
        W = 'W'
        b = 'b'
        W+=str(L-1)
        b+=str(L-1)
        cache_affine = 'cache_affine'
        cache_affine+=str(L-1)
        scores,cache[cache_affine] = affine_forward(Input[L-1],self.params[W],self.params[b])
            
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        data_loss,dscores  = softmax_loss(scores,y)
        reg_loss = 0
        for i in range(L):
            W = 'W'
            W+=str(i)
            reg_loss += 0.5*self.reg*np.sum(self.params[W]*self.params[W])
        loss = data_loss + reg_loss
        
        
        dW = {}
        db = {}
        dgamma = {}
        dbeta = {}

        
        W = 'W'
        b = 'b'
        gamma = 'gamma'
        beta = 'beta'
        
        cache_affine = 'cache_affine'
        cache_relu = 'cache_relu'
        cache_batchnorm = 'cache_batchnorm'
        cache_dropout = 'cache_dropout'
        
        cache_affine+=str(L-1)
        W +=str(L-1)
        b +=str(L-1)
       
    
    
        dout,dW[W],db[b] = affine_backward(dscores,cache[cache_affine])
        dW[W] += self.reg * self.params[W]
       
        for i in range(L-2,-1,-1):
            
            cache_affine = 'cache_affine'
            cache_relu = 'cache_relu'
            cache_batchnorm = 'cache_batchnorm'
            cache_dropout = 'cache_dropout'
            
            W = 'W'
            b = 'b'
            gamma = 'gamma'
            beta = 'beta'
            
            
            if self.use_batchnorm:
                if self.use_dropout:
                    cache_dropout+=str(i)
                    dh1 = dropout_backward(dout,cache[cache_dropout])
                    cache_relu+=str(i)
                    dh2 = relu_backward(dh1,cache[cache_relu])
                    cache_batchnorm+=str(i)
                    gamma+=str(i)
                    beta+=str(i)
                    dh3,dgamma[gamma],dbeta[beta] = batchnorm_backward(dh2,cache[cache_batchnorm])
                    W+=str(i)
                    b+=str(i)
                    cache_affine+=str(i)
                    dout,dW[W],db[b] = affine_backward(dh3,cache[cache_affine])
                    dW[W] += self.reg * self.params[W]
                    
                else:
                    
                    cache_relu+=str(i)
                    dh1 = relu_backward(dout,cache[cache_relu])
                    cache_batchnorm+=str(i)
                    gamma+=str(i)
                    beta+=str(i)
                    dh2,dgamma[gamma],dbeta[beta] = batchnorm_backward(dh1,cache[cache_batchnorm])
                    W+=str(i)
                    b+=str(i)
                    cache_affine+=str(i)
                    dout,dW[W],db[b] = affine_backward(dh2,cache[cache_affine])
                    dW[W] += self.reg * self.params[W]
                    
            else:
                if self.use_dropout:
                    cache_dropout+=str(i)
                    dh1 = dropout_backward(dout,cache[cache_dropout])
                    cache_relu+=str(i)
                    dh2 = relu_backward(dh1,cache[cache_relu])
                    W+=str(i)
                    b+=str(i)
                    cache_affine+=str(i)
                    dout,dW[W],db[b] = affine_backward(dh2,cache[cache_affine])
                    dW[W] += self.reg * self.params[W]
                    
                    
                else:
                    
                    cache_relu+=str(i)
                    dh1 = relu_backward(dout,cache[cache_relu])
                    W+=str(i)
                    b+=str(i)
                    cache_affine+=str(i)
                    dout,dW[W],db[b] = affine_backward(dh1,cache[cache_affine])
                    dW[W] += self.reg * self.params[W]
                    
                
                    
                    
                    
        grads = dW.copy()
        grads.update(db)
        grads.update(dgamma)
        grads.update(dbeta)
                 
            
          
        
        
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
