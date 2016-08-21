import numpy as np
import matplotlib.pyplot as plt

class Layer(object):
    def compute(self, x):
        raise NotImplemented()

    # Returns a maybe-empty matrix that contains all the parameters of the layer.
    def params(self):
        return self._params
    
    def dcostdx(self, x, y, dcostdy):
        raise NotImplemented()
        
    def dcostdparams(self, x, y, dcostdy):
        return np.array([])
        
    def update_params(self, gradients):
        if len(gradients.shape) == 1:
            gradients.shape = self._params.shape
        if gradients.shape != self._params.shape:
            raise Exception("Gradients should be the same size as the parameters")
        self._params -= gradients

class Network(object):
    def __init__(self, layers, cost_function):
        self._layers = layers
        self._cost_function = cost_function
        
    def compute(self, x):
        for layer in self._layers:
            x = layer.compute(x)
            
        return x
    
    def compute_single_cost(self, x, y):
        y_ = self.compute(x)
        return self._cost_function.compute(y, y_)
    
    def compute_cost(self, X, y):
        y_ = self.compute(X)
        cost = self._cost_function.compute(y, y_)
        assert cost.shape[0] == len(y)
        return np.sum(cost) / len(y)

    def gradients(self, x, y):
      return self.backprop_gradients(x, y)
    
    def backprop_gradients(self, x, y):
        results = [x]
        for layer in self._layers:
            x = layer.compute(x)
            results.append(x)

        gradients = []
        
        dcostdy = self._cost_function.dcostdy(y, x)
        for layer in reversed(self._layers):
            gradients.append(layer.dcostdparams(results[-2], results[-1], dcostdy))
            dcostdy = layer.dcostdx(results[-2], results[-1], dcostdy)
            results.pop()
            
        gradients.reverse()
        return gradients
    
    def numeric_gradients(self, X, y):
        "This is wrong. I don't know how, but it is wrong."
        epsilon = 0.0000001
        gradients = []
        for layer in self._layers:
            params = layer.params().reshape(layer.params().size)
            layer_gradients = np.zeros(params.size)
            gradients.append(layer_gradients)
            for i in range(len(params)):
                params[i] += epsilon
                high = self.compute_cost(X, y)
                params[i] -= 2*epsilon
                low = self.compute_cost(X, y)
                params[i] += epsilon
                layer_gradients[i] = (high-low)/epsilon
        return gradients
    
    def update_params(self, gradients):
        for i, layer in enumerate(self._layers):
            layer.update_params(gradients[i])
                
    def __repr__(self):
        return '<Network: (layers=%s, cost_function=%s)>' % (str(self._layers), str(self._cost_function))

# Inputs and outputs are row vectors
class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        # +1 for the bias.
        params = 2 * (np.random.rand(input_size + 1, output_size) - 0.5) / np.sqrt(input_size)
        self._params = np.round(params, decimals=2)
        
    def dcostdx(self, x, y, dcostdy):
        return dcostdy.dot(self._params.T)[:, 1:]
    
    def dcostdparams(self, x, y, dcostdy):
        # Accept 1-d arrays
        if len(x.shape) < 2:
            x.shape = (1, x.shape[0])

        x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
        return x.T.dot(dcostdy)
    
    def compute(self, x):
        # Accept 1-d arrays
        if len(x.shape) < 2:
            x.shape = (1, x.shape[0])

        x = np.hstack((np.ones(shape=(x.shape[0], 1)), x))
        ret = x.dot(self._params)
        return ret
    
    def __repr__(self):
        return 'Linear(%s, %s)' % (self._params.shape[1]-1, self._params.shape[0])
    
from scipy.special import expit

class SigmoidLayer(Layer):
    def __init__(self):
        self._params = np.array([])
        
    def dcostdx(self, x, y, dcostdy):
        return dcostdy * (y * (1-y))
        
    def compute(self, x):
        return expit(x)
    
    def __repr__(self):
        return 'Sigmoid'
sigmoidLayer = SigmoidLayer()

class SoftmaxLayer(Layer):
  def __init__(self):
    self._params = np.array([])

  def dydx(self, x, y):
    ret = -y.T.dot(y)
    ret.flat[::ret.shape[0]+1] = y * (1 - y)
    return ret

  def dcostdx(self, x, y, dcostdy):
    return dcostdy.dot(self.dydx(x, y))

  def compute(self, x):
    e = np.exp(x)
    return e / np.sum(e)
softmaxLayer = SoftmaxLayer()

class LogisticCost(object):
    def dcostdy(self, y, y_):
        epsilon = np.finfo('float64').eps
        y_ = np.clip(y_, epsilon, 1.0 - epsilon)
        return ((y_ - y) / (y_ * (1-y_))) / y.shape[0]
        
    def compute(self, y, y_):
        "y is the actual value, y_ is the predicted value."
        epsilon = np.finfo('float64').eps
        y_ = np.clip(y_, epsilon, 1.0 - epsilon)
        cost = -y * np.log(y_) - (1-y) * np.log(1-y_)
        return np.sum(cost, axis=1)
    
    def __repr__(self):
        return 'LogisticCostFunction'
logisticCost = LogisticCost()


def NewNeuralNet(*args):
    layers = []
    for i in range(len(args)-1):
        layers.append(LinearLayer(args[i], args[i+1]))
        layers.append(sigmoidLayer)

    return Network(layers, logisticCost)


def StochasticGradientDescent(model, X, y, iterations=None, learning_rate=0.1, callback_period=None, callback=None):
  assert X.shape[0] == y.shape[0]
  if callback_period:
    assert callback is not None

  if not iterations:
    iterations = len(y) * 10

  for i in range(iterations):
    if callback_period and i % callback_period == 0:
      callback(i)
    sample_i = np.random.randint(X.shape[0])
    x_sample = X[sample_i]
    y_sample = y[sample_i]
    gradients = model.gradients(x_sample, y_sample)
    gradients[0] *= learning_rate
    model.update_params(gradients)
