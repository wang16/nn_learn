import numpy as np
from layer import Layer

def relu(result, alpha, output=None):
  return np.maximum(result, alpha*result, output)

class LeakyReLU(Layer):
  def __init__(self, name, alpha, prev):
    assert prev != None
    Layer.__init__(self, name, prev._size, prev)
    self._alpha = alpha

  def calcValue(self):
    input = self._prev._output
    relu(input, self._alpha, self._output)

  # for softmax, dLdOutput accepted is actually batch labels,
  # although it's not technically real derivative, but the programing is much easier this way.
  def calcDeri(self):
    self._dLdInput[self._prev._output <= 0] = self._alpha
    self._dLdInput[self._prev._output > 0] = 1
    np.multiply(self._dLdInput, self._dLdOutput, self._dLdInput)