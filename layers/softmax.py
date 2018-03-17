import numpy as np
from layer import Layer


def softmax(result, output=None):
  temp = np.exp(result - np.max(result, 0))
  return np.divide(temp, np.sum(temp, 0), output)

class SoftMax(Layer):
  def __init__(self, name, prev):
    assert prev != None
    Layer.__init__(self, name, prev._size, prev)

  def calcValue(self):
    input = self._prev._output
    softmax(input, self._output)

  # for softmax, dLdOutput accepted is actually batch labels,
  # although it's not technically real derivative, but the programing is much easier this way.
  def calcDeri(self):
    np.add(self._output, -self._dLdOutput, self._dLdInput)