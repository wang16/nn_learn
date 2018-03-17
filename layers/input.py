import numpy as np
from layer import Layer

class InputLayer(Layer):
  def __init__(self, size):
    Layer.__init__(self, "input", size, None)

  def setData(self, samples, batchSize):
    self.setBatchSize(batchSize)
    np.divide(samples, 256.0, self._output)

  def backward(self, dLdOutput):
    pass