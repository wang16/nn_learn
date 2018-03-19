import numpy as np
from layer import Layer

class FCLayer(Layer):
  def __init__(self, name, size, prev):
    Layer.__init__(self, name, size, prev)
    assert prev != None
    # randomly initiate weight and bias
    self._w = np.random.uniform(0, 2.0 / 3.0 /prev._size, size=(size, prev._size))
    self._b = np.random.uniform(0, 0.5, size=(size, 1))
    self._dLdw = np.zeros((size, prev._size))
    self._dLdb = np.zeros((size, 1))

  def calcValue(self):
    input = self._prev._output
    np.matmul(self._w, input, self._output)
    np.add(self._output, self._b, self._output)

  def calcDeri(self):
    np.matmul(self._w.transpose(), self._dLdOutput, self._dLdInput)
    np.matmul(self._dLdOutput, self._prev._output.transpose(), self._dLdw)
    self._dLdb = np.sum(self._dLdOutput, (1)).reshape((self._size, 1))

  def updateParam(self, rate):
    #Logger.verbose("UPDATE", "update param: %s\n%s" % (self._name, self._dLdw))
    np.add(self._w, self._dLdw * -rate, self._w)
    np.add(self._b, self._dLdb * -rate, self._b)

  def getDerivative(self, idx):
    if idx < self._w.size:
      return self._dLdw[(idx / self._w.shape[1], idx % self._w.shape[1])]
    bidx = idx - self._w.size
    if bidx < self._b.size:
      return self._dLdb[(bidx,0)]
    return None

  def adjustParam(self, idx, delta):
    if idx < self._w.size:
      self._w[(idx / self._w.shape[1], idx % self._w.shape[1])] += delta
      return
    bidx = idx - self._w.size
    if bidx < self._b.size:
      self._b[(bidx,0)] += delta
      return
