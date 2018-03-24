import numpy as np
import math
from layer import Layer

class FCLayer(Layer):
  def __init__(self, name, size, prev):
    Layer.__init__(self, name, size, prev)
    assert prev != None
    # randomly initiate weight and bias
    self._w = np.random.uniform(0, 2.0 / prev._size, size=(size, prev._size))
    self._b = np.zeros((size, 1))
    self._dLdw = np.zeros((size, prev._size))
    self._dLdb = np.zeros((size, 1))
    self._Vdw = np.zeros((size, prev._size))
    self._Vdb = np.zeros((size, 1))
    self._Sdw = np.zeros((size, prev._size))
    self._Sdb = np.zeros((size, 1))

  def calcValue(self):
    input = self._prev._output
    np.matmul(self._w, input, self._output)
    np.add(self._output, self._b, self._output)

  def calcDeri(self):
    np.matmul(self._w.transpose(), self._dLdOutput, self._dLdInput)
    np.matmul(self._dLdOutput, self._prev._output.transpose(), self._dLdw)
    self._dLdb = np.sum(self._dLdOutput, (1)).reshape((self._size, 1))
    self._dLdw /= self._batchSize
    self._dLdb /= self._batchSize

  def updateParam(self, rate, lambd, betaS=0.999, betaV=0.9):
    #Logger.verbose("UPDATE", "update param: %s\n%s" % (self._name, self._dLdw))
    # epsilon = 0.00000001
    # self._Vdw = (betaV * self._Vdw + (1-betaV) * self._dLdw) / (1-math.pow(betaV, self._iter))
    # self._Vdb = (betaV * self._Vdb + (1-betaV) * self._dLdb) / (1-math.pow(betaV, self._iter))
    # self._Sdw = (betaS * self._Sdw + (1-betaS) * np.multiply(self._dLdw, self._dLdw)) / (1-math.pow(betaS, self._iter))
    # self._Sdb = (betaS * self._Sdb + (1-betaS) * np.multiply(self._dLdb, self._dLdb)) / (1-math.pow(betaS, self._iter))
    # adamW = self._Vdw / (np.sqrt(self._Sdw) + epsilon)
    # adamB = self._Vdb / (np.sqrt(self._Sdb) + epsilon)
    np.add(self._w, (self._dLdw + self._w * lambd) * -rate, self._w)
    np.add(self._b, (self._dLdb + self._b * lambd) * -rate, self._b)

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
