import numpy as np

from logger import Logger

TAG = "Layer"
class Layer:
  def __init__(self, name, size, prev):
    self._name = name
    self._prev = prev
    self._next = None
    if prev != None:
      prev._next = self
    self._size = size
    self._output = None
    self._dLdInput = None

  def setBatchSize(self, n):
    self._output = np.zeros((self._size, n))
    self._dLdOutput = np.zeros((self._size, n))
    if self._prev != None:
      self._dLdInput = np.zeros((self._prev._size, n))
    if self._next != None:
      self._next.setBatchSize(n)

  def calcValue(self):
    pass

  def calcDeri(self):
    pass

  def updateParam(self, rate):
    pass

  def forward(self):
    self.calcValue()
    #Logger.verbose(TAG, "forward pass of %s:\nresult:\n%s\noutput:\n%s" %
    #               (self._name, self._result, self._output))
    if self._next != None:
      self._next.forward()

  def backward(self, dLdOutput):
    assert np.shape(dLdOutput)[0] == self._size
    np.add(dLdOutput, 0, self._dLdOutput)
    self.calcDeri()
    #Logger.verbose(TAG, "back propagation of %s:\ndl/dout:\n%s\ndl/dResult:\n%s" %
    #               (self._name, dLdOutput, self._dLdResult))
    if self._prev != None:
      self._prev.backward(self._dLdInput)