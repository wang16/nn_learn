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
    self._batchSize = 1

  def setBatchSize(self, n):
    self._batchSize = n
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

  def updateParam(self, rate, lambd):
    pass

  def getDerivative(self, idx):
    return None

  def adjustParam(self, idx, delta):
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

  def checkBackProp(self, delta, output, lossCalc):
    Logger.info("CHECK_BACKPROP", "check for %s" % self._name)
    idx = 0
    while(True):
      derivative = self.getDerivative(idx)
      if derivative == None:
        break
      self.adjustParam(idx, delta)
      self.forward()
      loss1 = lossCalc()
      self.adjustParam(idx, -2*delta)
      self.forward()
      loss2 = lossCalc()
      self.adjustParam(idx, delta)
      idx += 1
      output.append((derivative, (loss1-loss2) / (delta*2)))
    if self._next != None:
      self._next.checkBackProp(delta, output, lossCalc)
