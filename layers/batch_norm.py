import numpy as np
import math
from layer import Layer

EPI = 0.00000001

class BatchNorm(Layer):
  def __init__(self, name, prev):
    assert prev != None
    Layer.__init__(self, name, prev._size, prev)
    self._average = np.zeros((prev._size, 1))
    self._variance = np.zeros((prev._size, 1))
    self._iter = 0
    self._isTraining = False
    self._beta = np.zeros((prev._size, 1))
    self._beta.fill(1)
    self._gamma = np.zeros((prev._size, 1))
    self._gamma.fill(1)

  def setIsTraining(self, isTrain):
    self._isTraining = isTrain

  def calcValue(self):
    if self._isTraining:
      average = self._average
      variance = self._variance
    else:
      self._iter += 1
      average = np.sum(self._prev._output, 1, keepdims=True)/self._batchSize
      variance = np.var(self._prev._output, 1, keepdims=True) + EPI
      self._average = (self._average * 0.9 + average * 0.1) / (1 - math.pow(0.9, self._iter))
      self._variance = (self._variance * 0.9 + variance * 0.1) / (1 - math.pow(0.9, self._iter))
    self._std  = (self._prev._output - average) / np.sqrt(variance)
    np.add(self._std * self._gamma, self._beta, self._output)

  def calcDeri(self):
    average = np.sum(self._prev._output, 1, keepdims=True)/self._batchSize
    variance = np.var(self._prev._output, 1, keepdims=True) + EPI
    dLdstd = self._dLdOutput * self._gamma
    varSqrtRecip = np.reciprocal(np.sqrt(variance + EPI))
    dLdAvg = np.sum(dLdstd * -varSqrtRecip, 1, keepdims=True)
    dLdVar = np.sum(-dLdstd * (self._prev._output-average) * np.power(variance+EPI, -1.5) / 2, 1, keepdims=True)
    np.add(dLdstd * varSqrtRecip + dLdVar * 2 * (self._prev._output - average) / self._batchSize,
           dLdAvg / self._batchSize, self._dLdInput)
    self._dLdGamma = np.sum(self._dLdOutput * self._std, 1, keepdims=True)
    self._dLdBeta = np.sum(self._dLdOutput, 1, keepdims=True)

  def updateParam(self, rate, lambd, betaS=0.999, betaV=0.9):
    np.add(self._gamma, (self._dLdGamma + self._gamma * lambd) * -rate, self._gamma)
    np.add(self._beta, (self._dLdBeta + self._beta * lambd) * -rate, self._beta)