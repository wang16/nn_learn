import numpy as np
import math
from layer import Layer

WIDTH=0
HEIGHT=1
CHANNEL=2
LEFT=0
TOP=1

class MaxPooling(Layer):
  def _calcWindows(self):
    self._windowNum = [
      int(math.ceil((self._inputInfo[i] + 0.0)/self._poolInfo[i]))
      for i in range(2)]
    self._paddings = [
      (self._inputInfo[i] % self._poolInfo[i]) / 2
      for i in range(2)]

  def _setupIOMapping(self, n):
    self._TransformMapping = np.zeros(
        (self._windowNum[0] * self._windowNum[1] * self._inputInfo[CHANNEL],
         self._poolInfo[0] * self._poolInfo[1]), dtype=np.int32)
    self._Transform = np.zeros(
        (self._windowNum[0] * self._windowNum[1] * self._inputInfo[CHANNEL], n,
         self._poolInfo[0] * self._poolInfo[1]))
    self._ArgmaxCache = np.zeros(
        (self._windowNum[0] * self._windowNum[1] * self._inputInfo[CHANNEL],
         n), dtype=np.int32)
    self._TransformMapping.fill(-1)
    windowSize = self._windowNum[0] * self._windowNum[1]
    rawSize = self._inputInfo[WIDTH] * self._inputInfo[HEIGHT]
    for j in range(self._windowNum[1]):
      topInRawInput = -self._paddings[TOP] + j * self._poolInfo[1]
      for i in range(self._windowNum[0]):
        leftInRawInput = -self._paddings[LEFT] + i * self._poolInfo[0]
        windowIdx = j*self._windowNum[0] + i
        for h in range(self._poolInfo[HEIGHT]):
          y = topInRawInput + h
          if (y < 0 or y >= self._inputInfo[HEIGHT]):
            continue
          for w in range(self._poolInfo[WIDTH]):
            x = leftInRawInput + w
            if (x < 0 or x >= self._inputInfo[WIDTH]):
              continue
            idxInPool = h*self._poolInfo[WIDTH] + w
            idxInRaw = y * self._inputInfo[WIDTH] + x
            for c in range(self._inputInfo[CHANNEL]):
              self._TransformMapping[c*windowSize + windowIdx][idxInPool] = c*rawSize + idxInRaw

  # Normally, if previous layer is conv, the input channel of pooling is actually the depth of conv.
  def __init__(self, name, pWidth, pHeight, inputWidth, inputHeight, channel, prev):
    self._poolInfo = [
      pWidth,
      pHeight
    ]
    self._inputInfo = [
      inputWidth, inputHeight, channel
    ]
    self._calcWindows()
    windowSize = self._windowNum[0] * self._windowNum[1]
    Layer.__init__(self, name, windowSize * self._inputInfo[CHANNEL], prev)
    assert prev != None
    assert prev._size == inputWidth*inputHeight*channel

  def setBatchSize(self, n):
    Layer.setBatchSize(self, n)
    self._setupIOMapping(n)

  def calcValue(self):
    self._Transform.fill(-1)
    for n in range(self._batchSize):
      for i in range(self._TransformMapping.shape[0]):
        for j in range(self._TransformMapping.shape[1]):
          if self._TransformMapping[(i,j)] >= 0:
            self._Transform[(i,n,j)] = self._prev._output[(self._TransformMapping[(i,j)], n)]
    np.max(self._Transform, 2, self._output)
    np.argmax(self._Transform, 2, self._ArgmaxCache)

  def calcDeri(self):
    self._dLdInput.fill(0)
    for n in range(self._batchSize):
      for i in range(self._TransformMapping.shape[0]):
        self._dLdInput[(self._TransformMapping[i][self._ArgmaxCache[(i,n)]], n)] = self._dLdOutput[(i, n)]

