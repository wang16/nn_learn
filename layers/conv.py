import numpy as np
import math
import datetime
from layer import Layer
from input import InputLayer

WIDTH=0
HEIGHT=1
CHANNEL=2
DEPTH=2
STRIDE=3
LEFT=0
TOP=1

def _mapping(input, map, output):
  inputDimen = len(input.shape)
  assert output.shape == map.shape[:-1]
  assert map.shape[-1] == inputDimen
  flat = output.flat
  for i in range(output.size):
    coords = flat.coords
    output[coords] = input[tuple(map[coords])]
    flat.next()

def _reverse_mapping(output, map, input):
  inputDimen = len(input.shape)
  assert output.shape == map.shape[:-1]
  assert map.shape[-1] == inputDimen
  flat = output.flat
  for i in range(output.size):
    coords = flat.coords
    input[tuple(map[coords])] += output[coords]
    flat.next()

class ConvLayer(Layer):
  def _calcWindows(self):
    self._windowNum = [
      int(math.ceil((self._inputInfo[i] - self._kernalInfo[i] + 0.0)/self._kernalInfo[3])) + 1
      for i in range(2)]
    self._paddings = [
      ((self._windowNum[i] - 1) * self._kernalInfo[STRIDE] + self._kernalInfo[i] - self._inputInfo[i]) / 2
      for i in range(2)]

  def _setupIOMapping(self, n):
    self._inputTransfom = np.zeros(
        (self._windowNum[0] * self._windowNum[1] * n,
         self._kernalInfo[WIDTH] * self._kernalInfo[HEIGHT] * self._inputInfo[CHANNEL]))
    self._inputMapping = np.zeros(
        (self._windowNum[0] * self._windowNum[1] * n,
         self._kernalInfo[WIDTH] * self._kernalInfo[HEIGHT] * self._inputInfo[CHANNEL],
         2),
        dtype=np.int32)
    self._inputMapping.fill(-1)
    windowSize = self._windowNum[0] * self._windowNum[1]
    rawSize = self._inputInfo[WIDTH] * self._inputInfo[HEIGHT]
    kernalSize = self._kernalInfo[WIDTH] * self._kernalInfo[HEIGHT]
    for j in range(self._windowNum[1]):
      topInRawInput = -self._paddings[TOP] + j * self._kernalInfo[STRIDE]
      for i in range(self._windowNum[0]):
        leftInRawInput = -self._paddings[LEFT] + i * self._kernalInfo[STRIDE]
        windowIdx = j*self._windowNum[0] + i
        for h in range(self._kernalInfo[HEIGHT]):
          y = topInRawInput + h
          if (y < 0 or y >= self._inputInfo[HEIGHT]):
            continue
          for w in range(self._kernalInfo[WIDTH]):
            x = leftInRawInput + w
            if (x < 0 or x >= self._inputInfo[WIDTH]):
              continue
            idxInKernal = h*self._kernalInfo[WIDTH] + w
            idxInRaw = y * self._inputInfo[WIDTH] + x
            for c in range(self._inputInfo[DEPTH]):
              k = c * kernalSize + idxInKernal
              for s in range(n):
                self._inputMapping[s * windowSize + windowIdx][k][0] = c * rawSize + idxInRaw
                self._inputMapping[s * windowSize + windowIdx][k][1] = s
    self._outputMapping = np.zeros((windowSize * self._kernalInfo[DEPTH], n ,2), dtype=np.int32)
    for s in range(n):
      for d in range(self._kernalInfo[DEPTH]):
        for i in range(windowSize):
          self._outputMapping[d * windowSize + i][s][0] = s * windowSize + i
          self._outputMapping[d * windowSize + i][s][1] = d
    self._outputTransform = np.zeros((self._windowNum[0]*self._windowNum[1]*n, self._kernalInfo[DEPTH]))
    self._dLdOutputTransform = np.zeros(self._outputTransform.shape)
    self._dLdInputTransform = np.zeros(self._inputTransfom.shape)

  def __init__(self, name, depth, kWidth, kHeight, stride, prev, inputWidth, inputHeight, inputChannel):
    self._kernalInfo = [
      kWidth,
      kHeight,
      depth,
      stride
    ]
    self._inputInfo = [
      inputWidth, inputHeight, inputChannel
    ]
    self._calcWindows()
    assert prev != None
    assert prev._size == inputWidth*inputHeight*inputChannel
    Layer.__init__(self, name, self._windowNum[0] * self._windowNum[1] * self._kernalInfo[DEPTH], prev)
    self._kernal = np.random.normal(
      size=(self._kernalInfo[WIDTH]*self._kernalInfo[HEIGHT]*self._inputInfo[CHANNEL],
            self._kernalInfo[DEPTH]))
    self._bias = np.random.normal(size=(1, self._kernalInfo[DEPTH]))
    self._dLdKernal = np.zeros(self._kernal.shape)
    self._dLdb = np.zeros(self._bias.shape)

  def setBatchSize(self, n):
    Layer.setBatchSize(self, n)
    self._setupIOMapping(n)

  def calcValue(self):
    _mapping(self._prev._output, self._inputMapping, self._inputTransfom)
    np.matmul(self._inputTransfom, self._kernal, self._outputTransform)
    np.add(self._outputTransform, self._bias, self._outputTransform)
    _mapping(self._outputTransform, self._outputMapping, self._output)

  def calcDeri(self):
    _reverse_mapping(self._dLdOutput, self._outputMapping, self._dLdOutputTransform)
    np.matmul(self._kernal.transpose(), self._dLdOutputTransform, self._dLdInputTransform)
    _reverse_mapping(self._dLdInputTransform, self._inputMapping, self._dLdInput)
    np.matmul(self._dLdOutputTransform, self._inputTransfom.transpose(), self._dLdKernal)
    self._dLdb = np.sum(self._dLdOutputTransform, (0)).reshape((1, self._kernalInfo[DEPTH]))

  def updateParam(self, rate):
    #Logger.verbose("UPDATE", "update param: %s\n%s" % (self._name, self._dLdw))
    np.add(self._kernal, self._dLdKernal * -rate, self._kernal)
    np.add(self._bias, self._dLdb * -rate, self._bias)

  def getDerivative(self, idx):
    if idx < self._kernal.size:
      return self._kernal[(idx / self._kernal.shape[1], idx % self._kernal.shape[1])]
    bidx = idx - self._kernal.size
    if bidx < self._bias.size:
      return self._dLdb[(0, bidx)]
    return None

  def adjustParam(self, idx, delta):
    if idx < self._kernal.size:
      self._kernal[(idx / self._kernal.shape[1], idx % self._kernal.shape[1])] += delta
      return
    bidx = idx - self._kernal.size
    if bidx < self._bias.size:
      self._bias[(0, bidx)] += delta
      return

if __name__ == "__main__":
  test = ConvLayer("conv", 4, 2, 2, 1, InputLayer(5*5*3), 5, 5, 3)
  test.setBatchSize(2)
  print test._windowNum
  print test._paddings
  print test._inputMapping
  print test._outputMapping