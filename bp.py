import numpy as np

from read_mnist import readMnistData

verbose = False

class Layer:
  def __init__(self, name, size, prev, activation):
    self._name = name
    self._prev = prev
    self._next = None
    if prev != None:
      prev._next = self
    self._size = size
    self._output = None
    self._dLdInput = None
    self._result = None
    self._act = activation

  def setBatchSize(self, n):
    self._output = np.zeros((self._size, n))
    self._result = np.zeros((self._size, n))
    self._dLdResult = np.zeros((self._size, n))
    if self._act != None:
      self._act.setBatchSize(n)
    if self._prev != None:
      self._dLdInput = np.zeros((self._prev._size, n))
    if self._next != None:
      self._next.setBatchSize(n)

  def calcValue(self):
    pass

  def calcDeri(self, dLdOutput):
    pass

  def updateParam(self, rate):
    pass

  def forward(self):
    self.calcValue()
    if verbose:
      print "forward pass of %s:" % self._name
      print "result:"
      print self._result
      print "output:"
      print self._output
    if self._next != None:
      self._next.forward()

  def backward(self, dLdOutput):
    assert np.shape(dLdOutput)[0] == self._size
    self.calcDeri(dLdOutput)
    if verbose:
      print "back propagation of %s:" % self._name
      print "dl/dout:"
      print dLdOutput
      print "dl/dResult:"
      print self._dLdResult
    if self._prev != None:
      self._prev.backward(self._dLdInput)

class InputLayer(Layer):
  def __init__(self, size):
    Layer.__init__(self, "input", size, None, None)

  def setData(self, samples, batchSize):
    self.setBatchSize(batchSize)
    np.divide(samples, 256.0, self._output)
    np.add(samples, 0, self._result)
  def backward(self, dLdOutput):
    pass

class FCLayer(Layer):
  def __init__(self, name, size, prev, activation):
    Layer.__init__(self, name, size, prev, activation)
    assert prev != None
    # randomly initiate weight and bias
    self._w = np.random.uniform(-1.0/prev._size, 1.0/prev._size, size=(size, prev._size))
    self._b = np.random.normal(size=(size, 1))
    self._dLdw = np.zeros((size, prev._size))
    self._dLdb = np.zeros((size, 1))

  def calcValue(self):
    input = self._prev._output
    np.matmul(self._w, input, self._result)
    np.add(self._result, self._b, self._result)
    if self._act != None:
      self._act.calc(self._result, self._output)
    else:
      np.add(self._result, 0, self._output)

  def calcDeri(self, dLdOutput):
    if self._act == None:
      np.add(dLdOutput, 0, self._dLdResult)
    else:
      self._act.deri(dLdOutput, self._dLdResult)
    np.matmul(self._w.transpose(), self._dLdResult, self._dLdInput)
    np.matmul(self._dLdResult, self._prev._output.transpose(), self._dLdw)
    self._dLdb = np.sum(self._dLdResult, (1)).reshape((self._size, 1))

  def updateParam(self, rate):
    if verbose:
      print "update param: %s" % self._name
      print self._dLdw
    np.add(self._w, self._dLdw * -rate, self._w)
    np.add(self._b, self._dLdb * -rate, self._b)

def relu(result, output=None):
  return np.maximum(result, 0, output)

def softmax(result, output=None):
  temp = np.exp(result - np.max(result, 0))
  return np.divide(temp, np.sum(temp, 0), output)

class Activation:
  def __init__(self, size):
    self._size = size
    self._input = None

  def setBatchSize(self, n):
    self._input = np.zeros((self._size, n))

  def calc(self, result, output):
    pass
  def deri(self, dLdOutput, dLdResult):
    pass

class ReLU(Activation):
  def calc(self, result, output):
    np.add(result, 0, self._input)
    relu(result, output)
  def deri(self, dLdOutput, dLdResult):
    dLdResult[self._input<=0] = 0
    dLdResult[self._input>0] = 1
    np.multiply(dLdResult, dLdOutput, dLdResult)


if __name__ == "__main__":
  output = 10
  input = 28*28
  # setup network graph
  IL = InputLayer(input)
  FC1 = FCLayer("fc1", 60, IL, ReLU(60))
  FC2 = FCLayer("fc2", 100, FC1, ReLU(100))
  FC3 = FCLayer("fc3", 30, FC2, ReLU(30))
  OL = FCLayer("output", output, FC3, None)

  train, test = readMnistData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 12)
  batchSize = 100
  epoch = 5
  learningRate = 0.003

  tempTestSize = 3000

  for i in range(epoch):
    progress = 0
    #for j in range(1):
    while progress + batchSize < len(train) - tempTestSize:
      batch = [train[i].pixels for i in range(progress, progress + batchSize)]
      batchlabel = [train[i].labelArray for i in range(progress, progress + batchSize)]
      IL.setData(np.array(batch).transpose(), batchSize)
      IL.forward()
      score = np.zeros((output, batchSize))
      #print OL._output.transpose()
      softmax(OL._output, score)
      if verbose:
        print "score:"
        print score.transpose()
      cross_entrophy = np.multiply(np.array(batchlabel).transpose(), np.log(score))
      if verbose:
        print cross_entrophy.transpose()
      loss = -np.sum(cross_entrophy, (0,1))/batchSize
      print "Loss: %f" % loss

      dLdOutput_OL = score - np.array(batchlabel).transpose()
      OL.backward(dLdOutput_OL)
      FC1.updateParam(learningRate)
      FC2.updateParam(learningRate)
      FC3.updateParam(learningRate)
      OL.updateParam(learningRate)

      predict = np.argmax(score, (0))
      labels = np.argmax(np.array(batchlabel).transpose(), (0))
      right = 0
      for i in range(batchSize):
        if predict[i] == labels[i]:
          right+=1
      print "Accuracy: %d/%d" % (right, batchSize)
      progress += batchSize
      #break

  batch = [train[i].pixels for i in range(len(train) - tempTestSize, len(train))]
  batchlabel = [train[i].labelArray for i in range(len(train) - tempTestSize, len(train))]
  IL.setData(np.array(batch).transpose(), tempTestSize)
  IL.forward()
  score = np.zeros((output, tempTestSize))
  softmax(OL._output, score)
  predict = np.argmax(score, (0))
  labels = np.argmax(np.array(batchlabel).transpose(), (0))
  right = 0
  for i in range(tempTestSize):
    if predict[i] == labels[i]:
      right+=1
  print "\n\nTemporary Test Accuracy: %d/%d" % (right, tempTestSize)