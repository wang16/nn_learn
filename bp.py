import numpy as np
import datetime
import math, sys

from logger import Logger
from read_mnist import readMnistData
from layers.input import InputLayer
from layers.fc import FCLayer
from layers.relu import ReLU
from layers.softmax import SoftMax


Logger.setLogLevel(Logger.INFO)



if __name__ == "__main__":
  output = 10
  input = 28 * 28
  # setup network graph
  layers = [600, 1000, 300]
  alpha = 0
  IL = InputLayer(input)
  FC1 = FCLayer("fc1", layers[0], IL)
  ACT1 = ReLU("relu1", FC1)
  FC2 = FCLayer("fc2", layers[1], ACT1)
  ACT2 = ReLU("relu2", FC2)
  FC3 = FCLayer("fc3", layers[2], ACT2)
  ACT3 = ReLU("relu3", FC3)
  OL = FCLayer("output", output, ACT3)
  SOFTMAX = SoftMax("softmax", OL)

  train, test = readMnistData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 12)
  #allTrainData = np.array([train[i].pixels for i in range(len(train))])
  #average = np.average(allTrainData, 0)
  #variance = np.var(allTrainData, axis=0) + 0.00000001
  batchSize = 100
  epoch = 20
  enableBackPropCheck = False
  learningRate = 0.1
  lambd = 0 if enableBackPropCheck else 0.01

  tempTestSize = 3000
  checkedProp = False
  for e in range(epoch):
    Logger.info("EPOCH", "epoch %d" % e)
    start = datetime.datetime.now()
    progress = 0
    tag = "ITER%d" % e
    # for j in range(10):
    while progress + batchSize < len(train) - tempTestSize:
      batch = [(train[i].pixels - 0)/256.0 for i in range(progress, progress + batchSize)]
      batchlabel = [train[i].labelArray for i in range(progress, progress + batchSize)]
      target = np.array(batchlabel).transpose()
      IL.setData(np.array(batch).transpose(), batchSize)
      IL.forward()
      cross_entrophy = np.multiply(target, np.log(SOFTMAX._output))
      loss = -np.sum(cross_entrophy, (0, 1)) / batchSize
      Logger.info(tag, "Loss: %f" % loss)

      SOFTMAX.backward(target)

      if enableBackPropCheck and e == 1 and not checkedProp:
        checkPropOutput = []
        IL.checkBackProp(0.0000001, checkPropOutput,
                         lambda : -np.sum(np.multiply(target, np.log(SOFTMAX._output)), (0, 1)) / batchSize)
        Logger.info("CHECK_BACKPROP", "Number of params is %d" % len(checkPropOutput))
        derivativeToCheck = np.array([checkPropOutput[i][0]/batchSize for i in range(len(checkPropOutput))])
        derivativeNumberred = np.array([checkPropOutput[i][1] for i in range(len(checkPropOutput))])
        diff = derivativeToCheck-derivativeNumberred
        error = math.sqrt(np.sum(np.multiply(diff, diff))) / \
                (math.sqrt(np.sum(np.multiply(derivativeNumberred, derivativeNumberred))) + \
                 math.sqrt(np.sum(np.multiply(derivativeToCheck, derivativeToCheck))))
        Logger.info("CHECK_BACKPROP", "checked result %f" % error)
        if (error > 0.00001):
          Logger.error("CHECK_BACKPROP", "back prop is not calculated right")
          sys.exit(-1)
        checkedProp = True

      FC1.updateParam(learningRate, lambd)
      FC2.updateParam(learningRate, lambd)
      FC3.updateParam(learningRate, lambd)
      OL.updateParam(learningRate, lambd)

      predict = np.argmax(SOFTMAX._output, (0))
      labels = np.argmax(np.array(batchlabel).transpose(), (0))
      right = 0
      for k in range(batchSize):
        if predict[k] == labels[k]:
          right += 1
      Logger.info(tag, "Accuracy: %d/%d" % (right, batchSize))
      progress += batchSize
      # break

    cost = datetime.datetime.now() - start
    Logger.warn("EPOCH", "epoch %d finished cost %d s" % (e, cost.total_seconds()))

  batch = [(train[i].pixels - 0)/256.0 for i in range(len(train) - tempTestSize, len(train))]
  batchlabel = [train[i].labelArray for i in range(len(train) - tempTestSize, len(train))]
  IL.setData(np.array(batch).transpose(), tempTestSize)
  IL.forward()
  predict = np.argmax(SOFTMAX._output, (0))
  labels = np.argmax(np.array(batchlabel).transpose(), (0))
  right = 0
  for i in range(tempTestSize):
    if predict[i] == labels[i]:
      right += 1
  Logger.info("RESULT", "\n\nTemporary Test Accuracy: %d/%d" % (right, tempTestSize))
