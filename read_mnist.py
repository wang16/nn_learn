import numpy as np
import random
#import Image
import os

class MnistData:
  def __init__(self, width, height, pixels, label):
    self.width = width
    self.height = height
    self.pixels = pixels
    self.label = label
    self.labelArray = np.zeros(10)
    self.labelArray[self.label-1] = 1

  def saveToBitmap(self, path):
    #im = Image.fromarray(np.reshape(self.pixels, [self.height, self.width]))
    #im.save(path)
    pass



# 1/testProportion of data will be test data
def readMnistData(imgs, labels, testProportion):
  train = []
  test = []
  with open(imgs, "rb") as imgFile, open(labels, "rb") as labelFile:
    imgHead = np.fromfile(imgFile, count=4, dtype='>u4')
    labelHead = np.fromfile(labelFile, count=2, dtype='>u4')
    assert imgHead[0] == 2051
    assert labelHead[0] == 2049
    assert imgHead[1] == labelHead[1]
    count = imgHead[1]
    height = imgHead[2]
    width = imgHead[3]
    testSize = count / testProportion
    rndThreshold = 1.0 / testProportion
    lbl = np.fromfile(labelFile, count=count, dtype=np.uint8)
    pixels = height * width
    for i in range(count):
      img = np.fromfile(imgFile, count=pixels, dtype=np.uint8)
      data = MnistData(width, height, img, lbl[i])
      if len(test) >= testSize:
        train.append(data)
      elif len(train) >= count - testSize:
        test.append(data)
      elif random.random() < rndThreshold:
        test.append(data)
        #data.saveToBitmap(os.path.join("test","%d-%d.bmp" % (i, lbl[i])))
      else:
        train.append(data)
  return train, test

if __name__ == "__main__":
  train, test = readMnistData("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 30)
  print "trainset: %d\ntestset: %d\n" % (len(train), len(test))